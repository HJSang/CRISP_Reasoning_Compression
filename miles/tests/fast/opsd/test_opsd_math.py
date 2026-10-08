"""Independent author-function, analytic-gradient and distributed OPSD checks.

Set OPSD_REFERENCE_DIR to an unmodified ae7d2519 author checkout to run the
independent oracle cases. No trainer dependencies or network calls are needed.
"""

import ast
import hashlib
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from miles.backends.training_utils.loss.hub.opsd_math import OPSDLossConfig, opsd_per_token_loss, vocab_log_softmax

_REFERENCE_SHA256 = "aa11fc2a3f3cd814da37db38c6fb2351cab7f98a5a572d808a9b75de16097c9d"


@pytest.fixture(scope="module")
def author_loss():
    reference_dir = os.environ.get("OPSD_REFERENCE_DIR")
    if reference_dir is None:
        pytest.skip("Set OPSD_REFERENCE_DIR for independent author-code parity")
    source = (Path(reference_dir) / "opsd_trainer.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == _REFERENCE_SHA256, "Reference source changed"
    tree = ast.parse(source)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "generalized_jsd_loss"]
    assert len(functions) == 1
    function = functions[0]
    function.decorator_list = []  # Extract the unchanged body, without importing the full TRL trainer.
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "original_opsd_loss", "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("temperature", [1.0, 1.1])
@pytest.mark.parametrize("clip", [None, 0.05])
def test_author_loss_and_gradient(author_loss, dtype, beta, temperature, clip):
    generator = torch.Generator().manual_seed(42)
    # Striding the vocabulary dimension exercises noncontiguous inputs.
    student = (torch.randn(2, 4, 26, generator=generator, dtype=dtype)[..., ::2] * 3).requires_grad_()
    teacher = torch.randn(2, 4, 13, generator=generator, dtype=dtype, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, -100], [1, -100, -100, -100]])
    expected = author_loss(student, teacher.detach(), labels, beta, temperature, token_clip=clip)
    expected_gradient = torch.autograd.grad(expected, student)[0]
    q = F.log_softmax(teacher.reshape(-1, 13) / temperature, dim=-1)
    actual_tokens = opsd_per_token_loss(
        student.reshape(-1, 13),
        q,
        vocab_size=13,
        config=OPSDLossConfig(beta=beta, temperature=temperature, token_clip=clip),
    )
    actual = actual_tokens[labels.flatten() != -100].mean()
    actual_gradient = torch.autograd.grad(actual, student, retain_graph=True)[0]
    atol, rtol = (1e-10, 1e-8) if dtype == torch.float64 else (2e-6, 2e-5)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=atol, rtol=rtol)
    assert torch.autograd.grad(actual, teacher, allow_unused=True)[0] is None


def test_unclipped_forward_kl_analytic_gradient_and_padded_vocab():
    generator = torch.Generator().manual_seed(9)
    logits = torch.randn(3, 12, generator=generator, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(3, 9, generator=generator, dtype=torch.float64).log_softmax(-1)
    config = OPSDLossConfig(temperature=1.1, token_clip=None)
    loss = opsd_per_token_loss(logits, teacher, vocab_size=9, config=config).mean()
    gradient = torch.autograd.grad(loss, logits)[0]
    expected = ((logits[:, :9] / 1.1).softmax(-1) - teacher.exp()) / (3 * 1.1)
    torch.testing.assert_close(gradient[:, :9], expected)
    assert torch.count_nonzero(gradient[:, 9:]) == 0


def test_clip_is_per_vocabulary_entry_and_can_produce_negative_sum():
    student = torch.tensor([[0.01, 0.99]], dtype=torch.float64).log().requires_grad_()
    teacher = torch.tensor([[0.5, 0.5]], dtype=torch.float64).log()
    loss = opsd_per_token_loss(student, teacher, vocab_size=2, config=OPSDLossConfig(temperature=1))
    assert loss.item() < 0  # Summing first and clipping a token KL cannot do this.
    assert torch.autograd.gradcheck(
        lambda x: opsd_per_token_loss(x, teacher, vocab_size=2, config=OPSDLossConfig(temperature=1)), (student,)
    )


@pytest.mark.parametrize("kwargs", [{"beta": -1}, {"beta": float("nan")}, {"temperature": 0}, {"token_clip": -1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        OPSDLossConfig(**kwargs)


def _tp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        generator = torch.Generator().manual_seed(5)
        full = torch.randn(4, 8, generator=generator, dtype=torch.float64) * 3
        teacher = torch.randn(4, 8, generator=generator, dtype=torch.float64)
        for vocab in [7, 3]:  # Also cover an entirely padded TP shard.
            for beta in [0.0, 0.4, 1.0]:
                config = OPSDLossConfig(beta=beta)
                local = full[:, rank * 4 : (rank + 1) * 4].clone().requires_grad_()
                q = vocab_log_softmax(
                    teacher[:, rank * 4 : (rank + 1) * 4],
                    vocab_size=vocab,
                    temperature=config.temperature,
                    group=dist.group.WORLD,
                )
                got = opsd_per_token_loss(local, q, vocab_size=vocab, config=config, group=dist.group.WORLD).mean()
                grad = torch.autograd.grad(got, local)[0]
                dense = full.clone().requires_grad_()
                q_dense = (teacher[:, :vocab] / config.temperature).log_softmax(-1)
                expected = opsd_per_token_loss(dense, q_dense, vocab_size=vocab, config=config).mean()
                expected_grad = torch.autograd.grad(expected, dense)[0][:, rank * 4 : (rank + 1) * 4]
                torch.testing.assert_close(got, expected, atol=1e-10, rtol=1e-8)
                torch.testing.assert_close(grad, expected_grad, atol=1e-10, rtol=1e-8)
    finally:
        dist.destroy_process_group()


def test_tp2_loss_and_backward_match_dense_with_padding(tmp_path):
    mp.spawn(_tp_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)
