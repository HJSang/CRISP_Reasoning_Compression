"""Unit tests: verify liger vs non-liger JSD loss and entropy produce equivalent results.

Run with:
    python -m pytest test_opsd_jsd.py -v
    # or simply:
    python test_opsd_jsd.py
"""

import math

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Extract the four static methods under test so we don't need VERL imports.
# ---------------------------------------------------------------------------

class _JSD:
    """Namespace holding the four static methods copied from OPSDWorker."""

    @staticmethod
    def compute_jsd_loss(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        beta: float = 0.5,
        chunk_size: int = 512,
    ) -> tuple[torch.Tensor, int]:
        n_tokens = teacher_logits.shape[0]
        if n_tokens == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

        jsd_sum = torch.tensor(0.0, device=student_logits.device)
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            t_chunk = teacher_logits[start:end].float()
            s_chunk = student_logits[start:end].float()
            t_log_probs = F.log_softmax(t_chunk, dim=-1)
            s_log_probs = F.log_softmax(s_chunk, dim=-1)
            del t_chunk, s_chunk
            t_probs = t_log_probs.exp()
            s_probs = s_log_probs.exp()
            m_log_probs = (beta * t_probs + (1.0 - beta) * s_probs).clamp(min=1e-8).log()
            kl_t = (t_probs * (t_log_probs - m_log_probs)).sum(dim=-1)
            del t_probs, t_log_probs
            kl_s = (s_probs * (s_log_probs - m_log_probs)).sum(dim=-1)
            del s_probs, s_log_probs, m_log_probs
            jsd_chunk = beta * kl_t + (1.0 - beta) * kl_s
            jsd_sum = jsd_sum + jsd_chunk.sum()
            del kl_t, kl_s, jsd_chunk

        loss = jsd_sum / n_tokens
        return loss, n_tokens

    @staticmethod
    def compute_jsd_loss_liger(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        beta: float = 0.5,
        chunk_size: int = 256,
    ) -> tuple[torch.Tensor, int]:
        n_tokens = teacher_logits.shape[0]
        if n_tokens == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

        log_beta = math.log(beta) if beta > 0 else float("-inf")
        log_1m_beta = math.log(1.0 - beta) if beta < 1 else float("-inf")

        teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
        del teacher_logits

        jsd_sum = torch.tensor(0.0, device=student_logits.device)
        for i, t_chunk in enumerate(teacher_chunks):
            start = i * chunk_size
            end = start + t_chunk.shape[0]
            t_lp = F.log_softmax(t_chunk.float(), dim=-1)
            s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
            del t_chunk
            teacher_chunks[i] = None
            log_m = torch.logsumexp(
                torch.stack([t_lp + log_beta, s_lp + log_1m_beta], dim=0), dim=0,
            )
            kl_t = F.kl_div(log_m, t_lp, reduction="none", log_target=True).sum(dim=-1)
            del t_lp
            kl_s = F.kl_div(log_m, s_lp, reduction="none", log_target=True).sum(dim=-1)
            del s_lp, log_m
            jsd_chunk = beta * kl_t + (1.0 - beta) * kl_s
            jsd_sum = jsd_sum + jsd_chunk.sum()
            del kl_t, kl_s, jsd_chunk

        loss = jsd_sum / n_tokens
        return loss, n_tokens

    @staticmethod
    def compute_entropy(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        chunk_size: int = 512,
    ) -> tuple[float, float]:
        n_tokens = student_logits.shape[0]
        if n_tokens == 0:
            return 0.0, 0.0
        s_entropy_sum = 0.0
        t_entropy_sum = 0.0
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            s_log_probs = F.log_softmax(student_logits[start:end].float(), dim=-1)
            s_probs = s_log_probs.exp()
            s_entropy_sum += -(s_probs * s_log_probs).sum(dim=-1).sum().item()
            del s_probs, s_log_probs
            t_log_probs = F.log_softmax(teacher_logits[start:end].float(), dim=-1)
            t_probs = t_log_probs.exp()
            t_entropy_sum += -(t_probs * t_log_probs).sum(dim=-1).sum().item()
            del t_probs, t_log_probs
        return s_entropy_sum / n_tokens, t_entropy_sum / n_tokens

    @staticmethod
    def compute_entropy_liger(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        chunk_size: int = 256,
    ) -> tuple[float, float]:
        n_tokens = student_logits.shape[0]
        if n_tokens == 0:
            return 0.0, 0.0
        s_entropy_sum = 0.0
        t_entropy_sum = 0.0
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
            s_entropy_sum += -(s_lp.exp() * s_lp).sum(dim=-1).sum().item()
            del s_lp
            t_lp = F.log_softmax(teacher_logits[start:end].float(), dim=-1)
            t_entropy_sum += -(t_lp.exp() * t_lp).sum(dim=-1).sum().item()
            del t_lp
        return s_entropy_sum / n_tokens, t_entropy_sum / n_tokens


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return "cpu"


def _make_logits(n_tokens, vocab_size, seed=42, device="cpu", requires_grad=False):
    """Create random logits in bf16 (like real model outputs)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(n_tokens, vocab_size, generator=gen, device=device, dtype=torch.bfloat16)
    if requires_grad:
        logits = logits.float().requires_grad_(True)
    return logits


# ---------------------------------------------------------------------------
# JSD Loss: numerical equivalence
# ---------------------------------------------------------------------------

class TestJSDLossEquivalence:
    """Verify _compute_jsd_loss and _compute_jsd_loss_liger produce the same values."""

    @pytest.mark.parametrize("n_tokens,vocab_size", [
        (1, 10),       # single token
        (8, 100),      # small
        (64, 1000),    # medium
        (300, 5000),   # larger than chunk_size=256
        (512, 2000),   # exactly one chunk for original
    ])
    @pytest.mark.parametrize("beta", [0.5, 0.3, 0.7])
    def test_loss_values_match(self, n_tokens, vocab_size, beta):
        """Both implementations should produce the same scalar loss."""
        teacher = _make_logits(n_tokens, vocab_size, seed=42)
        student_orig = _make_logits(n_tokens, vocab_size, seed=99, requires_grad=True)
        student_liger = student_orig.data.clone().requires_grad_(True)

        loss_orig, nt_orig = _JSD.compute_jsd_loss(teacher, student_orig, beta=beta)
        loss_liger, nt_liger = _JSD.compute_jsd_loss_liger(teacher.clone(), student_liger, beta=beta)

        assert nt_orig == nt_liger == n_tokens
        torch.testing.assert_close(loss_orig, loss_liger, atol=1e-5, rtol=1e-4)

    def test_loss_zero_when_identical(self):
        """JSD(p || p) = 0 for any distribution."""
        logits = _make_logits(32, 500, seed=7)
        student = logits.float().requires_grad_(True)

        loss_orig, _ = _JSD.compute_jsd_loss(logits, student, beta=0.5)
        loss_liger, _ = _JSD.compute_jsd_loss_liger(logits.clone(), student, beta=0.5)

        assert loss_orig.item() < 1e-6
        assert loss_liger.item() < 1e-6

    def test_loss_nonnegative(self):
        """JSD is always >= 0."""
        teacher = _make_logits(64, 200, seed=1)
        student = _make_logits(64, 200, seed=2, requires_grad=True)

        loss_orig, _ = _JSD.compute_jsd_loss(teacher, student)
        loss_liger, _ = _JSD.compute_jsd_loss_liger(teacher.clone(), student)

        assert loss_orig.item() >= -1e-7
        assert loss_liger.item() >= -1e-7

    def test_symmetric_beta(self):
        """With beta=0.5, JSD(T||S) == JSD(S||T)."""
        a = _make_logits(32, 300, seed=10)
        b = _make_logits(32, 300, seed=20, requires_grad=True)

        loss_ts, _ = _JSD.compute_jsd_loss_liger(a.clone(), b, beta=0.5)
        # Swap teacher/student
        b2 = a.float().requires_grad_(True)
        loss_st, _ = _JSD.compute_jsd_loss_liger(
            _make_logits(32, 300, seed=20), b2, beta=0.5
        )
        torch.testing.assert_close(loss_ts, loss_st, atol=1e-5, rtol=1e-4)

    def test_empty_input(self):
        """Both should handle zero-length input gracefully."""
        teacher = torch.empty(0, 100)
        student = torch.empty(0, 100, requires_grad=True)

        loss_orig, nt_orig = _JSD.compute_jsd_loss(teacher, student)
        loss_liger, nt_liger = _JSD.compute_jsd_loss_liger(teacher, student)

        assert nt_orig == 0
        assert nt_liger == 0
        assert loss_orig.item() == 0.0
        assert loss_liger.item() == 0.0


# ---------------------------------------------------------------------------
# JSD Loss: gradient equivalence
# ---------------------------------------------------------------------------

class TestJSDGradientEquivalence:
    """Verify gradients w.r.t. student_logits match between implementations."""

    @pytest.mark.parametrize("n_tokens,vocab_size", [
        (8, 100),
        (64, 500),
        (300, 1000),
    ])
    @pytest.mark.parametrize("beta", [0.5, 0.3])
    def test_gradient_match(self, n_tokens, vocab_size, beta):
        """Gradients of loss w.r.t. student_logits should be close."""
        teacher = _make_logits(n_tokens, vocab_size, seed=42)

        student_orig = _make_logits(n_tokens, vocab_size, seed=99, requires_grad=True)
        student_liger = student_orig.data.clone().requires_grad_(True)

        loss_orig, _ = _JSD.compute_jsd_loss(teacher, student_orig, beta=beta)
        loss_orig.backward()

        loss_liger, _ = _JSD.compute_jsd_loss_liger(teacher.clone(), student_liger, beta=beta)
        loss_liger.backward()

        assert student_orig.grad is not None
        assert student_liger.grad is not None
        torch.testing.assert_close(student_orig.grad, student_liger.grad, atol=1e-4, rtol=1e-3)

    def test_gradient_flows(self):
        """Ensure both versions produce non-zero gradients."""
        teacher = _make_logits(16, 200, seed=1)
        student_orig = _make_logits(16, 200, seed=2, requires_grad=True)
        student_liger = student_orig.data.clone().requires_grad_(True)

        loss_orig, _ = _JSD.compute_jsd_loss(teacher, student_orig)
        loss_orig.backward()
        assert student_orig.grad is not None
        assert student_orig.grad.abs().sum() > 0

        loss_liger, _ = _JSD.compute_jsd_loss_liger(teacher.clone(), student_liger)
        loss_liger.backward()
        assert student_liger.grad is not None
        assert student_liger.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Entropy: numerical equivalence
# ---------------------------------------------------------------------------

class TestEntropyEquivalence:
    """Verify _compute_entropy and _compute_entropy_liger produce the same values."""

    @pytest.mark.parametrize("n_tokens,vocab_size", [
        (1, 10),
        (8, 100),
        (64, 1000),
        (300, 5000),
    ])
    def test_entropy_values_match(self, n_tokens, vocab_size):
        """Both implementations should produce the same entropy values."""
        student = _make_logits(n_tokens, vocab_size, seed=42)
        teacher = _make_logits(n_tokens, vocab_size, seed=99)

        s_ent_orig, t_ent_orig = _JSD.compute_entropy(student, teacher)
        s_ent_liger, t_ent_liger = _JSD.compute_entropy_liger(student, teacher)

        assert abs(s_ent_orig - s_ent_liger) < 1e-4, (
            f"Student entropy mismatch: {s_ent_orig} vs {s_ent_liger}"
        )
        assert abs(t_ent_orig - t_ent_liger) < 1e-4, (
            f"Teacher entropy mismatch: {t_ent_orig} vs {t_ent_liger}"
        )

    def test_entropy_nonnegative(self):
        """Entropy is always >= 0."""
        student = _make_logits(32, 200, seed=10)
        teacher = _make_logits(32, 200, seed=20)

        s_ent, t_ent = _JSD.compute_entropy_liger(student, teacher)
        assert s_ent >= -1e-7
        assert t_ent >= -1e-7

    def test_entropy_uniform_distribution(self):
        """Uniform distribution has entropy = log(V)."""
        V = 100
        # Uniform logits = all zeros
        uniform = torch.zeros(16, V)

        s_ent, t_ent = _JSD.compute_entropy_liger(uniform, uniform)
        expected = math.log(V)
        assert abs(s_ent - expected) < 1e-4, f"Expected {expected}, got {s_ent}"
        assert abs(t_ent - expected) < 1e-4, f"Expected {expected}, got {t_ent}"

    def test_entropy_peaked_distribution(self):
        """Very peaked distribution has entropy near 0."""
        V = 100
        # One logit much larger than the rest
        peaked = torch.full((16, V), -100.0)
        peaked[:, 0] = 100.0

        s_ent, t_ent = _JSD.compute_entropy_liger(peaked, peaked)
        assert s_ent < 0.01, f"Expected near-zero entropy, got {s_ent}"

    def test_entropy_empty_input(self):
        """Both should handle zero-length input."""
        student = torch.empty(0, 100)
        teacher = torch.empty(0, 100)

        s_orig, t_orig = _JSD.compute_entropy(student, teacher)
        s_liger, t_liger = _JSD.compute_entropy_liger(student, teacher)

        assert s_orig == 0.0
        assert t_orig == 0.0
        assert s_liger == 0.0
        assert t_liger == 0.0


# ---------------------------------------------------------------------------
# Chunk boundary tests
# ---------------------------------------------------------------------------

class TestChunkBoundaries:
    """Verify results don't depend on chunk_size."""

    @pytest.mark.parametrize("chunk_size", [1, 7, 16, 64, 256, 512, 1024])
    def test_jsd_invariant_to_chunk_size(self, chunk_size):
        """JSD loss should be the same regardless of chunk_size."""
        teacher = _make_logits(50, 200, seed=42)
        student = _make_logits(50, 200, seed=99, requires_grad=True)

        loss_ref, _ = _JSD.compute_jsd_loss_liger(
            teacher.clone(), student, beta=0.5, chunk_size=50  # single chunk
        )
        student2 = student.data.clone().requires_grad_(True)
        loss_chunked, _ = _JSD.compute_jsd_loss_liger(
            teacher.clone(), student2, beta=0.5, chunk_size=chunk_size,
        )
        torch.testing.assert_close(loss_ref, loss_chunked, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("chunk_size", [1, 7, 16, 64, 256, 512])
    def test_entropy_invariant_to_chunk_size(self, chunk_size):
        """Entropy should be the same regardless of chunk_size."""
        student = _make_logits(50, 200, seed=42)
        teacher = _make_logits(50, 200, seed=99)

        s_ref, t_ref = _JSD.compute_entropy_liger(student, teacher, chunk_size=50)
        s_chunked, t_chunked = _JSD.compute_entropy_liger(
            student, teacher, chunk_size=chunk_size
        )
        assert abs(s_ref - s_chunked) < 1e-5
        assert abs(t_ref - t_chunked) < 1e-5


# ---------------------------------------------------------------------------
# Edge-case beta values
# ---------------------------------------------------------------------------

class TestBetaEdgeCases:
    """Test JSD with extreme beta values."""

    def test_beta_0_reduces_to_kl(self):
        """beta=0 => JSD = KL(p_S || p_S) = 0 (trivially). Actually beta=0 => m=p_S."""
        teacher = _make_logits(16, 100, seed=1)
        student = _make_logits(16, 100, seed=2, requires_grad=True)
        student2 = student.data.clone().requires_grad_(True)

        # With beta=0: m = (1-0)*p_S = p_S, so KL(p_S||m) = 0
        # and JSD = 0*KL(p_T||m) + 1*0 = 0... actually:
        # JSD = beta*KL(T||m) + (1-beta)*KL(S||m)
        #     = 0*KL(T||p_S) + 1*KL(S||p_S)
        #     = 0  (since m=p_S and KL(S||S)=0)
        loss_orig, _ = _JSD.compute_jsd_loss(teacher, student, beta=0.0)
        loss_liger, _ = _JSD.compute_jsd_loss_liger(teacher.clone(), student2, beta=0.0)

        # Both should give KL(S||S) = 0
        assert loss_orig.item() < 1e-5
        assert loss_liger.item() < 1e-5

    def test_beta_1_reduces_to_kl(self):
        """beta=1 => m = p_T, JSD = KL(T||T) + 0 = 0."""
        teacher = _make_logits(16, 100, seed=1)
        student = _make_logits(16, 100, seed=2, requires_grad=True)
        student2 = student.data.clone().requires_grad_(True)

        loss_orig, _ = _JSD.compute_jsd_loss(teacher, student, beta=1.0)
        loss_liger, _ = _JSD.compute_jsd_loss_liger(teacher.clone(), student2, beta=1.0)

        assert loss_orig.item() < 1e-5
        assert loss_liger.item() < 1e-5

    def test_asymmetric_beta_values_match(self):
        """Non-0.5 beta values should still match between implementations."""
        for beta in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]:
            teacher = _make_logits(32, 200, seed=42)
            student_orig = _make_logits(32, 200, seed=99, requires_grad=True)
            student_liger = student_orig.data.clone().requires_grad_(True)

            loss_orig, _ = _JSD.compute_jsd_loss(teacher, student_orig, beta=beta)
            loss_liger, _ = _JSD.compute_jsd_loss_liger(teacher.clone(), student_liger, beta=beta)

            torch.testing.assert_close(
                loss_orig, loss_liger, atol=1e-5, rtol=1e-4,
                msg=f"Mismatch at beta={beta}",
            )


# ---------------------------------------------------------------------------
# bf16 input handling
# ---------------------------------------------------------------------------

class TestBf16Inputs:
    """Verify both paths handle bf16 inputs (like real model outputs)."""

    def test_jsd_with_bf16(self):
        """bf16 inputs should produce close results."""
        teacher = _make_logits(32, 500, seed=42)  # bf16
        student_f32 = teacher.float().requires_grad_(True)

        loss_orig, _ = _JSD.compute_jsd_loss(teacher, student_f32)
        student_f32_2 = teacher.float().requires_grad_(True)
        loss_liger, _ = _JSD.compute_jsd_loss_liger(teacher.clone(), student_f32_2)

        # Same teacher=student => both should be ~0
        assert loss_orig.item() < 1e-4
        assert loss_liger.item() < 1e-4

    def test_entropy_with_bf16(self):
        """bf16 inputs should produce valid entropy."""
        student = _make_logits(32, 500, seed=42)  # bf16
        teacher = _make_logits(32, 500, seed=99)  # bf16

        s_orig, t_orig = _JSD.compute_entropy(student, teacher)
        s_liger, t_liger = _JSD.compute_entropy_liger(student, teacher)

        assert abs(s_orig - s_liger) < 1e-3
        assert abs(t_orig - t_liger) < 1e-3


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
