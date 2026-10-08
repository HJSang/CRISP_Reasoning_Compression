"""Independent teacher-input oracle using public synthetic text and fixed token tapes."""

from types import SimpleNamespace

import torch

from miles.backends.training_utils.data.opsd import teacher_batch
from miles.backends.training_utils.loss.hub.opsd import response_logits
from miles.utils.opsd_prompts import make_opsd_prefixes
from tests.opsd_reference import load_author_collator, load_author_input_builder


TOKENIZER_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
FEATURES = [
    {"problem": "What is 2 + 2?", "solution": "Add to obtain 4."},
    {"problem": "求 7−3。", "solution": "7−3=4.\nTherefore \\boxed{4}."},
    {"problem": "Solve $x^2=9$ over the reals.", "solution": "Factor: $(x-3)(x+3)=0$. Thus $x=\\pm3$."},
    {"problem": "What does `sum([1, 2, 3])` return?\n", "solution": "```python\n  1 + 2 + 3\n```\n\t6\n\n"},
    {"problem": "Use the words café, naive and naïve in a count.", "solution": "café\ncafe\u0301\nnaïve\nThree lines."},
    {
        "problem": "Add the first ten positive integers.",
        "solution": "Pair the smallest with the largest. " * 128 + "55.",
    },
]


def response_tapes(tokenizer):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    answer = encode("The answer is \\boxed{4}.") + [tokenizer.eos_token_id]
    # A model may choose adjacent IDs that merge if their decoded text is encoded.
    split = encode("a") + encode("b") + [tokenizer.eos_token_id]
    assert encode(tokenizer.decode(split, skip_special_tokens=False)) != split
    return {
        "answer_eos": answer,
        "noncanonical_ids": split,
        "thinking_boundary": encode("</think>\n\n4") + [tokenizer.eos_token_id],
        "completion_padding": answer + [tokenizer.pad_token_id] * 3,
    }


def compare_inputs(tokenizer, reference_dir, features, responses, *, max_length=4096):
    """Compare the actual author assembly block with Miles' teacher_batch.

    Responses are fixed IDs, not newly generated text. Author completion padding
    is supplied explicitly; Miles receives only each tape's original length.
    """
    collator = load_author_collator(reference_dir)(tokenizer, max_length=max_length, reason_first=False)
    original = collator(features)
    width = max(map(len, responses))
    completions = torch.full((len(features), width), tokenizer.pad_token_id, dtype=torch.long)
    for row, response in zip(completions, responses, strict=True):
        row[: len(response)] = torch.tensor(response)
    generated = torch.cat((original["student_prompts"], completions), dim=1)
    original = load_author_input_builder(reference_dir)(
        SimpleNamespace(processing_class=tokenizer), original, generated, generated.ne(tokenizer.pad_token_id).long()
    )
    prefixes = [make_opsd_prefixes(tokenizer, **feature) for feature in features]
    student_tokens = [
        torch.tensor([*prefix.student_ids, *response]) for prefix, response in zip(prefixes, responses, strict=True)
    ]
    data = {
        "tokens": student_tokens,
        "teacher_prompt_ids": [list(prefix.teacher_ids) for prefix in prefixes],
        "response_lengths": list(map(len, responses)),
        "total_lengths": list(map(len, student_tokens)),
        "loss_masks": [[int(token != tokenizer.pad_token_id) for token in response] for response in responses],
        "sample_indices": list(range(len(features))),
    }
    teacher = teacher_batch(
        data, seq_length=8192, local_vocab_width=1, cache_bytes=2**20, microbatch_bytes=2**20, micro_batch_size=1
    )
    teacher_width = original["teacher_prompt_length"]
    results = []
    for i, (prefix, response, full) in enumerate(zip(prefixes, responses, teacher["tokens"], strict=True)):
        author_length = int(original["teacher_prompt_lengths_per_example"][i])
        author_prefix = original["teacher_prompts"][i, :author_length].tolist()
        author_full = original["teacher_input_ids"][i, : teacher_width + len(response)].tolist()
        author_response = original["teacher_input_ids"][i, teacher_width : teacher_width + len(response)].tolist()
        teacher_prompt_ids = list(prefix.teacher_ids)
        full_ids = full.tolist()
        valid_labels = original["labels"][i, original["student_prompt_length"] :][: len(response)].ne(-100).tolist()
        assert author_response == response == full_ids[len(teacher_prompt_ids) :]
        assert valid_labels == data["loss_masks"][i]
        # Exercise the production causal shift. Each row index predicts the next ID.
        positions = torch.arange(len(full_ids)).view(1, -1, 1)
        prediction_rows = next(response_logits(positions, [len(full_ids)], [len(response)])).flatten().tolist()
        assert [full_ids[pos + 1] for pos in prediction_rows] == response
        prompt_padding = teacher_width - author_length
        normalized_author = author_full[:author_length] + author_full[teacher_width:]
        results.append(
            {
                "student_prefix_ids_equal": list(prefix.student_ids)
                == original["student_prompts"][i, : len(prefix.student_ids)].tolist(),
                "teacher_prefix_ids_equal": teacher_prompt_ids == author_prefix,
                "teacher_full_ids_equal": full_ids == author_full,
                "teacher_ids_equal_after_removing_batch_padding": full_ids == normalized_author,
                "response_ids_equal": True,
                "response_loss_masks_equal": True,
                "teacher_prompt_length": len(teacher_prompt_ids),
                "student_prompt_length": len(prefix.student_ids),
                "response_length": len(response),
                "author_teacher_padding_gap": prompt_padding,
                "author_first_prediction_row": teacher_width - 1,
                "miles_first_prediction_row": prediction_rows[0],
                "masked_pad_ids_inside_teacher_prefix": teacher_prompt_ids.count(tokenizer.pad_token_id),
                "student_boundary_ids": list(prefix.student_ids[-10:]),
                "teacher_boundary_ids": teacher_prompt_ids[-10:],
                "response_ids": response,
            }
        )
    return results


def run_token_sanity(tokenizer, reference_dir):
    native_pad = tokenizer.pad_token_id
    single = []
    for alias in (False, True):
        tokenizer.pad_token_id = tokenizer.eos_token_id if alias else native_pad
        for i, feature in enumerate(FEATURES):
            for name, response in response_tapes(tokenizer).items():
                row = compare_inputs(tokenizer, reference_dir, [feature], [response])[0]
                row.update(feature=i, tape=name, pad_equals_eos=alias)
                assert row["student_prefix_ids_equal"] and row["teacher_prefix_ids_equal"]
                assert row["teacher_full_ids_equal"]
                single.append(row)
    tokenizer.pad_token_id = native_pad
    responses = list(response_tapes(tokenizer).values())
    mixed = compare_inputs(tokenizer, reference_dir, FEATURES[:4], responses)
    assert all(row["teacher_ids_equal_after_removing_batch_padding"] for row in mixed)
    assert any(not row["teacher_full_ids_equal"] for row in mixed)
    # Deliberately over the author's configured token limit: IDs must differ.
    truncated = compare_inputs(tokenizer, reference_dir, [FEATURES[-1]], [responses[0]], max_length=256)[0]
    assert not truncated["teacher_prefix_ids_equal"]
    return {
        "tokenizer_revision": TOKENIZER_REVISION,
        "native_pad_id": native_pad,
        "eos_id": tokenizer.eos_token_id,
        "single": single,
        "mixed_batch": mixed,
        "truncation": truncated,
    }
