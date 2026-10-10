"""Single-turn token-in/token-out OPSD generation.

Dataset rows use the raw problem as prompt and metadata.solution as privileged
information. Do not apply a dataset-level chat template. Evaluation needs no PI.
"""

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.utils.http_utils import post
from miles.utils.opsd_prompts import make_opsd_prefixes, make_study_prefixes, validate_response_ids


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args, sample = input.args, input.sample
    if not isinstance(sample.prompt, str) or sample.multimodal_inputs or sample.response_length:
        raise ValueError("OPSD requires a raw text problem and a fresh single-turn sample")
    mode = getattr(args, "opsd_context", "original")
    if mode == "original":
        solution = None if input.evaluation else sample.metadata["solution"]
        if not input.evaluation and (not isinstance(solution, str) or not solution.strip()):
            raise ValueError("OPSD training requires a nonempty metadata.solution")
        prefixes = make_opsd_prefixes(input.state.tokenizer, problem=sample.prompt, solution=solution)
    else:
        key = {"answer": "answer", "worked": "solution", "unrelated": "unrelated_context"}.get(mode)
        context = None
        if not input.evaluation:
            context = sample.metadata[key] if key else ("" if mode == "empty" else None)
        if key and not input.evaluation and (not isinstance(context, str) or not context.strip()):
            raise ValueError(f"OPSD {mode} context must be nonempty text")
        prefixes = make_study_prefixes(
            input.state.tokenizer, problem=sample.prompt, context=context, task=getattr(args, "opsd_task", "math")
        )
    pad_token_id = input.state.tokenizer.pad_token_id
    # The author masks every PAD ID, including IDs inside a rendered chat prefix.
    # THD packing has no per-token attention holes: reject those inputs rather
    # than silently changing context (e.g. EOS-as-PAD masks Qwen chat delimiters).
    if any(pad_token_id in prefix for prefix in (prefixes.student_ids, prefixes.teacher_ids)):
        raise ValueError(
            "OPSD packed prompts must not contain PAD IDs; use distinct PAD/EOS and filter literal PAD text"
        )
    # Reserve the entire requested response for both conditions; no teacher-only truncation.
    response_cap = input.sampling_params["max_new_tokens"]
    sequence_cap = (getattr(args, "eval_max_context_len", None) or args.seq_length) if input.evaluation else args.seq_length
    for prefix in (prefixes.student_ids, prefixes.teacher_ids):
        if prefix and len(prefix) + response_cap > sequence_cap:
            raise ValueError("OPSD prefix plus response cap exceeds --seq-length; filter the dataset first")
    sample.teacher_prompt_ids = list(prefixes.teacher_ids) if prefixes.teacher_ids else None
    payload, halt_status = compute_request_payload(
        args,
        input_ids=list(prefixes.student_ids),
        sampling_params=input.sampling_params,
        evaluation=input.evaluation,
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    output = await post(url, payload, headers=compute_routing_headers(args, sample))
    await update_sample_from_response(args, sample, payload=payload, output=output)
    validate_response_ids(
        tokens=sample.tokens, student_prefix=prefixes.student_ids, response_length=sample.response_length
    )
    # Match the author's labels[labels == pad_token_id] = -100. Packed attention
    # can ignore trailing PAD for the loss, but cannot reproduce interior holes.
    response = sample.tokens[len(prefixes.student_ids) :]
    if pad_token_id in response and any(token != pad_token_id for token in response[response.index(pad_token_id) :]):
        raise ValueError("OPSD packed responses permit PAD only at the end; interior PAD changes teacher attention")
    sample.loss_mask = [int(token != pad_token_id) for token in response]
    if not input.evaluation:
        sample.reward = 0.0  # OPSD uses distributions, not a reward model.
    return GenerateFnOutput(samples=sample)
