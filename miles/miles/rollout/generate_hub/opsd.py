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
from miles.utils.opsd_prompts import make_opsd_prefixes, validate_response_ids


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args, sample = input.args, input.sample
    if not isinstance(sample.prompt, str) or sample.multimodal_inputs or sample.response_length:
        raise ValueError("OPSD requires a raw text problem and a fresh single-turn sample")
    solution = None if input.evaluation else sample.metadata["solution"]
    if not input.evaluation and (not isinstance(solution, str) or not solution.strip()):
        raise ValueError("OPSD training requires a nonempty metadata.solution")
    prefixes = make_opsd_prefixes(input.state.tokenizer, problem=sample.prompt, solution=solution)
    # Reserve the entire requested response for both conditions; no teacher-only truncation.
    response_cap = input.sampling_params["max_new_tokens"]
    for prefix in (prefixes.student_ids, prefixes.teacher_ids):
        if prefix and len(prefix) + response_cap > args.seq_length:
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
    # Match the author's labels[labels == pad_token_id] = -100, including EOS
    # if this tokenizer aliases EOS to PAD. Other mask semantics are a new recipe.
    response = sample.tokens[len(prefixes.student_ids) :]
    sample.loss_mask = [int(token != input.state.tokenizer.pad_token_id) for token in response]
    if not input.evaluation:
        sample.reward = 0.0  # OPSD uses distributions, not a reward model.
    return GenerateFnOutput(samples=sample)
