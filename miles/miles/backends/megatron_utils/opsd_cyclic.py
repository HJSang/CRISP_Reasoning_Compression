"""Cyclic OPSD lifecycle hooks using the actor's existing snapshot and Adam owners."""

from miles.utils.opsd_cyclic import cyclic_step


def initialize_cyclic(actor):
    actor.opsd_cyclic_completed_updates = 0
    actor.opsd_cyclic_teacher_source_update = 0
    if getattr(actor.args, "opsd_cyclic_teacher_policy", None) == "cycle_refresh":
        # Actor was backed up before loading the original teacher. Cycle one must
        # use its entry student even when that student is an explicit warm source.
        actor.weights_backuper.copy(src_tag="actor", dst_tag="teacher")


def begin_cyclic_step(actor, rollout_data, *, rollout_id, num_optimizer_steps):
    # Import only on the GPU runtime path; schedule/snapshot tests remain CPU-only.
    from miles.backends.megatron_utils.optimizer_utils import reset_optimizer_state

    if rollout_id != actor.opsd_cyclic_completed_updates or num_optimizer_steps != 1:
        raise ValueError("Cyclic OPSD requires exactly one sequential successful update per fresh rollout")
    step = cyclic_step(actor.args, rollout_id)
    if actor.opsd_cyclic_teacher_source_update != step.teacher_source_update:
        raise ValueError("Cyclic OPSD teacher is not the required cycle-entry source")
    for tokens, response_length, prefix in zip(
        rollout_data["tokens"], rollout_data["response_lengths"], rollout_data["teacher_prompt_ids"], strict=True
    ):
        student_prefix = tokens[:len(tokens) - response_length].tolist()
        if (list(prefix) == student_prefix) != (step.context == "none"):
            raise ValueError("Cyclic OPSD teacher prefix disagrees with the scheduled PI/OPD phase")
    if step.phase_start:
        reset_optimizer_state(
            actor.optimizer,
            stream_optimizer_state_to_disk=actor.args.stream_optimizer_state_to_disk,
            chunked_optimizer_state_offload=actor.args.chunked_optimizer_state_offload,
        )


def complete_cyclic_step(actor, *, rollout_id):
    """Called only after a successful optimizer update and the actor weight backup."""
    if rollout_id != actor.opsd_cyclic_completed_updates:
        raise ValueError("Cyclic OPSD cannot complete an out-of-order update")
    step = cyclic_step(actor.args, rollout_id)
    refresh = step.cycle_end and actor.args.opsd_cyclic_teacher_policy == "cycle_refresh"
    if refresh:
        actor.weights_backuper.copy(src_tag="actor", dst_tag="teacher")
        actor.opsd_cyclic_teacher_source_update = rollout_id + 1
    actor.opsd_cyclic_completed_updates = rollout_id + 1
    return {
        "train/step": rollout_id,
        **step.metrics(),
        "train/cyclic_teacher_refreshed": int(refresh),
        "train/cyclic_completed_updates": actor.opsd_cyclic_completed_updates,
    }
