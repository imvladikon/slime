import ray

from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.misc import should_run_periodic_action


def _update_rollout_weights(args, actor_model, rollout_manager, *, refresh_snapshot: bool) -> None:
    """Push actor weights and, in debug mode, prove the current state round-trips."""
    actor_model.update_weights()
    if not args.check_weight_update_equal:
        return
    if refresh_snapshot:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))
        actor_model.update_weights()
    ray.get(rollout_manager.check_weights.remote(action="compare"))


def _frozen_weight_fingerprint(responses, prefixes: list[str]):
    """Select stable per-rank checksums for frozen rollout-only parameters."""
    fingerprint = []
    matched = 0
    for engine_index, response in enumerate(responses):
        if not response.get("success") or not response.get("ranks"):
            raise RuntimeError(f"SGLang checksum failed for engine {engine_index}: {response}")
        for rank in response["ranks"]:
            selected = tuple(
                sorted(
                    (name, checksum)
                    for name, checksum in rank["checksums"].items()
                    if name.startswith(tuple(prefixes))
                )
            )
            matched += len(selected)
            fingerprint.append(selected)
    if matched == 0:
        raise RuntimeError(f"no SGLang weights matched frozen prefixes {prefixes}")
    return tuple(fingerprint)


def _checksum_frozen_weights(args, rollout_manager):
    prefixes = args.weight_checker_frozen_prefix
    if not prefixes or not args.check_weight_update_equal:
        return None
    responses = ray.get(
        rollout_manager.check_weights.remote(action="checksum", skip_prefixes=[])
    )
    return _frozen_weight_fingerprint(responses, prefixes)


def train(args):
    configure_logger()
    release_train = args.release_train

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    _update_rollout_weights(args, actor_model, rollout_manager, refresh_snapshot=False)
    frozen_weight_fingerprint = _checksum_frozen_weights(args, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        _update_rollout_weights(args, actor_model, rollout_manager, refresh_snapshot=True)

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    if frozen_weight_fingerprint is not None:
        final_fingerprint = _checksum_frozen_weights(args, rollout_manager)
        if final_fingerprint != frozen_weight_fingerprint:
            raise RuntimeError("frozen SGLang rollout weights changed during training")

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
