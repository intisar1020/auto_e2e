from .base import BasePlanner
from .flow_matching_planner import FlowMatchingPlanner
from .bezier_planner import BezierPlanner
from .gru_planner import GRUPlanner
from .pad_query_planner import PadQueryPlanner

PLANNER_REGISTRY = {
    "flow_matching": FlowMatchingPlanner,
    "bezier": BezierPlanner,
    "gru": GRUPlanner,
    "query": PadQueryPlanner,
}


def build_planner(planner_mode, **kwargs):
    """Construct a planner by name.

    Forwards ``**kwargs`` directly to the planner __init__; planners ignore
    args they do not understand by accepting only what they need (the
    registry does not pre-filter, so callers must pass kwargs the chosen
    planner accepts).
    """
    if planner_mode not in PLANNER_REGISTRY:
        raise ValueError(
            f"Unknown planner_mode {planner_mode!r}. "
            f"Available: {sorted(PLANNER_REGISTRY)}."
        )
    if planner_mode == "flow_matching":
        # train_il regresses forward()'s Euler-from-noise rollout, which is not
        # the flow-matching objective (velocity MSE against x1-x0) and drives the
        # model to the conditional mean. No training path computes the
        # flow-matching velocity objective yet.
        import warnings

        warnings.warn(
            "planner_mode='flow_matching' is NOT correctly trainable via the "
            "current train_il loop (it L1-regresses an Euler-from-noise rollout, "
            "not the velocity-MSE flow objective). Use 'bezier' unless/until "
            "compute_planner_loss is wired in.",
            RuntimeWarning,
            stacklevel=2,
        )
    return PLANNER_REGISTRY[planner_mode](**kwargs)


__all__ = [
    "BasePlanner",
    "FlowMatchingPlanner",
    "GRUPlanner",
    "PadQueryPlanner",
    "BezierPlanner",
    "PLANNER_REGISTRY",
    "build_planner",
]
