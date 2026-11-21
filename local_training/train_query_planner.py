"""Train and compare the GRU and trajectory-query planners locally.

This script intentionally stays close to the production KITScenes Stage B
recipe while avoiding the packed WebDataset and Ray infrastructure:

* raw KITScenes data is read directly with ``KitScenesDataset``;
* the R50/BEVFormer T8 camera path is loaded from the released checkpoint;
* camera BEV parameters remain frozen;
* T8 history, native Front, semantic map, route, and camera projections are
  reconstructed with the same contracts used by production training;
* the trajectory XY and route reconstruction objectives are reused;
* GRU and query planners both start from freshly initialized planner weights;
* the query planner is the PAD-derived implementation from
  ``model_components.trajectory_planning.pad_query_planner``;
* only the planner architecture differs between the two runs.

The local subset is useful for comparing planners, not for reproducing the full
KITScenes benchmark.

Example:

    python local_training/train_query_planner.py --planner gru
    python local_training/train_query_planner.py --planner query

The default training mode freezes everything except the planner so the
comparison measures the planner architecture. Use ``--trainable reactive`` to
also train the navigation encoder, fusion, and route head as in KITScenes Stage
B.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "Model"
for import_root in (REPO_ROOT, MODEL_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data_parsing.kit_scenes import KitScenesDataset  # noqa: E402
from model_components.auto_e2e import AutoE2E  # noqa: E402
from model_components.trajectory_planning import (  # noqa: E402
    GRUPlanner,
    PadQueryPlanner,
)
from model_components.view_fusion.projection import (  # noqa: E402
    PinholeProjection,
)
from navigation.artifacts import decode_sample_navigation  # noqa: E402
from reactive_training_contracts import (  # noqa: E402
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
)
from training.dataset_policy import (  # noqa: E402
    KITSCENES_TRAINING_POLICY,
    adapt_egomotion_history,
)
from training.losses.control_rollout import integrate_controls_torch  # noqa: E402
from training.reactive_multitask import (  # noqa: E402
    ReactiveMultitaskObjective,
    ReactiveTrainingScope,
    ReactiveTrainingStage,
    configure_model_for_stage,
)


IMAGE_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(1, 3, 1, 1)
OUTPUT_HORIZON_SECONDS = (1, 2, 3, 5, 6.4)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Apply the normalization used by the packed production loader."""
    if images.ndim < 3 or images.shape[-3] != 3:
        raise ValueError("camera tensors must have a three-channel RGB axis")
    values = images.to(dtype=torch.float32) / 255.0
    parameter_shape = [1] * (values.ndim - 3) + [3, 1, 1]
    mean = IMAGE_MEAN.reshape(parameter_shape).to(
        device=values.device,
        dtype=values.dtype,
    )
    std = IMAGE_STD.reshape(parameter_shape).to(
        device=values.device,
        dtype=values.dtype,
    )
    return (values - mean) / std


def _matrix(value: Mapping[str, Any], name: str) -> torch.Tensor:
    payload = value.get(name)
    if not isinstance(payload, Mapping):
        raise ValueError(f"calibration field {name!r} is missing")
    matrix = np.asarray(payload.get("matrix"), dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"calibration field {name!r} is non-finite")
    return torch.from_numpy(matrix.copy())


class LocalKITScenesDataset(Dataset):
    """Adapt one raw KITScenes scene subset to the Reactive model contract."""

    def __init__(
        self,
        *,
        data_root: Path,
        scene_ids: Sequence[str],
        camera_names: Sequence[str],
    ) -> None:
        self.base = KitScenesDataset(
            data_root=data_root,
            split="train",
            camera_names=list(camera_names),
            scene_ids=list(scene_ids),
            image_size=REACTIVE_CAMERA_IMAGE_SIZE,
            include_navigation=True,
            benchmark_protocol=False,
        )
        self.camera_index = list(camera_names).index("camera_base_front_center")

    def __len__(self) -> int:
        return len(self.base)

    def _history_images(self, idx: int) -> torch.Tensor:
        rows = self.base.bevformer_history_rows(idx)
        return torch.stack(
            [
                self.base._load_multiview_frame(scene_id, frame_idx)
                for scene_id, frame_idx in rows
            ]
        )

    def behavior_target(self, idx: int) -> tuple[torch.Tensor, float]:
        """Return raw future controls and initial speed without loading images."""
        history, controls = self.base.egomotion_for(idx)
        initial_speed = float(history.reshape(-1, 4)[-1, 0].item())
        return (
            controls.to(dtype=torch.float32).reshape(64, 2),
            initial_speed,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base[idx]
        scene_id = str(sample["scene_id"])
        frame_idx = int(sample["frame_idx"])

        current = _normalize_images(sample["visual_tiles"])
        front_native = _normalize_images(
            self.base.front_camera_for_row(
                scene_id,
                frame_idx,
                image_size=REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
            )
        )
        current[self.camera_index] = F.interpolate(
            front_native.unsqueeze(0),
            size=(REACTIVE_CAMERA_IMAGE_SIZE, REACTIVE_CAMERA_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        history = torch.stack(
            [_normalize_images(frame) for frame in self._history_images(idx)]
        )

        calibration = self.base.reactive_calibration_for(
            idx,
            image_size=REACTIVE_CAMERA_IMAGE_SIZE,
        )
        map_context, route_mask, navigation_metadata = decode_sample_navigation(
            sample["navigation_members"]
        )
        map_valid = bool(navigation_metadata["map_valid"])
        route_valid = bool(navigation_metadata["route_valid"])
        raw_channel_valid = navigation_metadata.get("route_channel_valid")
        if isinstance(raw_channel_valid, (list, tuple)):
            route_channel_valid = torch.tensor(
                raw_channel_valid,
                dtype=torch.bool,
            )
        else:
            route_channel_valid = torch.tensor(
                [
                    route_valid,
                    route_valid
                    and bool(
                        navigation_metadata.get(
                            "destination_visible",
                            False,
                        )
                    ),
                ],
                dtype=torch.bool,
            )

        ego_history = sample["egomotion_history"].to(dtype=torch.float32)
        initial_speed = float(ego_history.reshape(-1, 4)[-1, 0].item())
        return {
            "visual_tiles": current,
            "camera_history_tiles": history,
            "front_camera_tile": front_native,
            "camera_projection_matrix": _matrix(
                calibration,
                "projection",
            ),
            "camera_history_projection_matrix": _matrix(
                calibration,
                "history_projection",
            ),
            "front_camera_projection_matrix": _matrix(
                calibration,
                "front_projection",
            ),
            "camera_geometry_type": "pinhole",
            "map_context": torch.from_numpy(
                np.ascontiguousarray(map_context, dtype=np.float32)
            ),
            "route_mask": torch.from_numpy(
                np.ascontiguousarray(route_mask, dtype=np.float32)
            ),
            "map_valid": torch.tensor(map_valid, dtype=torch.bool),
            "route_valid": torch.tensor(route_valid, dtype=torch.bool),
            "route_channel_valid": route_channel_valid,
            "egomotion_history": ego_history,
            "visual_history": torch.zeros(896, dtype=torch.float32),
            "trajectory_target": sample["trajectory_target"].to(dtype=torch.float32),
            "trajectory_xy_m": torch.from_numpy(
                np.ascontiguousarray(
                    sample["trajectory_xy_m"],
                    dtype=np.float32,
                )
            ),
            "trajectory_valid": torch.from_numpy(
                np.ascontiguousarray(
                    sample["trajectory_valid"],
                    dtype=np.bool_,
                )
            ),
            "initial_speed_mps": torch.tensor(
                initial_speed,
                dtype=torch.float32,
            ),
            "source_frame_index": torch.tensor(frame_idx, dtype=torch.int64),
            "source_frame_interval_us": torch.tensor(
                REACTIVE_BEVFORMER_FRAME_INTERVAL_US // 5,
                dtype=torch.int64,
            ),
            "sample_uid": f"kitscenes-{scene_id}-f{frame_idx:06d}",
            "split_group_uid": f"kitscenes-{scene_id}",
        }


def _scene_split(
    data_root: Path,
    manifest_path: Path,
) -> tuple[list[str], list[str]]:
    """Use the production validation groups when they occur in the local data."""
    local_scenes = {
        path.name for path in data_root.joinpath("train").iterdir() if path.is_dir()
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_groups = {
        str(value).removeprefix("kitscenes-")
        for value in manifest["validation_group_uids"]
    }
    validation = sorted(local_scenes & validation_groups)
    training = sorted(local_scenes - validation_groups)
    if not training or not validation:
        raise ValueError(
            "the local subset must contain both training and validation scenes"
        )
    return training, validation


def _load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[AutoE2E, dict[str, Any]]:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = payload.get("config")
    state = payload.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain config/model_state_dict")
    accepted = set(inspect.signature(AutoE2E.__init__).parameters) - {"self"}
    kwargs = {key: value for key, value in config.items() if key in accepted}
    kwargs["is_pretrained"] = False
    model = AutoE2E(**kwargs)
    model.load_state_dict(state, strict=True)
    return model.to(device), dict(config)


def _replace_planner(
    model: AutoE2E,
    planner_name: str,
    *,
    num_proposals: int,
    pc_range: Sequence[float],
) -> None:
    if planner_name == "gru":
        planner: nn.Module = GRUPlanner(
            embed_dim=256,
            num_timesteps=64,
            num_signals=2,
            num_points=16,
            egomotion_dim=256,
            visual_history_dim=896,
        )
    elif planner_name == "query":
        planner = PadQueryPlanner(
            embed_dim=256,
            num_timesteps=64,
            num_signals=2,
            num_proposals=num_proposals,
            num_refinement_steps=4,
            num_bev_points=8,
            num_heads=8,
            egomotion_dim=256,
            visual_history_dim=896,
            pc_range=pc_range,
        )
    else:
        raise ValueError(f"unknown planner {planner_name!r}")
    model.Reactive_E2E.TrajectoryPlanner = planner
    model.Reactive_E2E.planner_mode = planner_name
    model.Reactive_E2E.FusedFeaturePooling = None


def _configure_trainable_parameters(
    model: AutoE2E,
    *,
    trainable: str,
) -> None:
    configure_model_for_stage(
        model,
        ReactiveTrainingStage.KITSCENES_FINETUNE,
        freeze_bevformer=True,
        train_bev_head=False,
        training_scope=ReactiveTrainingScope.MULTITASK,
    )
    if trainable == "planner-only":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.Reactive_E2E.TrajectoryPlanner.parameters():
            parameter.requires_grad_(True)
    elif trainable != "reactive":
        raise ValueError(f"unknown trainable mode {trainable!r}")


def _move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    applied = adapt_egomotion_history(
        batch["egomotion_history"],
        KITSCENES_TRAINING_POLICY,
    )
    moved = dict(batch)
    moved["egomotion_history"] = applied
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in moved.items()
    }


def _projection_operators(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[PinholeProjection, PinholeProjection, tuple[PinholeProjection, ...]]:
    current = PinholeProjection(
        batch["camera_projection_matrix"].to(device),
        geometry_type="pinhole",
    )
    front = PinholeProjection(
        batch["front_camera_projection_matrix"].to(device),
        geometry_type="pinhole",
    )
    history_matrix = batch["camera_history_projection_matrix"].to(device)
    history = tuple(
        PinholeProjection(
            history_matrix[:, index],
            geometry_type="pinhole",
        )
        for index in range(history_matrix.shape[1])
    )
    return current, front, history


def _forward_model(
    model: AutoE2E,
    batch: Mapping[str, Any],
    *,
    training: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    projection, front_projection, history_projection = _projection_operators(
        batch, batch["visual_tiles"].device
    )
    output = model(
        batch["visual_tiles"],
        batch["map_context"],
        batch["visual_history"],
        batch["egomotion_history"],
        route_mask=batch["route_mask"],
        map_valid=batch["map_valid"],
        route_valid=batch["route_valid"],
        projection=projection,
        geometry_type="pinhole",
        camera_history_tiles=batch["camera_history_tiles"],
        history_projections=history_projection,
        front_camera_tile=batch["front_camera_tile"],
        front_projection=front_projection,
        mode="train" if training else "infer",
        trajectory_target=batch["trajectory_target"] if training else None,
        trajectory_xy_m=batch["trajectory_xy_m"] if training else None,
        trajectory_valid=batch["trajectory_valid"] if training else None,
        initial_speed_mps=batch["initial_speed_mps"],
        return_auxiliary=not training,
        compute_bev_segmentation=False,
        compute_route_reconstruction=True,
    )
    if not isinstance(output, tuple):
        raise RuntimeError("Reactive model omitted training auxiliary outputs")
    return output


def _make_objective() -> ReactiveMultitaskObjective:
    return ReactiveMultitaskObjective(
        ReactiveTrainingStage.KITSCENES_FINETUNE,
        bev_pos_weight=(1.0,) * 8,
        trajectory_weight=1.0,
        bev_weight=0.0,
        route_weight=1.0,
        corridor_pos_weight=1.0,
        training_scope=ReactiveTrainingScope.MULTITASK,
    )


def _train_one_epoch(
    model: AutoE2E,
    loader: DataLoader,
    objective: ReactiveMultitaskObjective,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    gradient_accumulation: int,
    grad_clip: float,
    precision: str,
    query_loss_weight: float,
    log_every: int,
    steps_per_epoch: int,
    max_steps: int,
    completed_steps: int,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = defaultdict(float)
    micro_steps = 0
    samples_seen = 0
    optimizer_steps = 0
    started = time.perf_counter()

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16" and device.type == "cuda",
        ):
            controls, auxiliary = _forward_model(
                model,
                batch,
                training=True,
            )
            terms = objective(controls, auxiliary, batch)
            loss = terms["total"]
            query_loss = controls.new_zeros(())
            planner = model.Reactive_E2E.TrajectoryPlanner
            if isinstance(planner, PadQueryPlanner):
                if not torch.is_tensor(planner.last_aux_loss):
                    raise RuntimeError(
                        "PAD query planner did not produce auxiliary loss"
                    )
                query_loss = planner.last_aux_loss
                loss = loss + query_loss_weight * query_loss
        (loss / gradient_accumulation).backward()
        totals["loss"] += float(loss.detach())
        totals["trajectory"] += float(terms["trajectory"].detach())
        totals["route"] += float(terms["route_reconstruction"].detach())
        totals["query"] += float(query_loss.detach())
        micro_steps += 1
        samples_seen += int(batch["visual_tiles"].shape[0])

        should_step = micro_steps % gradient_accumulation == 0
        if not should_step:
            continue
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.grad is not None
            ],
            grad_clip,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        completed_steps += 1
        if optimizer_steps % log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"step={completed_steps} "
                f"loss={totals['loss'] / micro_steps:.5f} "
                f"route={totals['route'] / micro_steps:.5f} "
                f"query={totals['query'] / micro_steps:.5f} "
                f"samples/s={samples_seen / max(elapsed, 1e-6):.3f}",
                flush=True,
            )
        if steps_per_epoch and optimizer_steps >= steps_per_epoch:
            break
        if max_steps and completed_steps >= max_steps:
            break

    if micro_steps % gradient_accumulation:
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.grad is not None
            ],
            grad_clip,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        completed_steps += 1

    denominator = max(micro_steps, 1)
    return {key: value / denominator for key, value in totals.items()} | {
        "optimizer_steps": float(optimizer_steps),
        "micro_steps": float(micro_steps),
        "seconds": time.perf_counter() - started,
    }, completed_steps


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


@torch.no_grad()
def _validate(
    model: AutoE2E,
    loader: DataLoader,
    objective: ReactiveMultitaskObjective,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    records: list[dict[str, Any]] = []
    proposal_histogram: Counter[int] = Counter()
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16" and device.type == "cuda",
        ):
            controls, auxiliary = _forward_model(
                model,
                batch,
                training=False,
            )
            terms = objective(controls, auxiliary, batch)
        controls_3d = controls.reshape(controls.shape[0], -1, 2).float()
        predicted_xy, _, _ = integrate_controls_torch(
            controls_3d,
            batch["initial_speed_mps"].float(),
        )
        target_xy = batch["trajectory_xy_m"].float()
        valid = batch["trajectory_valid"].to(dtype=torch.bool)
        error = torch.linalg.vector_norm(
            predicted_xy - target_xy,
            dim=-1,
        )
        planner = model.Reactive_E2E.TrajectoryPlanner
        if isinstance(planner, PadQueryPlanner):
            if planner.last_selected_proposal is not None:
                proposal_histogram.update(
                    int(value)
                    for value in planner.last_selected_proposal.cpu().tolist()
                )
            candidate_xy = planner.last_proposal_xy
        else:
            candidate_xy = None

        for local_index in range(error.shape[0]):
            sample_valid = valid[local_index]
            record: dict[str, Any] = {
                "sample_uid": batch["sample_uid"][local_index],
                "split_group_uid": batch["split_group_uid"][local_index],
                "trajectory": float(terms["trajectory"].detach().cpu()),
                "route": float(terms["route_reconstruction"].detach().cpu()),
            }
            for seconds in OUTPUT_HORIZON_SECONDS:
                step = min(
                    error.shape[1] - 1,
                    max(0, round(seconds * 10) - 1),
                )
                prefix = sample_valid[: step + 1]
                horizon_error = error[local_index, : step + 1][prefix]
                record[f"ade_{seconds}s_m"] = (
                    float(horizon_error.mean().cpu()) if horizon_error.numel() else None
                )
                record[f"fde_{seconds}s_m"] = (
                    float(error[local_index, step].cpu())
                    if bool(sample_valid[step])
                    else None
                )
            if candidate_xy is not None:
                candidate_error = torch.linalg.vector_norm(
                    candidate_xy[local_index] - target_xy[local_index],
                    dim=-1,
                )
                candidate_mask = sample_valid[None, :]
                candidate_ade = (
                    candidate_error * candidate_mask.to(dtype=candidate_error.dtype)
                ).sum(dim=-1) / candidate_mask.sum(dim=-1).clamp_min(1)
                record["min_ade_6p4s_m"] = float(candidate_ade.min().cpu())
            records.append(record)

    metrics: dict[str, Any] = {
        "sample_count": len(records),
    }
    for seconds in OUTPUT_HORIZON_SECONDS:
        ade_values = [
            value
            for record in records
            if (value := record[f"ade_{seconds}s_m"]) is not None
        ]
        fde_values = [
            value
            for record in records
            if (value := record[f"fde_{seconds}s_m"]) is not None
        ]
        metrics[f"ade_{seconds}s_m"] = _mean(ade_values)
        metrics[f"fde_{seconds}s_m"] = _mean(fde_values)
    metrics["trajectory_loss"] = _mean([record["trajectory"] for record in records])
    metrics["route_loss"] = _mean([record["route"] for record in records])
    min_ade = [
        record["min_ade_6p4s_m"] for record in records if "min_ade_6p4s_m" in record
    ]
    if min_ade:
        metrics["min_ade_6p4s_m"] = _mean(min_ade)
        metrics["proposal_selection_histogram"] = {
            str(key): value for key, value in sorted(proposal_histogram.items())
        }

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_scene[record["split_group_uid"]].append(record)
    scene_metrics: dict[str, dict[str, float | None]] = {}
    for scene_id, scene_records in sorted(by_scene.items()):
        scene_metrics[scene_id] = {
            f"ade_{seconds}s_m": _mean(
                [
                    value
                    for record in scene_records
                    if (value := record[f"ade_{seconds}s_m"]) is not None
                ]
            )
            for seconds in OUTPUT_HORIZON_SECONDS
        }
    metrics["scenes"] = scene_metrics
    metrics["scene_balanced_ade_3s_m"] = _mean(
        [
            value
            for scene in scene_metrics.values()
            if (value := scene["ade_3s_m"]) is not None
        ]
    )
    metrics["selection_score"] = -float(metrics["scene_balanced_ade_3s_m"])
    return metrics


def _save_checkpoint(
    path: Path,
    model: AutoE2E,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    *,
    epoch: int,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "config": dict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": dict(metrics),
        },
        path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a locally initialized GRU or query planner.",
    )
    parser.add_argument(
        "--planner",
        choices=("gru", "query"),
        required=True,
    )
    parser.add_argument(
        "--trainable",
        choices=("planner-only", "reactive"),
        default="planner-only",
        help=(
            "planner-only isolates architecture; reactive also trains "
            "navigation fusion and the route head as in KITScenes Stage B."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "ckpts" / "checkpoint.pt",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=(REPO_ROOT / "Model" / "data_parsing" / "kit_scenes" / "datasets"),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=(MODEL_ROOT / "training" / "splits" / "kitscenes_train_dev_v4.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "local_training" / "runs",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print running loss after this many optimizer steps.",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=149)
    parser.add_argument("--num-proposals", type=int, default=16)
    parser.add_argument("--query-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="Optional local debug limit; zero uses the full local train subset.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=0,
        help="Optional local debug limit; zero uses the full local val subset.",
    )
    parser.add_argument("--train-scene-limit", type=int, default=0)
    parser.add_argument("--val-scene-limit", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-planners",
        choices=("fresh", "checkpoint"),
        default="fresh",
        help=(
            "fresh gives the fair architecture comparison. checkpoint keeps "
            "the existing GRU head and should only be used for ablations."
        ),
    )
    return parser.parse_args()


def _subset(dataset: Dataset, limit: int) -> Dataset:
    if limit <= 0:
        return dataset
    return torch.utils.data.Subset(
        dataset,
        range(min(limit, len(dataset))),
    )


def main() -> None:
    args = _parse_args()
    if args.gradient_accumulation <= 0:
        raise ValueError("--gradient-accumulation must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        args.precision = "fp32"
    _seed_everything(args.seed)

    model, checkpoint_config = _load_checkpoint_model(
        args.checkpoint,
        device,
    )
    camera_names = checkpoint_config["physical_camera_order"]
    pc_range = checkpoint_config["view_fusion_kwargs"]["pc_range"]
    train_scenes, validation_scenes = _scene_split(
        args.data_root,
        args.validation_manifest,
    )
    if args.train_scene_limit > 0:
        train_scenes = train_scenes[: args.train_scene_limit]
    if args.val_scene_limit > 0:
        validation_scenes = validation_scenes[: args.val_scene_limit]
    print(
        f"Local split: train={len(train_scenes)} scenes, "
        f"validation={len(validation_scenes)} scenes"
    )
    train_dataset = LocalKITScenesDataset(
        data_root=args.data_root,
        scene_ids=train_scenes,
        camera_names=camera_names,
    )
    validation_dataset = LocalKITScenesDataset(
        data_root=args.data_root,
        scene_ids=validation_scenes,
        camera_names=camera_names,
    )
    print(
        f"Local samples: train={len(train_dataset)}, "
        f"validation={len(validation_dataset)}"
    )

    # Data/model setup above does not define the architecture comparison.
    # Reseed before creating the fresh planner so both planner runs begin from
    # the same reproducible initialization stream.
    _seed_everything(args.seed)
    if args.init_planners == "fresh":
        _replace_planner(
            model,
            args.planner,
            num_proposals=args.num_proposals,
            pc_range=pc_range,
        )
        model.Reactive_E2E.TrajectoryPlanner.to(device)
    elif args.planner != "gru":
        raise ValueError(
            "--init-planners checkpoint currently supports only the GRU planner"
        )
    _configure_trainable_parameters(
        model,
        trainable=args.trainable,
    )
    objective = _make_objective()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("no trainable parameters remain")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-3,
        threshold_mode="rel",
        cooldown=1,
        min_lr=1e-5,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        _subset(train_dataset, args.train_limit),
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=default_collate,
    )
    validation_loader = DataLoader(
        _subset(validation_dataset, args.val_limit),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=default_collate,
    )

    run_name = args.run_name or (
        f"{args.planner}-{args.trainable}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_directory = args.output_dir / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    config = {
        "planner": args.planner,
        "trainable": args.trainable,
        "init_planners": args.init_planners,
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "train_scenes": train_scenes,
        "validation_scenes": validation_scenes,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": (args.batch_size * args.gradient_accumulation),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "precision": args.precision,
        "seed": args.seed,
        "num_proposals": args.num_proposals,
        "query_loss_weight": args.query_loss_weight,
        "checkpoint_architecture": checkpoint_config["model_architecture_version"],
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "total_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    (run_directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(config, indent=2, sort_keys=True))
    print(
        f"Logging every {args.log_every} optimizer step(s); "
        f"one optimizer step consumes "
        f"{args.gradient_accumulation} microbatch(es).",
        flush=True,
    )

    start_epoch = 1
    best_score = -math.inf
    completed_steps = 0
    if args.resume is not None:
        payload = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload.get("metrics", {}).get("selection_score", -math.inf))

    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics, completed_steps = _train_one_epoch(
            model,
            train_loader,
            objective,
            optimizer,
            device=device,
            gradient_accumulation=args.gradient_accumulation,
            grad_clip=args.grad_clip,
            precision=args.precision,
            query_loss_weight=args.query_loss_weight,
            log_every=args.log_every,
            steps_per_epoch=args.steps_per_epoch,
            max_steps=args.max_steps,
            completed_steps=completed_steps,
        )
        validation_metrics = _validate(
            model,
            validation_loader,
            objective,
            device=device,
            precision=args.precision,
        )
        scheduler.step(validation_metrics["selection_score"])
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        (run_directory / "metrics.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _save_checkpoint(
            run_directory / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            config=config,
            metrics=validation_metrics,
        )
        if validation_metrics["selection_score"] > best_score:
            best_score = validation_metrics["selection_score"]
            _save_checkpoint(
                run_directory / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                config=config,
                metrics=validation_metrics,
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "validation": validation_metrics,
                },
                sort_keys=True,
            )
        )
        if args.max_steps and completed_steps >= args.max_steps:
            break


if __name__ == "__main__":
    main()
