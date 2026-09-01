"""Physically explicit occupancy-suppression policies.

The functions in this module are deliberately stateless.  They consume only
the current point cloud, frozen static evidence and (for the explicitly named
PERSON policy) current exclusive RGB ownership.  No policy can read future
frames, calibration candidates or ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEGACY = "LEGACY"
PHYSICAL_DIRECT = "PHYSICAL_DIRECT"
GROUND_LIMITED = "GROUND_LIMITED"
HEIGHT_BANDED = "HEIGHT_BANDED"
TEMPORAL_PRIMARY = "TEMPORAL_PRIMARY_OCCUPANCY_SECONDARY"
PERSON_CONDITIONAL = "PERSON_CONDITIONAL_PROTECTION"


@dataclass(frozen=True)
class Policy:
    name: str
    low_height_m: float = 0.28
    middle_height_m: float = 1.90


def ground_height(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Return signed metric distance to ``normal . p + d = 0``.

    The normal is normalized explicitly rather than assumed to be unit length.
    Positive values are on the side pointed to by the supplied normal.
    """
    values = np.asarray(points, np.float64)
    direction = np.asarray(normal, np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("ground-plane normal must be finite and non-zero")
    return (values @ direction + float(d)) / norm


def occupancy_keep(
    points: np.ndarray,
    height_m: np.ndarray,
    occupancy_near: np.ndarray,
    temporal_keep: np.ndarray,
    policy: Policy,
    owners: np.ndarray | None = None,
    owner_classes: tuple[str, ...] | list[str] = (),
) -> tuple[np.ndarray, dict[str, int]]:
    """Return the occupancy keep mask and auditable decision counts.

    ``temporal_keep`` is not folded into the returned mask.  The caller keeps
    the existing priority explicitly as ``temporal_keep & occupancy_keep``.
    """
    values = np.asarray(points, np.float64)
    height = np.asarray(height_m, np.float64)
    near = np.asarray(occupancy_near, bool)
    temporal = np.asarray(temporal_keep, bool)
    if values.shape != (len(height), 3) or near.shape != height.shape or temporal.shape != height.shape:
        raise ValueError("suppression inputs have inconsistent shapes")

    if policy.name == LEGACY:
        # Historical bug retained only as the frozen production/reference path.
        feature = values[:, 2] - height
        remove = near & ((feature <= 0.28) | (feature >= 1.90))
    elif policy.name == PHYSICAL_DIRECT:
        remove = near & ((height <= 0.28) | (height >= 1.90))
    elif policy.name in {GROUND_LIMITED, HEIGHT_BANDED}:
        # HEIGHT_BANDED makes the middle/high bands explicit: occupancy has no
        # independent hard-delete authority there; temporal evidence still does.
        remove = near & (height >= 0.0) & (height <= policy.low_height_m)
    elif policy.name == TEMPORAL_PRIMARY:
        # Candidate C: occupancy is only corroborating evidence for a point
        # already marked static by the temporal prior.  It therefore adds no
        # independent hard delete after the temporal mask is applied.
        remove = near & ~temporal & (height >= 0.0) & (height <= policy.low_height_m)
    elif policy.name == PERSON_CONDITIONAL:
        remove = near & ((height <= 0.28) | (height >= 1.90))
    else:
        raise ValueError(f"unknown occupancy policy: {policy.name}")

    restored = np.zeros(len(height), bool)
    if policy.name == PERSON_CONDITIONAL:
        if owners is None:
            raise ValueError("PERSON_CONDITIONAL requires exclusive point owners")
        owned = np.asarray(owners, np.int32)
        if owned.shape != height.shape:
            raise ValueError("owners have inconsistent shape")
        person_owner = np.zeros(len(height), bool)
        for index, entity_class in enumerate(owner_classes):
            if entity_class == "PERSON":
                person_owner |= owned == index
        # A point receives protection only when occupancy is the sole suppressor,
        # it is exclusively owned by a current PERSON box and has person-plausible
        # signed ground height.  Temporal-static points are never restored.
        restored = (remove & temporal & person_owner
                    & (height >= 0.08) & (height <= 2.15))
        remove &= ~restored

    occupancy = ~remove
    return occupancy, {
        "occupancy_removed": int(np.count_nonzero(temporal & remove)),
        "points_restored_by_new_logic": int(np.count_nonzero(restored)),
        "person_box_restored_points": int(np.count_nonzero(restored)),
    }


def combined_keep(temporal_keep: np.ndarray, occupancy_keep_mask: np.ndarray) -> np.ndarray:
    """Keep temporal evidence primary and occupancy evidence secondary."""
    return np.asarray(temporal_keep, bool) & np.asarray(occupancy_keep_mask, bool)
