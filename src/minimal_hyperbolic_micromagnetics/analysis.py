"""Reusable extraction of observables from field-following hysteresis data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hysteresis import HysteresisResult


@dataclass(frozen=True)
class HysteresisBranch:
    """One maximal monotonic leg of a hysteresis result."""

    name: str
    occurrence: int
    indices: np.ndarray
    B_T: np.ndarray
    mz_avg: np.ndarray
    nu_min: np.ndarray
    tau_rad: np.ndarray
    uniform_vortex_curvature: np.ndarray | None = None


@dataclass(frozen=True)
class BranchObservables:
    """Signed observables extracted from one field branch."""

    name: str
    occurrence: int
    remanence: float
    coercive_field_T: float
    switching_field_T: float
    vortex_nucleation_field_T: float


@dataclass(frozen=True)
class HysteresisAnalysis:
    """Observable summaries for every monotonic field branch."""

    branches: tuple[BranchObservables, ...]

    @property
    def descending(self) -> BranchObservables | None:
        return next(
            (branch for branch in self.branches if branch.name == "descending"),
            None,
        )

    @property
    def ascending(self) -> BranchObservables | None:
        return next(
            (branch for branch in self.branches if branch.name == "ascending"),
            None,
        )


def _validate_result_arrays(result: HysteresisResult) -> None:
    fields = np.asarray(result.B_T, dtype=float)
    if fields.ndim != 1 or len(fields) < 2:
        raise ValueError("hysteresis result must contain at least two field points")
    if not np.all(np.isfinite(fields)):
        raise ValueError("hysteresis fields must be finite")
    for name in ("mz_avg", "nu_min", "tau_rad"):
        values = np.asarray(getattr(result, name), dtype=float)
        if values.shape != fields.shape:
            raise ValueError(f"result.{name} must have the same shape as result.B_T")


def _optional_slice(values, indices):
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    return array[indices]


def split_field_branches(result: HysteresisResult) -> tuple[HysteresisBranch, ...]:
    """Split a field path into maximal strictly descending/ascending legs.

    Repeated turning-point fields are assigned to their respective neighboring
    legs. A turning point that is not duplicated is shared by both legs.
    """
    _validate_result_arrays(result)
    fields = np.asarray(result.B_T, dtype=float)
    nonzero_edges = np.flatnonzero(np.diff(fields) != 0.0)
    if len(nonzero_edges) == 0:
        raise ValueError("field path contains no monotonic leg")

    segments = []
    start = 0
    direction = int(np.sign(fields[nonzero_edges[0] + 1] - fields[nonzero_edges[0]]))
    last_edge = int(nonzero_edges[0])
    occurrences = {"descending": 0, "ascending": 0}

    def append_segment(segment_start, segment_stop, segment_direction):
        name = "descending" if segment_direction < 0 else "ascending"
        occurrence = occurrences[name]
        occurrences[name] += 1
        indices = np.arange(segment_start, segment_stop, dtype=int)
        segments.append(
            HysteresisBranch(
                name=name,
                occurrence=occurrence,
                indices=indices,
                B_T=fields[indices],
                mz_avg=np.asarray(result.mz_avg, dtype=float)[indices],
                nu_min=np.asarray(result.nu_min, dtype=float)[indices],
                tau_rad=np.asarray(result.tau_rad, dtype=float)[indices],
                uniform_vortex_curvature=_optional_slice(
                    result.uniform_vortex_curvature,
                    indices,
                ),
            )
        )

    for edge in nonzero_edges[1:]:
        edge = int(edge)
        edge_direction = int(np.sign(fields[edge + 1] - fields[edge]))
        if edge_direction != direction:
            append_segment(start, last_edge + 2, direction)
            start = edge
            direction = edge_direction
        last_edge = edge
    append_segment(start, last_edge + 2, direction)
    return tuple(segments)


def _resolve_branch(result, branch, occurrence=0) -> HysteresisBranch:
    if isinstance(branch, HysteresisBranch):
        return branch
    if branch not in ("descending", "ascending"):
        raise ValueError("branch must be 'descending' or 'ascending'")
    matches = [
        item
        for item in split_field_branches(result)
        if item.name == branch and item.occurrence == occurrence
    ]
    if not matches:
        raise ValueError(f"hysteresis result contains no {branch} branch {occurrence}")
    return matches[0]


def _interpolate_at_crossing(independent, dependent, target=0.0) -> float:
    independent = np.asarray(independent, dtype=float)
    dependent = np.asarray(dependent, dtype=float)
    exact = np.flatnonzero(np.isclose(independent, target, rtol=0.0, atol=1e-14))
    if len(exact):
        return float(dependent[int(exact[0])])

    shifted = independent - target
    crossings = np.flatnonzero(shifted[:-1] * shifted[1:] < 0.0)
    if len(crossings) == 0:
        return float("nan")
    index = int(crossings[0])
    x0, x1 = independent[index], independent[index + 1]
    y0, y1 = dependent[index], dependent[index + 1]
    return float(y0 + (target - x0) * (y1 - y0) / (x1 - x0))


def remanent_magnetization(
    result: HysteresisResult,
    *,
    branch="descending",
    occurrence=0,
) -> float:
    """Return signed projected remanence by interpolation to ``B=0``."""
    selected = _resolve_branch(result, branch, occurrence)
    return _interpolate_at_crossing(selected.B_T, selected.mz_avg)


def coercive_field_from_hysteresis(
    result: HysteresisResult,
    *,
    branch="descending",
    occurrence=0,
) -> float:
    """Return the signed field where projected magnetization crosses zero."""
    selected = _resolve_branch(result, branch, occurrence)
    return _interpolate_at_crossing(selected.mz_avg, selected.B_T)


def switching_field_from_hysteresis(
    result: HysteresisResult,
    *,
    branch="descending",
    occurrence=0,
) -> float:
    """Return the midpoint field of the largest wrapped angular jump."""
    selected = _resolve_branch(result, branch, occurrence)
    if len(selected.B_T) < 2:
        return float("nan")
    angle_jump = np.abs(
        np.angle(np.exp(1j * np.diff(np.asarray(selected.tau_rad, dtype=float))))
    )
    index = int(np.argmax(angle_jump))
    return float(0.5 * (selected.B_T[index] + selected.B_T[index + 1]))


def _stability_nucleation_field(branch: HysteresisBranch) -> float:
    curvature = branch.uniform_vortex_curvature
    if curvature is None:
        return float("nan")
    curvature = np.asarray(curvature, dtype=float)
    finite = np.isfinite(curvature)
    crossings = np.flatnonzero(
        finite[:-1]
        & finite[1:]
        & (curvature[:-1] > 0.0)
        & (curvature[1:] <= 0.0)
    )
    if len(crossings) == 0:
        return float("nan")
    index = int(crossings[0])
    return _interpolate_at_crossing(
        curvature[index : index + 2],
        branch.B_T[index : index + 2],
    )


def _threshold_nucleation_field(
    branch: HysteresisBranch,
    nu_threshold,
) -> float:
    nu = np.asarray(branch.nu_min, dtype=float)
    if nu_threshold is None:
        positive = nu[nu > 1.0e-14]
        if len(positive) == 0:
            return float("nan")
        nu_threshold = 0.5 * float(np.min(positive))
    if not np.isfinite(nu_threshold) or nu_threshold < 0.0:
        raise ValueError("nu_threshold must be finite and non-negative")

    nonuniform = np.flatnonzero(nu > nu_threshold)
    if len(nonuniform) == 0:
        return float("nan")
    index = int(nonuniform[0])
    if index == 0:
        return float(branch.B_T[0])
    return _interpolate_at_crossing(
        nu[index - 1 : index + 1],
        branch.B_T[index - 1 : index + 1],
        target=nu_threshold,
    )


def vortex_nucleation_field_from_hysteresis(
    result: HysteresisResult,
    *,
    branch="descending",
    occurrence=0,
    method="stability",
    nu_threshold=None,
) -> float:
    """Return the signed first vortex-nucleation field on one branch.

    ``method='stability'`` uses the zero of the analytically tracked uniform
    vortex curvature. ``method='threshold'`` detects departure of ``nu_min``
    from zero. ``method='auto'`` prefers stability and falls back to the
    explicitly resolved vortex coordinate.
    """
    selected = _resolve_branch(result, branch, occurrence)
    if method not in ("stability", "threshold", "auto"):
        raise ValueError("method must be 'stability', 'threshold', or 'auto'")
    if method in ("stability", "auto"):
        field = _stability_nucleation_field(selected)
        if np.isfinite(field) or method == "stability":
            return field
    return _threshold_nucleation_field(selected, nu_threshold)


def analyze_hysteresis(
    result: HysteresisResult,
    *,
    nucleation_method="auto",
    nu_threshold=None,
) -> HysteresisAnalysis:
    """Extract standard signed observables from every monotonic field leg."""
    summaries = []
    for branch in split_field_branches(result):
        summaries.append(
            BranchObservables(
                name=branch.name,
                occurrence=branch.occurrence,
                remanence=remanent_magnetization(result, branch=branch),
                coercive_field_T=coercive_field_from_hysteresis(
                    result,
                    branch=branch,
                ),
                switching_field_T=switching_field_from_hysteresis(
                    result,
                    branch=branch,
                ),
                vortex_nucleation_field_T=(
                    vortex_nucleation_field_from_hysteresis(
                        result,
                        branch=branch,
                        method=nucleation_method,
                        nu_threshold=nu_threshold,
                    )
                ),
            )
        )
    return HysteresisAnalysis(tuple(summaries))
