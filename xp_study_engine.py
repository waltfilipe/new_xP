"""Match-level xP study: per-team destination rarity with hybrid league models."""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import pandas as pd

import passes_engine as pe

STUDY_MATCH_EVENT_ID = 15526003
XP_SMOOTHING = 1.0
XP_BLEND_ALPHA = 0.5
XP_PASS_MAX = 1.0

FIELD_X = pe.FIELD_X
FIELD_Y = pe.FIELD_Y
FIRST_THIRD_LINE_X = FIELD_X / 3.0
FIRST_THIRD_BLEND_END_X = 52.0
XP_FIRST_THIRD_MIN_MULT = 0.12

# Progress toward goal (StatsBomb x↑): penalize backward/lateral passes after rarity scoring.
# Calibrated on league completed passes — backward ratio bin mean xP ~0.28 vs forward ~0.38.
XP_PROGRESS_FLOOR_MULT = 0.30
XP_PROGRESS_LOGISTIC_K = 2.2

# Default grid (backward-compatible aliases).
XP_GRID_COLS = 12
XP_GRID_ROWS = 8
OD_GRID_COLS = 6
OD_GRID_ROWS = 4

XP_MODEL_HIER_DEST = "hierarchical_dest"
XP_MODEL_HIER_OD = "hierarchical_od"

XP_MODEL_LABELS: dict[str, str] = {
    XP_MODEL_HIER_DEST: "3 — Suavização hierárquica (destino 12×8)",
    XP_MODEL_HIER_OD: "4 — Suavização hierárquica (origem 8×6 → destino 12×8)",
}

XP_MODEL_COLUMNS: dict[str, str] = {
    XP_MODEL_HIER_DEST: "xp_hier_dest",
    XP_MODEL_HIER_OD: "xp_hier_od",
}

STUDY_MODELS: tuple[str, ...] = (XP_MODEL_HIER_DEST, XP_MODEL_HIER_OD)

THREAT_XP_THRESHOLDS: dict[str, float] = {
    XP_MODEL_HIER_DEST: 0.25,
    XP_MODEL_HIER_OD: 0.50,
}

DISTANCE_BAND_ORDER: tuple[str, ...] = ("short", "medium", "long")
DISTANCE_BAND_LABELS: dict[str, str] = {
    "short": "<12 m",
    "medium": "12–25 m",
    "long": ">25 m",
}


class GridConfig(NamedTuple):
    dest_cols: int
    dest_rows: int
    od_origin_cols: int
    od_origin_rows: int
    od_dest_cols: int
    od_dest_rows: int
    key: str
    label: str

    @property
    def od_cols(self) -> int:
        return self.od_origin_cols

    @property
    def od_rows(self) -> int:
        return self.od_origin_rows


STUDY_GRID = GridConfig(
    12, 8,
    8, 6,
    12, 8,
    "study_m3_m4",
    "Estudo — destino 12×8 · origem OD 8×6",
)

GRID_PRESETS: dict[str, GridConfig] = {
    "default": GridConfig(12, 8, 6, 4, 12, 8, "default", "Destino 12×8 · OD origem 6×4"),
    "dest_8x6": GridConfig(8, 6, 6, 4, 8, 6, "dest_8x6", "Destino 8×6 · OD origem 6×4"),
    "all_8x6": GridConfig(8, 6, 8, 6, 8, 6, "all_8x6", "Tudo 8×6"),
    "all_12x8": GridConfig(12, 8, 12, 8, 12, 8, "all_12x8", "Tudo 12×8"),
    STUDY_GRID.key: STUDY_GRID,
}
DEFAULT_GRID_PRESET = STUDY_GRID.key


def normalize_grid_preset(preset: str | None) -> str:
    key = str(preset or DEFAULT_GRID_PRESET).strip().lower()
    return key if key in GRID_PRESETS else DEFAULT_GRID_PRESET


def get_grid_config(preset: str | None = None) -> GridConfig:
    return GRID_PRESETS[normalize_grid_preset(preset)]


def list_grid_presets() -> tuple[GridConfig, ...]:
    """Ordered grid presets for UI selectors."""
    return tuple(GRID_PRESETS[k] for k in GRID_PRESETS)


def _parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "successful"})


def _cell_indices(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cols: int,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_bins = np.linspace(0.0, FIELD_X, cols + 1)
    y_bins = np.linspace(0.0, FIELD_Y, rows + 1)
    x_idx = np.clip(np.digitize(x, x_bins, right=True) - 1, 0, cols - 1)
    y_idx = np.clip(np.digitize(y, y_bins, right=True) - 1, 0, rows - 1)
    return x_idx, y_idx


def _progress_ratio_series(frame: pd.DataFrame) -> np.ndarray:
    """Signed progress toward opponent goal, normalized by pass length ∈ [-1, 1]."""
    dist = np.maximum(frame["pass_distance"].to_numpy(dtype=float), 0.5)
    dx = frame["progress_m"].to_numpy(dtype=float)
    return np.clip(dx / dist, -1.0, 1.0)


def progress_toward_goal_multiplier(progress_ratio: np.ndarray | pd.Series) -> np.ndarray:
    """Smooth monotone map: backward passes ↓ xP, forward passes ≈ 1.

    Logistic centered at lateral (ratio=0), floor at fully backward, cap at 1.0.
    Parameters calibrated against league marginal xP by progress-ratio bin.
    """
    ratio = np.asarray(progress_ratio, dtype=float)
    logistic = 1.0 / (1.0 + np.exp(-XP_PROGRESS_LOGISTIC_K * ratio))
    return XP_PROGRESS_FLOOR_MULT + (1.0 - XP_PROGRESS_FLOOR_MULT) * logistic


def _first_third_multiplier_vec(x_end: np.ndarray) -> np.ndarray:
    x = np.asarray(x_end, dtype=float)
    mult = np.ones(len(x), dtype=float)
    deep = x <= FIRST_THIRD_LINE_X
    mult[deep] = XP_FIRST_THIRD_MIN_MULT
    blend = (x > FIRST_THIRD_LINE_X) & (x < FIRST_THIRD_BLEND_END_X)
    if blend.any():
        t = (x[blend] - FIRST_THIRD_LINE_X) / (FIRST_THIRD_BLEND_END_X - FIRST_THIRD_LINE_X)
        mult[blend] = XP_FIRST_THIRD_MIN_MULT + (1.0 - XP_FIRST_THIRD_MIN_MULT) * t
    return mult


def _scale_grid_max_one(xp_grid: np.ndarray) -> np.ndarray:
    """Scale so the rarest cell equals XP_PASS_MAX (1.0); other cells are proportional."""
    mx = float(np.max(xp_grid))
    if mx <= 0:
        return xp_grid
    return xp_grid * (XP_PASS_MAX / mx)


def _enrich_match_passes(frame: pd.DataFrame) -> pd.DataFrame:
    sx, sy = pe._wyscout_to_sb(frame["start_x"], frame["start_y"])
    has_end = frame["end_x"].notna() & frame["end_y"].notna()
    ex = np.full(len(frame), np.nan)
    ey = np.full(len(frame), np.nan)
    if has_end.any():
        ex[has_end.to_numpy()], ey[has_end.to_numpy()] = pe._wyscout_to_sb(
            frame.loc[has_end, "end_x"], frame.loc[has_end, "end_y"]
        )

    out = pd.DataFrame({
        "player_id": frame["player_id"].astype(str),
        "player_name": frame["player_name"].astype(str),
        "position": frame["position"].astype(str).str.strip().str.upper() if "position" in frame.columns else "CM",
        "team": np.where(
            _parse_bool_series(frame["isHome"]),
            frame["home_team"].astype(str),
            frame["away_team"].astype(str),
        ),
        "is_success": _parse_bool_series(frame["outcome"]) if "outcome" in frame.columns else False,
        "action_type": frame["eventActionType"].astype(str).str.strip().str.lower(),
        "x_start": sx,
        "y_start": sy,
        "x_end": ex,
        "y_end": ey,
        "has_end": has_end.to_numpy(),
        "event_id": frame["event_id"].astype(int),
        "home_team": frame["home_team"].astype(str),
        "away_team": frame["away_team"].astype(str),
        "match_date": frame["match_date"].astype(str),
    })
    out["is_won"] = out["is_success"].astype(bool)
    out["pass_distance"] = np.where(
        out["has_end"],
        np.sqrt((out["x_end"] - out["x_start"]) ** 2 + (out["y_end"] - out["y_start"]) ** 2),
        0.0,
    )
    out["progress_m"] = np.where(
        out["has_end"],
        out["x_end"].to_numpy(dtype=float) - out["x_start"].to_numpy(dtype=float),
        0.0,
    )
    out["progress_ratio"] = _progress_ratio_series(out)
    out["is_restart"] = pe._restart_pass_mask(
        out["x_start"].to_numpy(dtype=float),
        out["y_start"].to_numpy(dtype=float),
        out["action_type"].astype(str).to_numpy(),
    )
    return out


def _count_destination_grid(
    passes: pd.DataFrame,
    grid: GridConfig,
) -> np.ndarray:
    count_grid = np.zeros((grid.dest_rows, grid.dest_cols), dtype=float)
    if passes is None or passes.empty:
        return count_grid

    completed = passes[passes["is_won"] & passes["has_end"]]
    if completed.empty:
        return count_grid

    x_idx, y_idx = _cell_indices(
        completed["x_end"].to_numpy(dtype=float),
        completed["y_end"].to_numpy(dtype=float),
        cols=grid.dest_cols,
        rows=grid.dest_rows,
    )
    for ix, iy in zip(x_idx, y_idx):
        count_grid[iy, ix] += 1.0
    return count_grid


def _count_od_tensor(passes: pd.DataFrame, grid: GridConfig) -> np.ndarray:
    o_rows, o_cols = grid.od_origin_rows, grid.od_origin_cols
    d_rows, d_cols = grid.od_dest_rows, grid.od_dest_cols
    tensor = np.zeros((o_rows, o_cols, d_rows, d_cols), dtype=float)
    if passes is None or passes.empty:
        return tensor

    completed = passes[passes["is_won"] & passes["has_end"]]
    if completed.empty:
        return tensor

    ox, oy = _cell_indices(
        completed["x_start"].to_numpy(dtype=float),
        completed["y_start"].to_numpy(dtype=float),
        cols=grid.od_origin_cols,
        rows=grid.od_origin_rows,
    )
    dx, dy = _cell_indices(
        completed["x_end"].to_numpy(dtype=float),
        completed["y_end"].to_numpy(dtype=float),
        cols=grid.od_dest_cols,
        rows=grid.od_dest_rows,
    )
    for oxi, oyi, dxi, dyi in zip(ox, oy, dx, dy):
        tensor[oyi, oxi, dyi, dxi] += 1.0
    return tensor


def _counts_to_xp_grid(
    count_grid: np.ndarray,
    *,
    smoothing: float = XP_SMOOTHING,
) -> np.ndarray:
    rows, cols = count_grid.shape
    xp_grid = np.ones((rows, cols), dtype=float)
    total = float(count_grid.sum())
    num_cells = rows * cols
    denom = total + smoothing * num_cells
    for iy in range(rows):
        for ix in range(cols):
            smoothed_count = count_grid[iy, ix] + smoothing
            freq = smoothed_count / denom
            xp_grid[iy, ix] = 1.0 / freq
    return _scale_grid_max_one(xp_grid)


def _blend_count_grid(
    match_count: np.ndarray,
    league_count_per_match: np.ndarray,
    *,
    alpha: float = XP_BLEND_ALPHA,
) -> np.ndarray:
    return alpha * match_count + (1.0 - alpha) * league_count_per_match


def _blend_od_tensor(
    match_tensor: np.ndarray,
    league_tensor_per_match: np.ndarray,
    *,
    alpha: float = XP_BLEND_ALPHA,
) -> np.ndarray:
    return alpha * match_tensor + (1.0 - alpha) * league_tensor_per_match


def _od_counts_to_lookup(tensor: np.ndarray) -> np.ndarray:
    o_rows, o_cols = tensor.shape[0], tensor.shape[1]
    d_rows, d_cols = tensor.shape[2], tensor.shape[3]
    total = float(tensor.sum())
    num_cells = tensor.size
    denom = total + XP_SMOOTHING * num_cells
    lookup = np.ones_like(tensor, dtype=float)
    for oyi in range(o_rows):
        for oxi in range(o_cols):
            for dyi in range(d_rows):
                for dxi in range(d_cols):
                    smoothed = tensor[oyi, oxi, dyi, dxi] + XP_SMOOTHING
                    lookup[oyi, oxi, dyi, dxi] = 1.0 / (smoothed / denom)
    return _scale_grid_max_one(lookup)


def build_destination_xp_grid(passes: pd.DataFrame, grid: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    count_grid = _count_destination_grid(passes, grid)
    return _counts_to_xp_grid(count_grid), count_grid


def build_team_xp_surfaces(
    passes: pd.DataFrame,
    grid: GridConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    xp_grids: dict[str, np.ndarray] = {}
    count_grids: dict[str, np.ndarray] = {}
    if passes is None or passes.empty:
        return xp_grids, count_grids

    for team, grp in passes.groupby("team", sort=False):
        team_name = str(team)
        xp_grids[team_name], count_grids[team_name] = build_destination_xp_grid(grp, grid)
    return xp_grids, count_grids


@functools.lru_cache(maxsize=1)
def _load_combined_league_pass_frame() -> pd.DataFrame:
    """Serie B + Serie A pass events for the global xP reference pool."""
    frames: list[pd.DataFrame] = []
    serie_b = pe._load_season_pass_frame()
    if not serie_b.empty:
        sb = serie_b.copy()
        sb["league_source"] = "serie_b"
        frames.append(sb)
    serie_a = pe._load_br_pass_frame()
    if not serie_a.empty:
        sa = serie_a.copy()
        sa["league_source"] = "serie_a"
        frames.append(sa)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@functools.lru_cache(maxsize=1)
def _league_completed_passes() -> pd.DataFrame:
    frame = _load_combined_league_pass_frame()
    if frame.empty:
        return pd.DataFrame()
    league_source = frame["league_source"].copy() if "league_source" in frame.columns else None
    passes = _enrich_match_passes(frame)
    if league_source is not None and len(league_source) == len(passes):
        passes = passes.copy()
        passes["league_source"] = league_source.to_numpy()
    passes = pe.filter_live_ball_passes(passes)
    if passes is None or passes.empty:
        return pd.DataFrame()
    completed = passes[passes["is_won"] & passes["has_end"]].copy()
    return completed


@functools.lru_cache(maxsize=16)
def _league_reference_surfaces(
    dest_cols: int,
    dest_rows: int,
    od_origin_cols: int,
    od_origin_rows: int,
    od_dest_cols: int,
    od_dest_rows: int,
) -> dict[str, np.ndarray | float | int]:
    grid = GridConfig(
        dest_cols, dest_rows,
        od_origin_cols, od_origin_rows,
        od_dest_cols, od_dest_rows,
        "", "",
    )
    completed = _league_completed_passes()
    if completed.empty:
        empty_dest = np.zeros((dest_rows, dest_cols), dtype=float)
        empty_od = np.zeros((od_origin_rows, od_origin_cols, od_dest_rows, od_dest_cols), dtype=float)
        return {
            "dest_count": empty_dest,
            "dest_count_per_match": empty_dest,
            "dest_xp": _counts_to_xp_grid(empty_dest),
            "od_count": empty_od,
            "od_count_per_match": empty_od,
            "num_matches": 1,
        }

    dest_count = _count_destination_grid(completed, grid)
    od_count = _count_od_tensor(completed, grid)
    if "league_source" in completed.columns:
        matches_by_league = completed.groupby("league_source")["event_id"].nunique()
        num_matches_serie_b = int(matches_by_league.get("serie_b", 0))
        num_matches_serie_a = int(matches_by_league.get("serie_a", 0))
        num_matches = max(num_matches_serie_b + num_matches_serie_a, 1)
    else:
        num_matches_serie_b = max(int(completed["event_id"].nunique()), 0)
        num_matches_serie_a = 0
        num_matches = max(num_matches_serie_b, 1)
    dest_per_match = dest_count / num_matches
    od_per_match = od_count / num_matches

    return {
        "dest_count": dest_count,
        "dest_count_per_match": dest_per_match,
        "dest_xp": _counts_to_xp_grid(dest_count),
        "od_count": od_count,
        "od_count_per_match": od_per_match,
        "num_matches": num_matches,
        "num_matches_serie_b": num_matches_serie_b,
        "num_matches_serie_a": num_matches_serie_a,
        "league_passes": int(len(completed)),
    }


def _cap_pass_xp(values: np.ndarray, zone_mult: np.ndarray) -> np.ndarray:
    return np.minimum(values * zone_mult, XP_PASS_MAX)


def _assign_study_xp_models(
    passes: pd.DataFrame,
    *,
    grid: GridConfig,
    count_grids_by_team: dict[str, np.ndarray],
    league: dict[str, np.ndarray | float | int],
    alpha: float = XP_BLEND_ALPHA,
) -> pd.DataFrame:
    out = passes.copy()
    for col in XP_MODEL_COLUMNS.values():
        out[col] = 0.0
    out["xp_zone_mult"] = 0.0
    out["dest_ix"] = -1
    out["dest_iy"] = -1

    mask = out["is_won"] & out["has_end"]
    if not mask.any():
        return out

    sub = out.loc[mask]
    x_idx, y_idx = _cell_indices(
        sub["x_end"].to_numpy(dtype=float),
        sub["y_end"].to_numpy(dtype=float),
        cols=grid.dest_cols,
        rows=grid.dest_rows,
    )
    ox, oy = _cell_indices(
        sub["x_start"].to_numpy(dtype=float),
        sub["y_start"].to_numpy(dtype=float),
        cols=grid.od_origin_cols,
        rows=grid.od_origin_rows,
    )
    odx, ody = _cell_indices(
        sub["x_end"].to_numpy(dtype=float),
        sub["y_end"].to_numpy(dtype=float),
        cols=grid.od_dest_cols,
        rows=grid.od_dest_rows,
    )
    zone_mult = _first_third_multiplier_vec(sub["x_end"].to_numpy(dtype=float))

    league_dest_per_match = league["dest_count_per_match"]  # type: ignore[index]
    league_od_per_match = league["od_count_per_match"]  # type: ignore[index]

    hier_dest_xp_by_team: dict[str, np.ndarray] = {}
    hier_od_lookup_by_team: dict[str, np.ndarray] = {}
    for team, count_grid in count_grids_by_team.items():
        blended_dest = _blend_count_grid(count_grid, league_dest_per_match, alpha=alpha)
        hier_dest_xp_by_team[team] = _counts_to_xp_grid(blended_dest)
        team_od = _count_od_tensor(passes[passes["team"].astype(str) == team], grid)
        blended_od = _blend_od_tensor(team_od, league_od_per_match, alpha=alpha)
        hier_od_lookup_by_team[team] = _od_counts_to_lookup(blended_od)

    hier_dest_vals = np.zeros(len(sub), dtype=float)
    hier_od_vals = np.zeros(len(sub), dtype=float)

    for i, (team, iy, ix, oyi, oxi, dyi, dxi) in enumerate(
        zip(sub["team"].astype(str), y_idx, x_idx, oy, ox, ody, odx)
    ):
        hier_grid = hier_dest_xp_by_team.get(team)
        hier_dest_vals[i] = float(hier_grid[iy, ix]) if hier_grid is not None else 0.0

        od_lookup = hier_od_lookup_by_team.get(team)
        hier_od_vals[i] = float(od_lookup[oyi, oxi, dyi, dxi]) if od_lookup is not None else 0.0

    out.loc[mask, "dest_ix"] = x_idx
    out.loc[mask, "dest_iy"] = y_idx
    out.loc[mask, "xp_zone_mult"] = zone_mult
    out.loc[mask, "xp_hier_dest"] = _cap_pass_xp(hier_dest_vals, zone_mult)
    out.loc[mask, "xp_hier_od"] = _cap_pass_xp(hier_od_vals, zone_mult)
    return out


def rank_players_by_xp(
    passes: pd.DataFrame,
    *,
    model: str = XP_MODEL_HIER_DEST,
) -> pd.DataFrame:
    col = XP_MODEL_COLUMNS.get(model, "xp_hier_dest")
    if passes is None or passes.empty or col not in passes.columns:
        return pd.DataFrame()

    scored = passes[passes["is_won"] & passes["has_end"]].copy()
    if scored.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for pid, grp in scored.groupby("player_id", sort=False):
        rows.append({
            "player_id": str(pid),
            "player_name": str(grp["player_name"].iloc[0]),
            "position": str(grp["position"].iloc[0]),
            "team": str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else grp["team"].iloc[0]),
            "passes_completed": int(len(grp)),
            "xp_total": float(grp[col].sum()),
            "xp_per_pass": float(grp[col].mean()),
            "xp_max_pass": float(grp[col].max()),
        })

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    ranking = ranking.sort_values(["xp_total", "xp_per_pass"], ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def build_model34_comparison_table(passes: pd.DataFrame) -> pd.DataFrame:
    """Compare xP totals and ranks for models 3 and 4 only."""
    if passes is None or passes.empty:
        return pd.DataFrame()

    scored = passes[passes["is_won"] & passes["has_end"]].copy()
    if scored.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for pid, grp in scored.groupby("player_id", sort=False):
        row = {
            "player_id": str(pid),
            "player_name": str(grp["player_name"].iloc[0]),
            "position": str(grp["position"].iloc[0]),
            "team": str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else grp["team"].iloc[0]),
            "passes_completed": int(len(grp)),
        }
        for model, col in XP_MODEL_COLUMNS.items():
            row[f"xp_{model}"] = float(grp[col].sum())
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    for model in XP_MODEL_COLUMNS:
        col = f"xp_{model}"
        table[f"rank_{model}"] = table[col].rank(ascending=False, method="min").astype(int)

    dest_col = f"xp_{XP_MODEL_HIER_DEST}"
    return table.sort_values(dest_col, ascending=False).reset_index(drop=True)


def top_xp_passes_for_player(
    passes: pd.DataFrame,
    player_id: str,
    *,
    n: int = 5,
    model: str = XP_MODEL_HIER_DEST,
) -> pd.DataFrame:
    col = XP_MODEL_COLUMNS.get(model, "xp_hier_dest")
    subset = passes[
        (passes["player_id"].astype(str) == str(player_id))
        & passes["is_won"]
        & passes["has_end"]
    ].copy()
    if subset.empty or col not in subset.columns:
        return subset
    subset = subset.assign(xp_value=subset[col])
    return subset.sort_values("xp_value", ascending=False).head(n).reset_index(drop=True)


def match_label(meta: dict) -> str:
    return f"{meta['home_team']} vs {meta['away_team']}"


def team_surface_for_player(
    xp_grids_by_team: dict[str, np.ndarray],
    count_grids_by_team: dict[str, np.ndarray],
    *,
    team: str,
    dest_rows: int,
    dest_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    empty_xp = np.ones((dest_rows, dest_cols), dtype=float) * 0.0
    empty_count = np.zeros((dest_rows, dest_cols), dtype=float)
    return (
        xp_grids_by_team.get(team, empty_xp),
        count_grids_by_team.get(team, empty_count),
    )


def grid_occupancy_stats(count_grids_by_team: dict[str, np.ndarray], grid: GridConfig) -> dict[str, float | int]:
    total_dest_cells = grid.dest_cols * grid.dest_rows
    total_od_cells = (
        grid.od_origin_cols * grid.od_origin_rows
        * grid.od_dest_cols * grid.od_dest_rows
    )
    dest_used = sum(int((cg > 0).sum()) for cg in count_grids_by_team.values())
    return {
        "dest_cells": total_dest_cells,
        "od_cells": total_od_cells,
        "dest_used": dest_used,
    }


def normalize_xp_model(model: str | None) -> str:
    key = str(model or XP_MODEL_HIER_DEST).strip().lower()
    return key if key in XP_MODEL_COLUMNS else XP_MODEL_HIER_DEST


def _distance_band_series(distances: pd.Series | np.ndarray) -> pd.Series:
    dist = np.asarray(distances, dtype=float)
    bands = np.full(len(dist), "medium", dtype=object)
    bands[dist < pe.DISTANCE_SHORT_MAX_M] = "short"
    bands[dist >= pe.DISTANCE_MEDIUM_MAX_M] = "long"
    return pd.Series(bands, index=getattr(distances, "index", None))


def build_distance_threat_study(passes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize mean xP and threat-pass counts by distance band (models 3 and 4)."""
    col_m3 = XP_MODEL_COLUMNS[XP_MODEL_HIER_DEST]
    col_m4 = XP_MODEL_COLUMNS[XP_MODEL_HIER_OD]
    thr_m3 = THREAT_XP_THRESHOLDS[XP_MODEL_HIER_DEST]
    thr_m4 = THREAT_XP_THRESHOLDS[XP_MODEL_HIER_OD]

    empty_band = pd.DataFrame(columns=[
        "distance_band", "band_label", "passes",
        "mean_xp_m3", "mean_xp_m4",
        "threat_m3", "threat_m4", "pct_threat_m3", "pct_threat_m4",
    ])
    empty_player = pd.DataFrame(columns=[
        "player_id", "player_name", "team", "distance_band", "band_label", "passes",
        "mean_xp_m3", "mean_xp_m4", "threat_m3", "threat_m4",
    ])

    if passes is None or passes.empty:
        return empty_band, empty_player

    scored = passes[passes["is_won"] & passes["has_end"]].copy()
    if scored.empty:
        return empty_band, empty_player

    scored["distance_band"] = _distance_band_series(scored["pass_distance"])
    scored["band_label"] = scored["distance_band"].map(DISTANCE_BAND_LABELS)
    scored["is_threat_m3"] = scored[col_m3] > thr_m3
    scored["is_threat_m4"] = scored[col_m4] > thr_m4

    band_rows: list[dict] = []
    for band in DISTANCE_BAND_ORDER:
        grp = scored[scored["distance_band"] == band]
        if grp.empty:
            continue
        n = int(len(grp))
        band_rows.append({
            "distance_band": band,
            "band_label": DISTANCE_BAND_LABELS[band],
            "passes": n,
            "mean_xp_m3": float(grp[col_m3].mean()),
            "mean_xp_m4": float(grp[col_m4].mean()),
            "threat_m3": int(grp["is_threat_m3"].sum()),
            "threat_m4": int(grp["is_threat_m4"].sum()),
            "pct_threat_m3": float(grp["is_threat_m3"].mean() * 100.0),
            "pct_threat_m4": float(grp["is_threat_m4"].mean() * 100.0),
        })
    band_summary = pd.DataFrame(band_rows)

    player_rows: list[dict] = []
    for (pid, band), grp in scored.groupby(["player_id", "distance_band"], sort=False):
        player_rows.append({
            "player_id": str(pid),
            "player_name": str(grp["player_name"].iloc[0]),
            "team": str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else grp["team"].iloc[0]),
            "distance_band": str(band),
            "band_label": DISTANCE_BAND_LABELS.get(str(band), str(band)),
            "passes": int(len(grp)),
            "mean_xp_m3": float(grp[col_m3].mean()),
            "mean_xp_m4": float(grp[col_m4].mean()),
            "threat_m3": int(grp["is_threat_m3"].sum()),
            "threat_m4": int(grp["is_threat_m4"].sum()),
        })
    player_summary = pd.DataFrame(player_rows)
    if not player_summary.empty:
        player_summary = player_summary.sort_values(
            ["distance_band", "threat_m3", "threat_m4", "mean_xp_m3"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)

    return band_summary, player_summary


@functools.lru_cache(maxsize=4)
def load_study_match_bundle(
    event_id: int = STUDY_MATCH_EVENT_ID,
    grid_preset: str = DEFAULT_GRID_PRESET,
) -> dict:
    grid = get_grid_config(grid_preset)
    empty = {
        "passes": pd.DataFrame(),
        "xp_grids_by_team": {},
        "count_grids_by_team": {},
        "league": {},
        "rankings_by_model": {},
        "comparison": pd.DataFrame(),
        "distance_study": pd.DataFrame(),
        "distance_study_by_player": pd.DataFrame(),
        "meta": {},
        "grid": grid,
    }
    frame = pe._load_season_pass_frame()
    if frame.empty:
        return empty

    match_frame = frame[frame["event_id"].astype(int) == int(event_id)].copy()
    if match_frame.empty:
        return empty

    passes = _enrich_match_passes(match_frame)
    passes = pe.filter_live_ball_passes(passes)
    if passes is None:
        passes = pd.DataFrame()

    league = _league_reference_surfaces(
        grid.dest_cols, grid.dest_rows,
        grid.od_origin_cols, grid.od_origin_rows,
        grid.od_dest_cols, grid.od_dest_rows,
    )
    xp_grids_by_team, count_grids_by_team = build_team_xp_surfaces(passes, grid)
    if not passes.empty:
        passes = _assign_study_xp_models(
            passes,
            grid=grid,
            count_grids_by_team=count_grids_by_team,
            league=league,
        )

    first = match_frame.iloc[0]
    home_team = str(first["home_team"])
    away_team = str(first["away_team"])
    meta = {
        "event_id": int(event_id),
        "home_team": home_team,
        "away_team": away_team,
        "match_date": str(first["match_date"])[:10],
        "pass_events": int(len(match_frame)),
        "live_ball_passes": int(len(passes)),
        "completed_passes": int((passes["is_won"] & passes["has_end"]).sum()) if not passes.empty else 0,
        "players": int(passes["player_id"].nunique()) if not passes.empty else 0,
        "home_completed": int(
            (passes["is_won"] & passes["has_end"] & (passes["team"] == home_team)).sum()
        ) if not passes.empty else 0,
        "away_completed": int(
            (passes["is_won"] & passes["has_end"] & (passes["team"] == away_team)).sum()
        ) if not passes.empty else 0,
        "league_matches": int(league.get("num_matches", 0)),
        "league_matches_serie_b": int(league.get("num_matches_serie_b", 0)),
        "league_matches_serie_a": int(league.get("num_matches_serie_a", 0)),
        "league_passes": int(league.get("league_passes", 0)),
        "blend_alpha": XP_BLEND_ALPHA,
        "xp_pass_max": XP_PASS_MAX,
        "grid_preset": grid.key,
        "grid_label": grid.label,
        "dest_cols": grid.dest_cols,
        "dest_rows": grid.dest_rows,
        "od_origin_cols": grid.od_origin_cols,
        "od_origin_rows": grid.od_origin_rows,
        "od_dest_cols": grid.od_dest_cols,
        "od_dest_rows": grid.od_dest_rows,
    }

    rankings_by_model = {
        model: rank_players_by_xp(passes, model=model)
        for model in STUDY_MODELS
    }
    comparison = build_model34_comparison_table(passes)
    distance_study, distance_study_by_player = build_distance_threat_study(passes)

    return {
        "passes": passes,
        "xp_grids_by_team": xp_grids_by_team,
        "count_grids_by_team": count_grids_by_team,
        "league": league,
        "rankings_by_model": rankings_by_model,
        "comparison": comparison,
        "distance_study": distance_study,
        "distance_study_by_player": distance_study_by_player,
        "meta": meta,
        "grid": grid,
    }
