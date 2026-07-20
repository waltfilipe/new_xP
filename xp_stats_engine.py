"""Extended xP player stats for the Stats tab."""

from __future__ import annotations

import numpy as np
import pandas as pd

import passes_engine as pe
import xp_study_engine as xse

FIELD_X = pe.FIELD_X
FIELD_Y = pe.FIELD_Y
DEF_FIELD_SHARE = 0.40
FINAL_FIELD_SHARE = 0.40
DEF_X_MAX = FIELD_X * DEF_FIELD_SHARE
FINAL_X_MIN = FIELD_X * (1.0 - FINAL_FIELD_SHARE)
FIRST_THIRD_X = FIELD_X / 3.0
CENTRAL_Y_MIN = 20.0
CENTRAL_Y_MAX = 60.0
LATERAL_INNER_SHARE = 0.30
LINE_BREAK_FORWARD_ANGLE_DEG = 40.0
LINE_BREAK_FIELD_BEFORE_30_X = FIELD_X * 0.30
LINE_BREAK_FIELD_MID_50_X = FIELD_X * 0.50
LINE_BREAK_DIST_MIN_ATTACK_M = 10.0
LINE_BREAK_DIST_MIN_MID_M = 15.0
LINE_BREAK_DIST_MAX_M = 25.0
CROSS_DIST_MIN_M = 15.0
CROSS_LATERAL_DELTA_MIN_M = 8.0
CROSS_MAX_START_X = 102.0
FROM_DEEP_DIST_MIN_M = 15.0
PENALTY_X_MIN = pe.PENALTY_BOX_X_MIN
PENALTY_Y_MIN = pe.PENALTY_BOX_Y_MIN
PENALTY_Y_MAX = pe.PENALTY_BOX_Y_MAX

XP_COL = "xp_m4"
THREAT_COL = "is_threat_m4"
RESIDUAL_COL = "xp_residual"
DISTANCE_BAND_LABELS = xse.DISTANCE_BAND_LABELS
DISTANCE_SHORT_MAX_M = pe.DISTANCE_SHORT_MAX_M
DISTANCE_MEDIUM_MAX_M = pe.DISTANCE_MEDIUM_MAX_M
BANDS = xse.DISTANCE_BAND_ORDER
DISTANCE_INDEX_MIN_PASS_PERCENTILE = 20

DISTANCE_INDEX_GRADES: tuple[tuple[str, float], ...] = (
    ("Good", 0.20),
    ("Above Average", 0.40),
    ("Average", 0.60),
    ("Under Average", 0.80),
    ("Poor", 1.00),
)
DISTANCE_INDEX_GRADE_ORDER: dict[str, int] = {
    "Poor": 1,
    "Under Average": 2,
    "Average": 3,
    "Above Average": 4,
    "Good": 5,
}
# Skill metrics share the bulk of the index; volume enters with a small weight.
DISTANCE_INDEX_SKILL_WEIGHT = 0.30
DISTANCE_INDEX_VOLUME_WEIGHT = 0.10
DISTANCE_INDEX_BALANCE_MIN_WEIGHT = 0.40
DISTANCE_INDEX_BALANCE_MEAN_WEIGHT = 0.60
DISTANCE_INDEX_VOLUME_GRADE_PENALTY_PCTS: tuple[tuple[float, int], ...] = (
    (0.85, 2),
    (0.70, 1),
)


def _zone_x(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), "mid", dtype=object)
    out[x <= FIRST_THIRD_X] = "def"
    out[x > FINAL_X_MIN] = "att"
    return out


def _is_left_corridor(y: np.ndarray) -> np.ndarray:
    return y < CENTRAL_Y_MIN


def _is_right_corridor(y: np.ndarray) -> np.ndarray:
    return y > CENTRAL_Y_MAX


def _is_central_corridor(y: np.ndarray) -> np.ndarray:
    return (y >= CENTRAL_Y_MIN) & (y <= CENTRAL_Y_MAX)


def _is_lateral_corridor(y: np.ndarray) -> np.ndarray:
    return _is_left_corridor(y) | _is_right_corridor(y)


def _is_diagonal_long_pass(y_start: np.ndarray, y_end: np.ndarray) -> np.ndarray:
    """Swap laterally: left/central -> right, or right/central -> left."""
    to_right = (
        (_is_left_corridor(y_start) | _is_central_corridor(y_start))
        & _is_right_corridor(y_end)
    )
    to_left = (
        (_is_right_corridor(y_start) | _is_central_corridor(y_start))
        & _is_left_corridor(y_end)
    )
    return to_right | to_left


def _line_break_origin_corridor(y: np.ndarray) -> np.ndarray:
    """Central corridor plus the inner 30% of each lateral band (adjacent to center)."""
    left_inner = (y < CENTRAL_Y_MIN) & (y >= CENTRAL_Y_MIN * (1.0 - LATERAL_INNER_SHARE))
    right_inner = (y > CENTRAL_Y_MAX) & (
        y <= CENTRAL_Y_MAX + (FIELD_Y - CENTRAL_Y_MAX) * LATERAL_INNER_SHARE
    )
    central = _is_central_corridor(y)
    return central | left_inner | right_inner


def _line_break_distance_ok(x_start: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Distance bands by origin zone: none before 30%, 15–25 m at 30–50%, 10–25 m in attack."""
    before_30 = x_start <= LINE_BREAK_FIELD_BEFORE_30_X
    mid_zone = (x_start > LINE_BREAK_FIELD_BEFORE_30_X) & (x_start <= LINE_BREAK_FIELD_MID_50_X)
    attack_zone = x_start > LINE_BREAK_FIELD_MID_50_X
    return (
        ~before_30
        & (
            (mid_zone & (dist >= LINE_BREAK_DIST_MIN_MID_M) & (dist <= LINE_BREAK_DIST_MAX_M))
            | (attack_zone & (dist >= LINE_BREAK_DIST_MIN_ATTACK_M) & (dist <= LINE_BREAK_DIST_MAX_M))
        )
    )


def _is_forward_angle(dx: np.ndarray, dy: np.ndarray, *, max_angle_deg: float) -> np.ndarray:
    """True when the pass aims forward (+x) within ±max_angle_deg of the goal direction."""
    forward = dx > 0.0
    angle_deg = np.degrees(np.arctan2(dy, np.where(forward, dx, 1.0)))
    return forward & (np.abs(angle_deg) <= max_angle_deg)


def _is_left_right_inversion(y_start: np.ndarray, y_end: np.ndarray) -> np.ndarray:
    """Long pass that switches directly between left and right lateral corridors."""
    left_to_right = _is_left_corridor(y_start) & _is_right_corridor(y_end)
    right_to_left = _is_right_corridor(y_start) & _is_left_corridor(y_end)
    return left_to_right | right_to_left


def _in_penalty_box(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x >= PENALTY_X_MIN) & (y >= PENALTY_Y_MIN) & (y <= PENALTY_Y_MAX)


def _is_long_pass(scored: pd.DataFrame, dist: np.ndarray) -> np.ndarray:
    if "distance_band" in scored.columns:
        return scored["distance_band"].astype(str).to_numpy() == "long"
    return dist > DISTANCE_MEDIUM_MAX_M


SPECIAL_PASS_MAP_FILTERS: tuple[tuple[str, str], ...] = (
    ("diagonal_long", "Diagonal Longa"),
    ("line_break", "Quebra linha"),
    ("inversion", "Inversões"),
    ("cross", "Cruzamento"),
    ("from_deep", "xP from deep"),
    ("final_third", "% xP no terço final"),
    ("in_box", "% xP na área"),
)
SPECIAL_PASS_MAP_FILTER_KEYS: tuple[str, ...] = tuple(key for key, _label in SPECIAL_PASS_MAP_FILTERS)
SPECIAL_PASS_MAP_FILTER_LABELS: dict[str, str] = dict(SPECIAL_PASS_MAP_FILTERS)
SPECIAL_PASS_COUNT_KEYS: tuple[str, ...] = SPECIAL_PASS_MAP_FILTER_KEYS


def special_pass_count_key(filter_key: str) -> str:
    return f"special_{filter_key}"


def special_pass_per_game_key(filter_key: str) -> str:
    return f"special_{filter_key}_p90"


def _completed_pass_frame(passes: pd.DataFrame) -> pd.DataFrame:
    if passes is None or passes.empty:
        return pd.DataFrame()
    mask = passes["is_won"] & passes["has_end"] if "is_won" in passes.columns else passes["has_end"]
    return passes[mask].copy()


def compute_special_pass_masks(scored: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks for each special-pass category on completed passes."""
    n = len(scored)
    empty = np.zeros(n, dtype=bool)
    if scored is None or scored.empty:
        return {key: empty.copy() for key in SPECIAL_PASS_MAP_FILTER_KEYS}

    x_start = scored["x_start"].to_numpy(dtype=float)
    y_start = scored["y_start"].to_numpy(dtype=float)
    x_end = scored["x_end"].to_numpy(dtype=float)
    y_end = scored["y_end"].to_numpy(dtype=float)
    dist = scored["pass_distance"].to_numpy(dtype=float)
    dx = x_end - x_start
    dy = y_end - y_start

    start_zone = _zone_x(x_start)
    end_zone = _zone_x(x_end)
    long_pass = _is_long_pass(scored, dist)
    lateral_start = _is_lateral_corridor(y_start)
    lateral_end = _is_lateral_corridor(y_end)
    in_box = _in_penalty_box(x_end, y_end)

    return {
        "diagonal_long": (
            long_pass
            & (x_start <= DEF_X_MAX)
            & (x_end >= FINAL_X_MIN)
            & lateral_end
            & _is_diagonal_long_pass(y_start, y_end)
        ),
        "line_break": (
            _line_break_origin_corridor(y_start)
            & _is_central_corridor(y_end)
            & (x_end > x_start)
            & _line_break_distance_ok(x_start, dist)
            & _is_forward_angle(dx, dy, max_angle_deg=LINE_BREAK_FORWARD_ANGLE_DEG)
        ),
        "inversion": long_pass & _is_left_right_inversion(y_start, y_end),
        "cross": (
            lateral_start
            & (x_start >= FINAL_X_MIN)
            & (x_start < CROSS_MAX_START_X)
            & in_box
            & (dist >= CROSS_DIST_MIN_M)
            & (np.abs(dy) >= CROSS_LATERAL_DELTA_MIN_M)
        ),
        "from_deep": (
            (start_zone == "def")
            & (end_zone == "att")
            & (dist >= FROM_DEEP_DIST_MIN_M)
        ),
        "final_third": start_zone == "att",
        "in_box": in_box,
    }


def filter_passes_by_special_type(passes: pd.DataFrame, filter_key: str) -> pd.DataFrame:
    """Return completed passes matching a special-pass map filter."""
    work = _completed_pass_frame(passes)
    if work.empty:
        return work
    key = str(filter_key or "").strip()
    if key not in SPECIAL_PASS_MAP_FILTER_KEYS:
        return work.iloc[0:0].copy()
    masks = compute_special_pass_masks(work)
    return work.loc[masks[key]].copy()


def special_pass_map_label(filter_key: str) -> str:
    return SPECIAL_PASS_MAP_FILTER_LABELS.get(str(filter_key), str(filter_key))


def _sum_xp(mask: np.ndarray, xp: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(xp[mask].sum())


def compute_extended_xp_stats(grp: pd.DataFrame) -> dict[str, float | int]:
    """Compute full xP stat bundle for one player's season passes."""
    import xp_engine as xe

    base = xe.compute_player_xp_metrics(grp)
    if not base:
        return {}

    scored = grp[grp["is_won"] & grp["has_end"]].copy()
    if scored.empty or XP_COL not in scored.columns:
        return base

    xp = scored[XP_COL].to_numpy(dtype=float)
    n = len(scored)
    threat = (
        scored[THREAT_COL].to_numpy(dtype=bool)
        if THREAT_COL in scored.columns
        else np.zeros(n, dtype=bool)
    )
    if "progress_ratio" not in scored.columns:
        scored["progress_ratio"] = xse._progress_ratio_series(scored)

    xp_total = float(xp.sum())
    masks = compute_special_pass_masks(scored)

    out: dict[str, float | int] = dict(base)
    for sp_key in SPECIAL_PASS_COUNT_KEYS:
        out[special_pass_count_key(sp_key)] = int(masks[sp_key].sum())
    out.update({
        "xp_diagonal_long_total": _sum_xp(masks["diagonal_long"], xp),
        "xp_line_break_total": _sum_xp(masks["line_break"], xp),
        "xp_inversion_total": _sum_xp(masks["inversion"], xp),
        "xp_cross_total": _sum_xp(masks["cross"], xp),
        "xp_final_third_share": _sum_xp(masks["final_third"], xp) / xp_total if xp_total > 0 else 0.0,
        "xp_box_share": _sum_xp(masks["in_box"], xp) / xp_total if xp_total > 0 else 0.0,
        "xp_from_deep": _sum_xp(masks["from_deep"], xp),
        "xp_max_pass": float(xp.max()) if n else 0.0,
        "xp_pass_std": float(xp.std()) if n > 1 else 0.0,
        "xp_pass_cv": float(xp.std() / xp.mean()) if n > 1 and xp.mean() > 0 else 0.0,
    })

    if RESIDUAL_COL in scored.columns:
        residual = scored[RESIDUAL_COL].to_numpy(dtype=float)
        out["xp_residual_positive"] = float(np.maximum(residual, 0.0).sum())
        out["xp_residual_negative"] = float(np.minimum(residual, 0.0).sum())
        out["xp_surprise_rate"] = float((residual > 0).mean())
        p75 = float(np.quantile(xp, 0.75)) if n else 0.0
        high_xp = xp >= p75
        out["xp_threat_conversion"] = float(threat.sum() / high_xp.sum()) if high_xp.any() else 0.0
        if threat.any():
            out["xp_threat_mean_xp"] = float(xp[threat].mean())
            out["xp_threat_mean_residual"] = float(residual[threat].mean())
        else:
            out["xp_threat_mean_xp"] = 0.0
            out["xp_threat_mean_residual"] = 0.0
    else:
        out["xp_residual_positive"] = 0.0
        out["xp_residual_negative"] = 0.0
        out["xp_surprise_rate"] = 0.0
        out["xp_threat_conversion"] = 0.0
        out["xp_threat_mean_xp"] = 0.0
        out["xp_threat_mean_residual"] = 0.0

    if "event_id" in scored.columns:
        game_xp = scored.groupby("event_id")[XP_COL].sum()
        out["xp_game_mean"] = float(game_xp.mean()) if len(game_xp) else 0.0
        out["xp_game_std"] = float(game_xp.std()) if len(game_xp) > 1 else 0.0
        med = float(game_xp.median()) if len(game_xp) else 0.0
        out["xp_games_above_median_pct"] = float((game_xp > med).mean()) if len(game_xp) else 0.0
    else:
        out["xp_game_mean"] = 0.0
        out["xp_game_std"] = 0.0
        out["xp_games_above_median_pct"] = 0.0

    return out


def apply_per90_metrics(metrics: dict[str, float | int], minutes: float | None) -> None:
    """Add per-90 variants in place."""
    if not minutes or float(minutes) <= 0:
        metrics["xp_per_90"] = 0.0
        for sp_key in SPECIAL_PASS_COUNT_KEYS:
            metrics[special_pass_per_game_key(sp_key)] = 0.0
        return
    mins_f = float(minutes)
    factor = 90.0 / mins_f
    metrics["xp_per_90"] = float(metrics.get("xp_m4_total", 0.0)) * factor
    threat_total = int(metrics.get("xp_m4_threat_passes", 0))
    metrics["xp_m4_threat_passes_p90"] = float(threat_total) * factor
    for band in BANDS:
        band_threats = int(metrics.get(f"xp_m4_threat_{band}", 0))
        metrics[f"xp_m4_threat_{band}_p90"] = float(band_threats) * factor
    for sp_key in SPECIAL_PASS_COUNT_KEYS:
        count = int(metrics.get(special_pass_count_key(sp_key), 0))
        metrics[special_pass_per_game_key(sp_key)] = float(count) * factor


# (section_title, metric_keys)
XP_STATS_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Total", (
        "xp_per_90", "xp_m4_per_pass", "xp_m4_threat_rate",
    )),
    (f"Short ({DISTANCE_BAND_LABELS['short']})", (
        "xp_m4_per_pass_short", "xp_m4_threat_rate_short",
    )),
    (f"Medium ({DISTANCE_BAND_LABELS['medium']})", (
        "xp_m4_per_pass_medium", "xp_m4_threat_rate_medium",
    )),
    (f"Long ({DISTANCE_BAND_LABELS['long']})", (
        "xp_m4_per_pass_long", "xp_m4_threat_rate_long",
    )),
    ("SPECIAL PASSES", (
        "xp_diagonal_long_total", "special_diagonal_long_p90",
        "xp_line_break_total", "special_line_break_p90",
        "xp_inversion_total", "special_inversion_p90",
        "xp_cross_total", "special_cross_p90",
        "xp_final_third_share", "special_final_third_p90",
        "xp_box_share", "special_in_box_p90",
        "xp_from_deep", "special_from_deep_p90",
    )),
    ("Qualidade e threat", (
        "xp_residual_positive", "xp_residual_negative", "xp_surprise_rate",
        "xp_threat_conversion", "xp_threat_mean_xp", "xp_threat_mean_residual",
        "xp_m4_p90", "xp_max_pass",
    )),
    ("Consistência", (
        "xp_game_mean", "xp_game_std", "xp_pass_cv",
        "xp_games_above_median_pct", "xp_pass_std",
    )),
)


def p20_pass_thresholds_by_group(
    players: list[dict],
    passes_col: str,
    *,
    percentile: int = DISTANCE_INDEX_MIN_PASS_PERCENTILE,
) -> dict[str, float]:
    """Minimum passes at the position-group percentile (default P20)."""
    pools: dict[str, list[float]] = {}
    for player in players:
        group = str(player.get("position_group") or "CM")
        pools.setdefault(group, []).append(float(player.get(passes_col) or 0.0))
    return {
        group: float(np.percentile(counts, percentile)) if counts else 0.0
        for group, counts in pools.items()
    }


def iter_xp_stats_sections() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Yield (title, keys) for every Stats tab section."""
    for entry in XP_STATS_SECTIONS:
        if len(entry) == 2:
            title, keys = entry
            yield title, keys
        else:
            title, keys, _summary = entry
            yield title, keys


XP_STATS_LABELS: dict[str, str] = {
    "xp_per_90": "xP (Per game)",
    "xp_m4_per_pass": "xP/Passe",
    "xp_m4_threat_rate": "% Threat Passes",
    "xp_m4_per_pass_short": "xP/Passe",
    "xp_m4_per_pass_medium": "xP/Passe",
    "xp_m4_per_pass_long": "xP/Passe",
    "xp_m4_threat_rate_short": "% Threat Passes",
    "xp_m4_threat_rate_medium": "% Threat Passes",
    "xp_m4_threat_rate_long": "% Threat Passes",
    "xp_m4_threat_short_p90": "xP Threat Passes (Per game)",
    "xp_m4_threat_medium_p90": "xP Threat Passes (Per game)",
    "xp_m4_threat_long_p90": "xP Threat Passes (Per game)",
    "passes_short": "Passes na faixa",
    "passes_medium": "Passes na faixa",
    "passes_long": "Passes na faixa",
    "xp_m4_total_short": "xP Total (Short)",
    "xp_m4_threat_short_p90": "Threat p/game (Short)",
    "xp_m4_total_medium": "xP Total (Medium)",
    "xp_m4_threat_medium_p90": "Threat p/game (Medium)",
    "xp_m4_total_long": "xP Total (Long)",
    "xp_m4_threat_long_p90": "Threat p/game (Long)",
    "xp_diagonal_long_total": "Diagonal Longa (xP)",
    "special_diagonal_long_p90": "Diagonal Longa (Per game)",
    "xp_line_break_total": "Quebra linha (xP)",
    "special_line_break_p90": "Quebra linha (Per game)",
    "xp_inversion_total": "Inversões (xP)",
    "special_inversion_p90": "Inversões (Per game)",
    "xp_cross_total": "Cruzamento (xP)",
    "special_cross_p90": "Cruzamento (Per game)",
    "xp_final_third_share": "% xP no terço final",
    "special_final_third_p90": "Terço final (Per game)",
    "xp_box_share": "% xP na área",
    "special_in_box_p90": "Na área (Per game)",
    "xp_from_deep": "xP from deep",
    "special_from_deep_p90": "From deep (Per game)",
    "xp_residual_positive": "xP acima do esperado",
    "xp_residual_negative": "xP abaixo do esperado",
    "xp_surprise_rate": "Surprise rate",
    "xp_threat_conversion": "Threat conversion",
    "xp_threat_mean_xp": "Mean threat xP",
    "xp_threat_mean_residual": "Mean threat residual",
    "xp_m4_p90": "xP P90 (passe)",
    "xp_max_pass": "Max single-pass xP",
    "xp_game_mean": "xP médio por jogo",
    "xp_game_std": "Desvio xP entre jogos",
    "xp_pass_cv": "xP CV (passes)",
    "xp_games_above_median_pct": "% jogos acima da mediana",
    "xp_pass_std": "Desvio xP (passes)",
}

XP_STATS_RANK_METRICS: tuple[str, ...] = tuple(
    dict.fromkeys(
        key
        for _title, keys in iter_xp_stats_sections()
        for key in keys
    )
)


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std())
    if std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _rank_descending(values: pd.Series) -> pd.Series:
    return values.astype(float).rank(method="min", ascending=False)


def _grade_from_rank_pct(pct: float) -> str:
    for label, cutoff in DISTANCE_INDEX_GRADES:
        if pct <= cutoff:
            return label
    return DISTANCE_INDEX_GRADES[-1][0]


def _grade_from_tier_value(value: float) -> str:
    tier = max(1.0, min(5.0, float(value)))
    if tier >= 4.5:
        return "Good"
    if tier >= 3.5:
        return "Above Average"
    if tier >= 2.5:
        return "Average"
    if tier >= 1.5:
        return "Under Average"
    return "Poor"


def _balanced_grade_from_rank_pcts(rank_pcts: list[float]) -> str:
    """Blend mean and worst metric so one outlier cannot inflate the grade."""
    if not rank_pcts:
        return "Poor"
    tier_vals = [DISTANCE_INDEX_GRADE_ORDER[_grade_from_rank_pct(pct)] for pct in rank_pcts]
    blended = (
        DISTANCE_INDEX_BALANCE_MIN_WEIGHT * min(tier_vals)
        + DISTANCE_INDEX_BALANCE_MEAN_WEIGHT * float(np.mean(tier_vals))
    )
    return _grade_from_tier_value(blended)


def _apply_volume_grade_penalty(grade: str, volume_rank_pct: float) -> str:
    tier = DISTANCE_INDEX_GRADE_ORDER.get(grade, 3)
    for cutoff, steps in DISTANCE_INDEX_VOLUME_GRADE_PENALTY_PCTS:
        if volume_rank_pct > cutoff:
            tier -= steps
    return _grade_from_tier_value(float(max(1, tier)))


def attach_composite_indices(players: list[dict]) -> None:
    """Within-position z-score composites."""
    if not players:
        return
    pools: dict[str, list[dict]] = {}
    for player in players:
        group = str(player.get("position_group") or "CM")
        pools.setdefault(group, []).append(player)

    for rows in pools.values():
        df = pd.DataFrame(rows)
        z_total = _zscore(df["xp_m4_total"].astype(float))
        z_threat_p90 = _zscore(df["xp_m4_threat_passes_p90"].astype(float))
        z_threat_rate = _zscore(df["xp_m4_threat_rate"].astype(float))
        z_prog = _zscore(df.get("xp_prog_total", pd.Series(0.0, index=df.index)).astype(float))
        z_long_threat = _zscore(df.get("xp_m4_threat_long_p90", pd.Series(0.0, index=df.index)).astype(float))
        z_deep = _zscore(df.get("xp_from_deep", pd.Series(0.0, index=df.index)).astype(float))
        z_att = _zscore(df.get("xp_total_att", pd.Series(0.0, index=df.index)).astype(float))
        z_short_prog = _zscore(df.get("xp_m4_total_short", pd.Series(0.0, index=df.index)).astype(float))

        for i, row in enumerate(rows):
            row["xp_threat_index"] = float(z_total.iloc[i] + z_threat_p90.iloc[i] + z_threat_rate.iloc[i])
            row["xp_progressive_index"] = float(z_prog.iloc[i] + z_long_threat.iloc[i])
            row["xp_creator_score"] = float(z_att.iloc[i] + z_threat_p90.iloc[i] + z_long_threat.iloc[i])
            row["xp_builder_score"] = float(z_deep.iloc[i] + z_short_prog.iloc[i])


def attach_distance_indices(players: list[dict]) -> None:
    """Within-position index per band with balanced grades and light volume weight."""
    if not players:
        return
    pools: dict[str, list[dict]] = {}
    for player in players:
        group = str(player.get("position_group") or "CM")
        pools.setdefault(group, []).append(player)

    skill_weight = DISTANCE_INDEX_SKILL_WEIGHT
    volume_weight = DISTANCE_INDEX_VOLUME_WEIGHT

    for rows in pools.values():
        df = pd.DataFrame(rows)
        for band in BANDS:
            per_pass_col = f"xp_m4_per_pass_{band}"
            rate_col = f"xp_m4_threat_rate_{band}"
            p90_col = f"xp_m4_threat_{band}_p90"
            passes_col = f"passes_{band}"
            pass_counts = df.get(passes_col, pd.Series(0, index=df.index)).astype(float)
            min_passes = float(
                np.percentile(pass_counts.to_numpy(dtype=float), DISTANCE_INDEX_MIN_PASS_PERCENTILE)
            )
            eligible = pass_counts >= min_passes

            for i, row in enumerate(rows):
                row[f"xp_dist_index_{band}_min_passes"] = min_passes
                row[f"xp_dist_index_{band}_eligible"] = bool(eligible.iloc[i])
                row.pop(f"xp_dist_index_{band}_grade", None)

            if int(eligible.sum()) < 2:
                for row in rows:
                    row[f"xp_dist_index_{band}"] = None
                    row.pop(f"xp_dist_index_{band}_rank_in_group", None)
                    row.pop(f"xp_dist_index_{band}_rank_pool_in_group", None)
                continue

            sub = df.loc[eligible]
            pool_size = int(len(sub))
            z_per = _zscore(sub[per_pass_col].astype(float))
            z_rate = _zscore(sub[rate_col].astype(float))
            z_p90 = _zscore(sub[p90_col].astype(float))
            z_vol = _zscore(np.log1p(sub[passes_col].astype(float)))
            composite = (
                skill_weight * z_per
                + skill_weight * z_rate
                + skill_weight * z_p90
                + volume_weight * z_vol
            )

            rank_per = _rank_descending(sub[per_pass_col])
            rank_rate = _rank_descending(sub[rate_col])
            rank_p90 = _rank_descending(sub[p90_col])
            rank_vol = _rank_descending(sub[passes_col])

            eligible_rows = [rows[i] for i in sub.index]
            ranked = sorted(
                zip(eligible_rows, composite.tolist(), sub.index.tolist()),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            for rank, (row, z_val, sub_idx) in enumerate(ranked, start=1):
                row[f"xp_dist_index_{band}"] = float(z_val)
                row[f"xp_dist_index_{band}_rank_in_group"] = rank
                row[f"xp_dist_index_{band}_rank_pool_in_group"] = pool_size

                skill_pcts = [
                    float(rank_per.loc[sub_idx]) / pool_size,
                    float(rank_rate.loc[sub_idx]) / pool_size,
                    float(rank_p90.loc[sub_idx]) / pool_size,
                ]
                grade = _balanced_grade_from_rank_pcts(skill_pcts)
                vol_pct = float(rank_vol.loc[sub_idx]) / pool_size
                row[f"xp_dist_index_{band}_grade"] = _apply_volume_grade_penalty(grade, vol_pct)

            for i, row in enumerate(rows):
                if not eligible.iloc[i]:
                    row[f"xp_dist_index_{band}"] = None
                    row.pop(f"xp_dist_index_{band}_rank_in_group", None)
                    row.pop(f"xp_dist_index_{band}_rank_pool_in_group", None)
                    row.pop(f"xp_dist_index_{band}_grade", None)

        for row in rows:
            band_vals = [
                float(row[f"xp_dist_index_{band}"])
                for band in BANDS
                if row.get(f"xp_dist_index_{band}_eligible")
                and row.get(f"xp_dist_index_{band}") is not None
            ]
            row["xp_dist_index_mean"] = float(np.mean(band_vals)) if band_vals else None


def distance_index_grade(rank: int | None, total: int | None) -> str | None:
    """Legacy helper: map composite rank to a grade label."""
    if not rank or not total or rank <= 0 or total <= 0:
        return None
    return _grade_from_rank_pct(float(rank) / float(total))


def distance_index_grade_for_profile(profile: dict, band: str) -> str | None:
    if not profile.get(f"xp_dist_index_{band}_eligible", True):
        return None
    stored = profile.get(f"xp_dist_index_{band}_grade")
    if stored:
        return str(stored)
    return distance_index_grade(
        profile.get(f"xp_dist_index_{band}_rank_in_group"),
        profile.get(f"xp_dist_index_{band}_rank_pool_in_group"),
    )


def attach_all_stats_ranks(players: list[dict]) -> None:
    """Rank every stats-tab metric within position group."""
    pools: dict[str, list[dict]] = {}
    for player in players:
        group = str(player.get("position_group") or "CM")
        pools.setdefault(group, []).append(player)
    for rows in pools.values():
        pool_size = len(rows)
        for metric in XP_STATS_RANK_METRICS:
            if metric.startswith("xp_dist_index_"):
                continue
            rows.sort(key=lambda row: float(row.get(metric) or 0.0), reverse=True)
            for rank, row in enumerate(rows, start=1):
                row[f"{metric}_rank_in_group"] = rank
                row[f"{metric}_rank_pool_in_group"] = pool_size


def metric_qualitative_grade(profile: dict, key: str) -> str | None:
    rank = profile.get(f"{key}_rank_in_group")
    total = profile.get(f"{key}_rank_pool_in_group")
    if not rank or not total:
        return None
    return _grade_from_rank_pct(float(rank) / float(total))


def format_threat_rate_display(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{100 * float(value):.1f}%"


def stats_metric_label(key: str) -> str:
    return XP_STATS_LABELS.get(key, key)


def format_stats_value(key: str, value: float | int | None) -> str:
    if value is None:
        return "—"
    val = float(value)
    if key == "passes_completed":
        return f"{int(val):,}"
    if key.startswith("passes_"):
        return f"{int(val):,}"
    if key.startswith("xp_dist_index_"):
        if value is None:
            return "— (< P20)"
        return f"{val:.2f}"
    if key.startswith("xp_m4_threat_rate"):
        return f"{100 * val:.1f}%"
    if key.endswith("_rate") or key.endswith("_share") or key.endswith("_pct") or key == "xp_surprise_rate" or key == "xp_threat_conversion":
        return f"{100 * val:.1f}%"
    if key.startswith("xp_m4_per_pass_"):
        return f"{val:.3f}"
    if key == "xp_m4_per_pass":
        return f"{val:.3f}"
    if key.endswith("_p90") or key == "xp_per_90" or key == "xp_game_mean" or key == "xp_game_std":
        return f"{val:.2f}"
    if key == "xp_pass_cv" or key == "xp_pass_std":
        return f"{val:.3f}"
    if key == "xp_max_pass" or key == "xp_m4_p90":
        return f"{val:.3f}"
    return f"{val:.1f}"
