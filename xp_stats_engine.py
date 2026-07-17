"""Extended xP player stats for the Stats tab."""

from __future__ import annotations

import numpy as np
import pandas as pd

import passes_engine as pe
import xp_study_engine as xse

FIELD_X = pe.FIELD_X
FIELD_Y = pe.FIELD_Y
FINAL_THIRD_X = 80.0
WIDE_Y_MAX = 20.0
WIDE_Y_MIN = 60.0
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


def _zone_x(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), "mid", dtype=object)
    out[x <= 40] = "def"
    out[x > 80] = "att"
    return out


def _sum_xp(mask: np.ndarray, xp: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(xp[mask].sum())


def _mean_xp(mask: np.ndarray, xp: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(xp[mask].mean())


def _count_threats(mask: np.ndarray, threat: np.ndarray) -> int:
    return int((mask & threat).sum())


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
    pr = scored["progress_ratio"].to_numpy(dtype=float)
    x_start = scored["x_start"].to_numpy(dtype=float)
    y_start = scored["y_start"].to_numpy(dtype=float)
    x_end = scored["x_end"].to_numpy(dtype=float)
    y_end = scored["y_end"].to_numpy(dtype=float)
    dist = scored["pass_distance"].to_numpy(dtype=float)

    start_zone = _zone_x(x_start)
    end_zone = _zone_x(x_end)
    wide = (y_start < WIDE_Y_MAX) | (y_start > WIDE_Y_MIN)
    central = ~wide

    prog = pr > 0
    forward = pr > 0.15
    static = pr <= 0

    xp_total = float(xp.sum())
    xp_prog = _sum_xp(prog, xp)
    progress_m = np.maximum(
        scored["x_end"].to_numpy(dtype=float) - scored["x_start"].to_numpy(dtype=float),
        0.0,
    )

    out: dict[str, float | int] = dict(base)
    out.update({
        "xp_prog_total": xp_prog,
        "xp_static_total": _sum_xp(static, xp),
        "xp_prog_share": xp_prog / xp_total if xp_total > 0 else 0.0,
        "xp_threat_forward": _count_threats(forward, threat),
        "xp_prog_efficiency": xp_prog / progress_m.sum() if progress_m.sum() > 0 else 0.0,
        "xp_total_def": _sum_xp(start_zone == "def", xp),
        "xp_total_mid": _sum_xp(start_zone == "mid", xp),
        "xp_total_att": _sum_xp(start_zone == "att", xp),
        "xp_threat_def": _count_threats(start_zone == "def", threat),
        "xp_threat_mid": _count_threats(start_zone == "mid", threat),
        "xp_threat_att": _count_threats(start_zone == "att", threat),
        "xp_final_third_share": _sum_xp(start_zone == "att", xp) / xp_total if xp_total > 0 else 0.0,
        "xp_from_deep": _sum_xp(start_zone == "def", xp),
        "xp_zone_lift_att_def": (
            _mean_xp(start_zone == "att", xp) - _mean_xp(start_zone == "def", xp)
        ),
        "xp_wide_total": _sum_xp(wide, xp),
        "xp_central_total": _sum_xp(central, xp),
        "xp_wide_share": _sum_xp(wide, xp) / xp_total if xp_total > 0 else 0.0,
        "xp_switch_total": _sum_xp(wide & (scored["distance_band"].to_numpy() == "long"), xp),
        "xp_line_break_total": _sum_xp((x_start < FINAL_THIRD_X) & (x_end > FINAL_THIRD_X), xp),
        "xp_build_up": _sum_xp(start_zone == "def", xp),
        "xp_finalization": _sum_xp(
            (x_end >= PENALTY_X_MIN) & (y_end >= PENALTY_Y_MIN) & (y_end <= PENALTY_Y_MAX),
            xp,
        ),
        "xp_max_pass": float(xp.max()) if n else 0.0,
        "xp_pass_std": float(xp.std()) if n > 1 else 0.0,
        "xp_pass_cv": float(xp.std() / xp.mean()) if n > 1 and xp.mean() > 0 else 0.0,
    })

    if "isHome" in scored.columns:
        home = scored["isHome"].astype(bool).to_numpy()
        out["xp_home_total"] = _sum_xp(home, xp)
        out["xp_away_total"] = _sum_xp(~home, xp)
        out["xp_home_share"] = out["xp_home_total"] / xp_total if xp_total > 0 else 0.0
    else:
        out["xp_home_total"] = 0.0
        out["xp_away_total"] = 0.0
        out["xp_home_share"] = 0.0

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
        for key in ("def", "mid", "att"):
            metrics[f"xp_threat_{key}_p90"] = 0.0
        return
    mins_f = float(minutes)
    factor = 90.0 / mins_f
    metrics["xp_per_90"] = float(metrics.get("xp_m4_total", 0.0)) * factor
    threat_total = int(metrics.get("xp_m4_threat_passes", 0))
    metrics["xp_m4_threat_passes_p90"] = float(threat_total) * factor
    for band in BANDS:
        band_threats = int(metrics.get(f"xp_m4_threat_{band}", 0))
        metrics[f"xp_m4_threat_{band}_p90"] = float(band_threats) * factor
    for key in ("def", "mid", "att"):
        threats = int(metrics.get(f"xp_threat_{key}", 0))
        metrics[f"xp_threat_{key}_p90"] = float(threats) * factor
    metrics["xp_threat_forward_p90"] = float(metrics.get("xp_threat_forward", 0)) * factor


# (section_title, metric_keys)
XP_STATS_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Volume & Effectiveness", (
        "xp_m4_total", "xp_per_90", "xp_m4_threat_passes_p90", "xp_m4_per_pass",
        "xp_m4_threat_rate", "passes_completed",
    )),
    (f"Short ({DISTANCE_BAND_LABELS['short']})", (
        "xp_dist_index_short",
        "xp_m4_per_pass_short",
        "xp_m4_threat_rate_short",
        "xp_m4_threat_short_p90",
        "passes_short",
    )),
    (f"Medium ({DISTANCE_BAND_LABELS['medium']})", (
        "xp_dist_index_medium",
        "xp_m4_per_pass_medium",
        "xp_m4_threat_rate_medium",
        "xp_m4_threat_medium_p90",
        "passes_medium",
    )),
    (f"Long ({DISTANCE_BAND_LABELS['long']})", (
        "xp_dist_index_long",
        "xp_m4_per_pass_long",
        "xp_m4_threat_rate_long",
        "xp_m4_threat_long_p90",
        "passes_long",
    )),
    ("Progressão e direção", (
        "xp_prog_total", "xp_static_total", "xp_prog_share",
        "xp_threat_forward_p90", "xp_prog_efficiency",
    )),
    ("Por zona do campo", (
        "xp_total_def", "xp_total_mid", "xp_total_att",
        "xp_threat_def_p90", "xp_threat_mid_p90", "xp_threat_att_p90",
        "xp_final_third_share", "xp_from_deep", "xp_zone_lift_att_def",
    )),
    ("Canais e rotas", (
        "xp_wide_total", "xp_central_total", "xp_wide_share",
        "xp_switch_total", "xp_line_break_total",
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
    ("Contexto", (
        "xp_build_up", "xp_finalization",
        "xp_home_total", "xp_away_total", "xp_home_share",
    )),
    ("Índices compostos", (
        "xp_threat_index", "xp_progressive_index", "xp_creator_score", "xp_builder_score",
    )),
)

XP_STATS_LABELS: dict[str, str] = {
    "xp_m4_total": "xP Total",
    "xp_per_90": "xP per 90",
    "xp_m4_threat_passes_p90": "xP Threat Passes (Per game)",
    "xp_m4_per_pass": "xP/Passe",
    "xp_m4_threat_rate": "% Threat Passes",
    "passes_completed": "Passes completados",
    "xp_dist_index_short": "Distance Index",
    "xp_dist_index_medium": "Distance Index",
    "xp_dist_index_long": "Distance Index",
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
    "xp_prog_total": "xP progressivo",
    "xp_static_total": "xP estático/recuado",
    "xp_prog_share": "% xP em progressão",
    "xp_threat_forward_p90": "Threat forward p/game",
    "xp_prog_efficiency": "Progress efficiency",
    "xp_total_def": "xP terço defensivo",
    "xp_total_mid": "xP terço médio",
    "xp_total_att": "xP terço ofensivo",
    "xp_threat_def_p90": "Threat p/game (Def)",
    "xp_threat_mid_p90": "Threat p/game (Mid)",
    "xp_threat_att_p90": "Threat p/game (Att)",
    "xp_final_third_share": "% xP no terço final",
    "xp_from_deep": "xP from deep",
    "xp_zone_lift_att_def": "Zone lift (Att − Def)",
    "xp_wide_total": "xP wide channels",
    "xp_central_total": "xP central corridor",
    "xp_wide_share": "% xP wide",
    "xp_switch_total": "Switch xP (long wide)",
    "xp_line_break_total": "Line-breaking xP",
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
    "xp_build_up": "xP in build-up",
    "xp_finalization": "xP in finalization",
    "xp_home_total": "xP em casa",
    "xp_away_total": "xP fora",
    "xp_home_share": "% xP em casa",
    "xp_threat_index": "xP Threat Index",
    "xp_progressive_index": "Progressive xP Index",
    "xp_creator_score": "Creator score",
    "xp_builder_score": "Builder score",
}

XP_STATS_RANK_METRICS: tuple[str, ...] = tuple(
    dict.fromkeys(
        key
        for _title, keys in XP_STATS_SECTIONS
        for key in keys
    )
)


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std())
    if std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


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
    """Within-position z-score index per distance band (quality + threat rate + threat p90)."""
    if not players:
        return
    pools: dict[str, list[dict]] = {}
    for player in players:
        group = str(player.get("position_group") or "CM")
        pools.setdefault(group, []).append(player)

    for rows in pools.values():
        df = pd.DataFrame(rows)
        for band in BANDS:
            per_pass_col = f"xp_m4_per_pass_{band}"
            rate_col = f"xp_m4_threat_rate_{band}"
            p90_col = f"xp_m4_threat_{band}_p90"
            z_per = _zscore(df.get(per_pass_col, pd.Series(0.0, index=df.index)).astype(float))
            z_rate = _zscore(df.get(rate_col, pd.Series(0.0, index=df.index)).astype(float))
            z_p90 = _zscore(df.get(p90_col, pd.Series(0.0, index=df.index)).astype(float))
            for i, row in enumerate(rows):
                row[f"xp_dist_index_{band}"] = float(z_per.iloc[i] + z_rate.iloc[i] + z_p90.iloc[i])


def attach_all_stats_ranks(players: list[dict]) -> None:
    """Rank every stats-tab metric within position group."""
    pools: dict[str, list[dict]] = {}
    for player in players:
        group = str(player.get("position_group") or "CM")
        pools.setdefault(group, []).append(player)
    for rows in pools.values():
        pool_size = len(rows)
        for metric in XP_STATS_RANK_METRICS:
            rows.sort(key=lambda row: float(row.get(metric) or 0.0), reverse=True)
            for rank, row in enumerate(rows, start=1):
                row[f"{metric}_rank_in_group"] = rank
                row[f"{metric}_rank_pool_in_group"] = pool_size


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
        return f"{val:.2f}"
    if key.endswith("_rate") or key.endswith("_share") or key.endswith("_pct") or key == "xp_surprise_rate" or key == "xp_threat_conversion":
        if key == "xp_m4_threat_rate" or key.startswith("xp_m4_threat_rate_"):
            return f"{100 * val:.1f}%"
        return f"{100 * val:.1f}%"
    if key.startswith("xp_m4_per_pass_"):
        return f"{val:.3f}"
    if key.startswith("xp_threat_index") or key.endswith("_index") or key.endswith("_score"):
        return f"{val:.2f}"
    if key == "xp_m4_per_pass" or key == "xp_prog_efficiency" or key == "xp_zone_lift_att_def":
        return f"{val:.3f}"
    if key.endswith("_p90") or key == "xp_per_90" or key == "xp_game_mean" or key == "xp_game_std":
        return f"{val:.2f}"
    if key == "xp_pass_cv" or key == "xp_pass_std":
        return f"{val:.3f}"
    if key == "xp_max_pass" or key == "xp_m4_p90":
        return f"{val:.3f}"
    return f"{val:.1f}"
