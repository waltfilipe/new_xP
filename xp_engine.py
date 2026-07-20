"""Season-wide xP Model 4 scoring, expected-xP regression and threat classification."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import passes_engine as pe
import xp_study_engine as xse

XP_DATA_CACHE_VERSION = 25
XP_POSITION_RANK_METRICS: tuple[str, ...] = (
    "xp_m4_total",
    "xp_m4_per_pass",
    "xp_m4_threat_passes_p90",
    "xp_m4_threat_rate",
    "xp_m4_total_short",
    "xp_m4_threat_short_p90",
    "xp_m4_total_medium",
    "xp_m4_threat_medium_p90",
    "xp_m4_total_long",
    "xp_m4_threat_long_p90",
)
XP_MODEL_VERSION = "m4_od_12x8_b4_a2_pr0_global_ita_dist15_ridge_global_v2"
THREAT_QUANTILE = 0.10
THREAT_PROGRESS_MIN = 0.0
XP_COL = "xp_m4"
XP_EXPECTED_COL = "xp_expected"
XP_RESIDUAL_COL = "xp_residual"
THREAT_COL = "is_threat_m4"

GRID = xse.STUDY_GRID
BANDS = xse.DISTANCE_BAND_ORDER
BAND_LABELS = xse.DISTANCE_BAND_LABELS

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RIDGE_MODEL_PATH = MODELS_DIR / "xp_expected_ridge.joblib"
THREAT_THRESHOLDS_PATH = MODELS_DIR / "xp_threat_quantile.json"
XP_PASSES_PARQUET = DATA_DIR / "xp_passes_serieb.parquet"
XP_META_PATH = DATA_DIR / "xp_season_meta.json"


def _n_origin_cells() -> int:
    return GRID.od_origin_rows * GRID.od_origin_cols


def _n_dest_cells() -> int:
    return GRID.od_dest_rows * GRID.od_dest_cols


def attach_od_cells(passes: pd.DataFrame, grid: xse.GridConfig = GRID) -> pd.DataFrame:
    out = passes.copy()
    mask = out["is_won"] & out["has_end"]
    out["ox"] = -1
    out["oy"] = -1
    out["dx"] = -1
    out["dy"] = -1
    if not mask.any():
        out["distance_band"] = xse._distance_band_series(out["pass_distance"])
        return out
    sub = out.loc[mask]
    ox, oy = xse._cell_indices(
        sub["x_start"].to_numpy(dtype=float),
        sub["y_start"].to_numpy(dtype=float),
        cols=grid.od_origin_cols,
        rows=grid.od_origin_rows,
    )
    dx, dy = xse._cell_indices(
        sub["x_end"].to_numpy(dtype=float),
        sub["y_end"].to_numpy(dtype=float),
        cols=grid.od_dest_cols,
        rows=grid.od_dest_rows,
    )
    out.loc[mask, "ox"] = ox
    out.loc[mask, "oy"] = oy
    out.loc[mask, "dx"] = dx
    out.loc[mask, "dy"] = dy
    out["distance_band"] = xse._distance_band_series(out["pass_distance"])
    return out


def _progress_ratio_array(df: pd.DataFrame) -> np.ndarray:
    if "progress_ratio" in df.columns:
        return df["progress_ratio"].to_numpy(dtype=float)
    dist = np.maximum(df["pass_distance"].to_numpy(dtype=float), 0.5)
    dx = df["x_end"].to_numpy(dtype=float) - df["x_start"].to_numpy(dtype=float)
    return np.clip(dx / dist, -1.0, 1.0)


def _build_design_matrix(df: pd.DataFrame, grid: xse.GridConfig = GRID) -> sparse.csr_matrix:
    dist = df["pass_distance"].to_numpy(dtype=float)
    dist_feats = np.column_stack([dist, dist ** 2, np.sqrt(dist)])
    progress = _progress_ratio_array(df)
    progress_feats = np.column_stack([progress, progress ** 2])
    n = len(df)
    o_idx = df["oy"].to_numpy(int) * grid.od_origin_cols + df["ox"].to_numpy(int)
    d_idx = df["dy"].to_numpy(int) * grid.od_dest_cols + df["dx"].to_numpy(int)
    n_o = _n_origin_cells()
    n_d = _n_dest_cells()
    return sparse.hstack([
        sparse.csr_matrix(dist_feats),
        sparse.csr_matrix(progress_feats),
        sparse.csr_matrix((np.ones(n), (np.arange(n), o_idx)), shape=(n, n_o)),
        sparse.csr_matrix((np.ones(n), (np.arange(n), d_idx)), shape=(n, n_d)),
    ])


def score_match_passes_m4(
    match_frame: pd.DataFrame,
    league: dict[str, np.ndarray | float | int],
    *,
    grid: xse.GridConfig = GRID,
    team_season_od: dict[str, np.ndarray] | None = None,
    team_n_matches: dict[str, int] | None = None,
) -> pd.DataFrame:
    passes = xse._enrich_match_passes(match_frame)
    passes = pe.filter_live_ball_passes(passes)
    if passes is None or passes.empty:
        return pd.DataFrame()
    _, count_grids = xse.build_team_xp_surfaces(passes, grid)
    scored = xse._assign_study_xp_models(
        passes,
        grid=grid,
        count_grids_by_team=count_grids,
        league=league,
        team_season_od=team_season_od,
        team_n_matches=team_n_matches,
    )
    scored = attach_od_cells(scored, grid)
    if "progress_ratio" not in scored.columns:
        scored["progress_ratio"] = xse._progress_ratio_series(scored)
    progress_mult = xse.progress_toward_goal_multiplier(scored["progress_ratio"].to_numpy(dtype=float))
    scored["xp_progress_mult"] = progress_mult
    scored[XP_COL] = (
        scored[xse.XP_MODEL_COLUMNS[xse.XP_MODEL_HIER_OD]].to_numpy(dtype=float) * progress_mult
    )
    return scored


def _build_team_season_od_maps(
    frame: pd.DataFrame,
    *,
    grid: xse.GridConfig = GRID,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    enriched_chunks: list[pd.DataFrame] = []
    for eid in frame["event_id"].astype(int).unique():
        mf = frame[frame["event_id"].astype(int) == int(eid)]
        ep = xse._enrich_match_passes(mf)
        ep = pe.filter_live_ball_passes(ep)
        if ep is not None and not ep.empty:
            enriched_chunks.append(ep[ep["is_won"] & ep["has_end"]])
    if not enriched_chunks:
        return {}, {}
    all_comp = pd.concat(enriched_chunks, ignore_index=True)
    team_season_od: dict[str, np.ndarray] = {}
    team_n_matches: dict[str, int] = {}
    for team, grp in all_comp.groupby("team", sort=False):
        team_season_od[str(team)] = xse._count_od_tensor(grp, grid)
        team_n_matches[str(team)] = int(grp["event_id"].nunique())
    return team_season_od, team_n_matches


def _score_frame_completed(
    frame: pd.DataFrame,
    league_ref: dict[str, np.ndarray | float | int],
    team_season_od: dict[str, np.ndarray] | None = None,
    team_n_matches: dict[str, int] | None = None,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for eid in frame["event_id"].astype(int).unique():
        mf = frame[frame["event_id"].astype(int) == int(eid)].copy()
        scored = score_match_passes_m4(
            mf,
            league_ref,
            team_season_od=team_season_od or {},
            team_n_matches=team_n_matches or {},
        )
        if scored.empty:
            continue
        comp = scored[scored["is_won"] & scored["has_end"]].copy()
        if comp.empty:
            continue
        chunks.append(comp)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _fit_artifacts_on_passes(train_passes: pd.DataFrame) -> dict:
    train = train_passes[
        (train_passes["ox"] >= 0)
        & (train_passes["dx"] >= 0)
    ].copy()
    if train.empty:
        raise RuntimeError("No completed passes available for xP artifact training.")

    X = _build_design_matrix(train)
    y = train[XP_COL].to_numpy(dtype=float)
    model = Pipeline([
        ("ridge", Ridge(alpha=10.0, solver="lsqr")),
    ])
    model.fit(X, y)
    joblib.dump(model, RIDGE_MODEL_PATH)

    train[XP_EXPECTED_COL] = model.predict(X)
    train[XP_RESIDUAL_COL] = train[XP_COL].to_numpy() - train[XP_EXPECTED_COL]

    thresholds: dict[str, float] = {}
    for band in BANDS:
        sub = train[train["distance_band"] == band]
        if sub.empty:
            thresholds[band] = 0.0
        else:
            thresholds[band] = float(sub[XP_RESIDUAL_COL].quantile(1.0 - THREAT_QUANTILE))

    meta = {
        "version": XP_MODEL_VERSION,
        "threat_quantile": THREAT_QUANTILE,
        "threat_progress_min": THREAT_PROGRESS_MIN,
        "residual_thresholds": thresholds,
        "residual_threshold_labels": {BAND_LABELS[k]: v for k, v in thresholds.items()},
        "progress_floor_mult": xse.XP_PROGRESS_FLOOR_MULT,
        "progress_logistic_k": xse.XP_PROGRESS_LOGISTIC_K,
        "blend_alpha": xse.XP_BLEND_ALPHA,
        "grid": "12x8",
        "training_pool": "global",
        "team_surface": "global_reference_only",
        "training_passes": int(len(train)),
        "training_matches": int(train["event_id"].nunique()) if "event_id" in train.columns else 0,
    }
    with open(THREAT_THRESHOLDS_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _load_global_scored_completed_passes() -> pd.DataFrame:
    """Score completed passes from the global multi-league pool for Ridge training."""
    league_ref = xse._league_reference_surfaces(
        GRID.dest_cols, GRID.dest_rows,
        GRID.od_origin_cols, GRID.od_origin_rows,
        GRID.od_dest_cols, GRID.od_dest_rows,
    )
    frame = xse._load_combined_league_pass_frame()
    if frame.empty:
        raise RuntimeError("No passes available in the global xP reference pool.")
    return _score_frame_completed(frame, league_ref)


def fit_and_save_artifacts(*, force: bool = False) -> dict:
    """Train expected-xP ridge and quantile threat thresholds on the global pass pool."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if (
        not force
        and RIDGE_MODEL_PATH.exists()
        and THREAT_THRESHOLDS_PATH.exists()
    ):
        with open(THREAT_THRESHOLDS_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
        if str(meta.get("version", "")) == XP_MODEL_VERSION:
            return meta

    league_passes = _load_global_scored_completed_passes()
    if league_passes.empty:
        raise RuntimeError("No globally scored passes available for xP artifact training.")
    return _fit_artifacts_on_passes(league_passes)


def load_threat_thresholds() -> dict[str, float]:
    if not THREAT_THRESHOLDS_PATH.exists():
        fit_and_save_artifacts()
    with open(THREAT_THRESHOLDS_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)
    return {str(k): float(v) for k, v in meta["residual_thresholds"].items()}


def load_expected_model() -> Pipeline:
    if not RIDGE_MODEL_PATH.exists():
        fit_and_save_artifacts()
    return joblib.load(RIDGE_MODEL_PATH)


def apply_expected_and_threat(passes: pd.DataFrame) -> pd.DataFrame:
    out = passes.copy()
    out[XP_EXPECTED_COL] = 0.0
    out[XP_RESIDUAL_COL] = 0.0
    out[THREAT_COL] = False
    mask = out["is_won"] & out["has_end"] & (out["ox"] >= 0) & (out["dx"] >= 0)
    if not mask.any():
        return out

    model = load_expected_model()
    thresholds = load_threat_thresholds()
    sub_idx = out.index[mask]
    sub = out.loc[mask]
    X = _build_design_matrix(sub)
    expected = model.predict(X)
    residual = sub[XP_COL].to_numpy(dtype=float) - expected
    out.loc[sub_idx, XP_EXPECTED_COL] = expected
    out.loc[sub_idx, XP_RESIDUAL_COL] = residual

    threat_flags = np.zeros(len(sub), dtype=bool)
    bands = sub["distance_band"].astype(str).to_numpy()
    for i, band in enumerate(bands):
        threat_flags[i] = residual[i] > thresholds.get(band, np.inf)
    if THREAT_PROGRESS_MIN is not None:
        progress = _progress_ratio_array(sub)
        threat_flags &= progress >= THREAT_PROGRESS_MIN
    out.loc[sub_idx, THREAT_COL] = threat_flags
    return out


def build_serie_b_season_passes(*, force_artifacts: bool = False) -> pd.DataFrame:
    league_ref = xse._league_reference_surfaces(
        GRID.dest_cols, GRID.dest_rows,
        GRID.od_origin_cols, GRID.od_origin_rows,
        GRID.od_dest_cols, GRID.od_dest_rows,
    )
    frame = pe._load_season_pass_frame()
    if frame.empty:
        return pd.DataFrame()

    team_season_od, team_n_matches = _build_team_season_od_maps(frame)

    chunks: list[pd.DataFrame] = []
    for eid in frame["event_id"].astype(int).unique():
        mf = frame[frame["event_id"].astype(int) == int(eid)].copy()
        scored = score_match_passes_m4(
            mf,
            league_ref,
            team_season_od=team_season_od or {},
            team_n_matches=team_n_matches or {},
        )
        if scored.empty:
            continue
        chunks.append(scored)

    if not chunks:
        return pd.DataFrame()
    season_raw = pd.concat(chunks, ignore_index=True)

    need_fit = force_artifacts
    if not need_fit and THREAT_THRESHOLDS_PATH.exists():
        with open(THREAT_THRESHOLDS_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
        need_fit = str(meta.get("version", "")) != XP_MODEL_VERSION
    else:
        need_fit = True

    if need_fit:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        global_passes = _load_global_scored_completed_passes()
        _fit_artifacts_on_passes(global_passes)

    season = apply_expected_and_threat(season_raw)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    season.to_parquet(XP_PASSES_PARQUET, index=False)
    meta = {
        "version": XP_MODEL_VERSION,
        "passes": int(len(season)),
        "completed": int((season["is_won"] & season["has_end"]).sum()),
        "threats": int(season[THREAT_COL].sum()),
        "players": int(season["player_id"].nunique()),
        "matches": int(season["event_id"].nunique()) if "event_id" in season.columns else 0,
    }
    with open(XP_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return season


def load_season_passes(*, rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not XP_PASSES_PARQUET.exists():
        return build_serie_b_season_passes(force_artifacts=rebuild)
    df = pd.read_parquet(XP_PASSES_PARQUET)
    if (
        THREAT_COL not in df.columns
        or XP_COL not in df.columns
        or "xp_progress_mult" not in df.columns
    ):
        return build_serie_b_season_passes(force_artifacts=True)
    if XP_META_PATH.exists():
        with open(XP_META_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
        if str(meta.get("version", "")) != XP_MODEL_VERSION:
            return build_serie_b_season_passes(force_artifacts=True)
    return df


@functools.lru_cache(maxsize=4)
def load_xp_passes_grouped(cache_version: int = XP_DATA_CACHE_VERSION) -> dict[str, pd.DataFrame]:
    _ = cache_version
    season = load_season_passes()
    if season.empty:
        return {}
    return {str(pid): grp for pid, grp in season.groupby("player_id", sort=False)}


def compute_player_xp_metrics(grp: pd.DataFrame) -> dict[str, float | int]:
    scored = grp[grp["is_won"] & grp["has_end"]]
    if scored.empty or XP_COL not in scored.columns:
        return {}
    n_passes = len(scored)
    out: dict[str, float | int] = {
        "xp_m4_total": float(scored[XP_COL].sum()),
        "xp_m4_per_pass": float(scored[XP_COL].mean()),
        "xp_m4_p90": float(scored[XP_COL].quantile(0.90)),
        "xp_m4_threat_passes": int(scored[THREAT_COL].sum()) if THREAT_COL in scored.columns else 0,
        "xp_m4_threat_rate": float(scored[THREAT_COL].mean()) if THREAT_COL in scored.columns else 0.0,
        "pass_mean_distance": (
            float(scored["pass_distance"].mean())
            if n_passes and "pass_distance" in scored.columns
            else 0.0
        ),
    }
    for band in BANDS:
        sub = scored[scored["distance_band"] == band]
        n_band = len(sub)
        out[f"passes_{band}"] = int(n_band)
        out[f"xp_m4_threat_{band}"] = int(sub[THREAT_COL].sum()) if THREAT_COL in sub.columns and n_band else 0
        out[f"xp_m4_mean_{band}"] = float(sub[XP_COL].mean()) if n_band else 0.0
        out[f"xp_m4_per_pass_{band}"] = float(sub[XP_COL].mean()) if n_band else 0.0
        out[f"xp_m4_total_{band}"] = float(sub[XP_COL].sum()) if n_band else 0.0
        out[f"xp_m4_threat_rate_{band}"] = (
            float(sub[THREAT_COL].mean()) if THREAT_COL in sub.columns and n_band else 0.0
        )
    return out


def build_xp_analytics(
    cache_version: int = XP_DATA_CACHE_VERSION,
) -> tuple[list[dict], list[dict]]:
    import xp_stats_engine as xstats

    _ = cache_version
    season = load_season_passes()
    frame = pe._load_season_pass_frame()
    if season.empty or frame.empty:
        return [], []

    registry = pe.build_player_registry(frame)
    minutes_info = pe._load_minutes_info(frame)
    players: list[dict] = []

    for player in registry:
        if not pe.is_outfield_position(player.get("position")):
            continue
        pid = player["code"]
        grp = season[season["player_id"].astype(str) == str(pid)]
        if grp.empty:
            continue
        mins = minutes_info.get(pid, {})
        metrics = xstats.compute_extended_xp_stats(grp)
        if not metrics:
            continue
        minutes = mins.get("minutes")
        xstats.apply_per90_metrics(metrics, minutes)
        players.append({
            "player_id": pid,
            "player_name": player["name"],
            "position": player.get("position", "—"),
            "position_group": pe.rating_position_group(player.get("position")),
            "team": mins.get("team", str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else "—")),
            "minutes": mins.get("minutes"),
            "minutes_pct": mins.get("minutes_pct"),
            "passes_completed": int((grp["is_won"] & grp["has_end"]).sum()),
            **metrics,
        })

    players.sort(key=lambda p: float(p.get("xp_m4_total", 0.0)), reverse=True)
    for i, p in enumerate(players, start=1):
        p["xp_m4_rank"] = i
    xstats.attach_distance_indices(players)
    xstats.attach_all_stats_ranks(players)
    attach_xp_metric_ranks(players)
    return registry, players


def attach_xp_metric_ranks(players: list[dict]) -> None:
    """Attach within-position ranks for core xP dashboard metrics."""
    pools: dict[str, list[dict]] = {}
    for player in players:
        group = str(player.get("position_group") or "CM")
        pools.setdefault(group, []).append(player)
    for rows in pools.values():
        pool_size = len(rows)
        for metric in XP_POSITION_RANK_METRICS:
            rows.sort(key=lambda row: float(row.get(metric) or 0.0), reverse=True)
            for rank, row in enumerate(rows, start=1):
                row[f"{metric}_rank_in_group"] = rank
                row[f"{metric}_rank_pool_in_group"] = pool_size


def rank_xp_players_by_position(players: list[dict]) -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {}
    for p in players:
        grp = str(p.get("position_group") or "CM")
        pools.setdefault(grp, []).append(p)
    for grp, rows in pools.items():
        rows.sort(key=lambda r: float(r.get("xp_m4_total", 0.0)), reverse=True)
        for i, row in enumerate(rows, start=1):
            row["xp_m4_rank_in_group"] = i
    return pools


def season_meta() -> dict:
    if XP_META_PATH.exists():
        with open(XP_META_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    season = load_season_passes()
    if season.empty:
        return {}
    return {
        "version": XP_MODEL_VERSION,
        "passes": int(len(season)),
        "threats": int(season[THREAT_COL].sum()),
        "players": int(season["player_id"].nunique()),
    }
