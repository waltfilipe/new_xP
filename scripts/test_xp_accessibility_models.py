#!/usr/bin/env python3
"""Offline comparison of xP accessibility alternatives A (blend), B, and C."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import xp_engine as xe
import xp_stats_engine as xstats
import xp_study_engine as xse

GRID = xse.STUDY_GRID
SMOOTH = xse.XP_SMOOTHING
FIELD_X = xse.FIELD_X


def _scale_max_one(values: np.ndarray) -> np.ndarray:
    mx = float(np.max(values))
    if mx <= 0:
        return values.astype(float)
    return values * (xse.XP_PASS_MAX / mx)


def attach_cells(df: pd.DataFrame) -> pd.DataFrame:
    out = xe.attach_od_cells(df, GRID)
    if "progress_ratio" not in out.columns:
        out["progress_ratio"] = xse._progress_ratio_series(out)
    return out


def build_transition_model(completed: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """P(D|O) lookup and origin-conditional xP fake (Alt A base)."""
    o_rows, o_cols = GRID.od_origin_rows, GRID.od_origin_cols
    d_rows, d_cols = GRID.od_dest_rows, GRID.od_dest_cols
    tensor = xse._count_od_tensor(completed, GRID)
    origin_totals = tensor.sum(axis=(2, 3))
    lookup = np.zeros_like(tensor, dtype=float)
    for oy in range(o_rows):
        for ox in range(o_cols):
            denom = float(origin_totals[oy, ox] + SMOOTH * d_rows * d_cols)
            for dy in range(d_rows):
                for dx in range(d_cols):
                    count = float(tensor[oy, ox, dy, dx] + SMOOTH)
                    lookup[oy, ox, dy, dx] = count / denom
    fake = 1.0 / np.maximum(lookup, 1e-12)
    fake = _scale_max_one(fake)
    return lookup, fake


def vector_from_lookup(df: pd.DataFrame, lookup: np.ndarray) -> np.ndarray:
    oy = df["oy"].to_numpy(int)
    ox = df["ox"].to_numpy(int)
    dy = df["dy"].to_numpy(int)
    dx = df["dx"].to_numpy(int)
    return lookup[oy, ox, dy, dx].astype(float)


def cell_distance(oy: np.ndarray, ox: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    return np.sqrt((oy - dy) ** 2 + (ox - dx) ** 2)


def field_pressure(x_start: np.ndarray) -> np.ndarray:
    """0 = deep own half, 1 = attacking third."""
    return 1.0 / (1.0 + np.exp(-(x_start - 52.0) / 12.0))


def calibrate_by_band(
    values: np.ndarray,
    bands: np.ndarray,
    target_mean: dict[str, float],
    *,
    clip_lo: float = 0.0,
    clip_hi: float = 1.0,
) -> np.ndarray:
    """Anti-bias: restore per-band mean to baseline M4 levels."""
    out = values.copy()
    for band, target in target_mean.items():
        mask = bands == band
        if not mask.any():
            continue
        cur = out[mask]
        mu = float(cur.mean())
        if mu > 1e-9:
            scaled = cur * (target / mu)
        else:
            scaled = cur
        out[mask] = np.clip(scaled, clip_lo, clip_hi)
    return out


def model_a_blend(df: pd.DataFrame, fake_lookup: np.ndarray) -> np.ndarray:
    """Alt A: origin-conditional fake xP blended with M4."""
    xp_m4 = df["xp_m4"].to_numpy(float)
    xp_fake = vector_from_lookup(df, fake_lookup)
    # Blend: keep most of M4, inject local accessibility signal.
    w_fake = 0.35
    blended = (1.0 - w_fake) * xp_m4 + w_fake * xp_fake
    # Anti-bias: band mean preservation + cap shrink for easy deep shorts.
    bands = df["distance_band"].astype(str).to_numpy()
    target = {
        band: float(xp_m4[bands == band].mean())
        for band in np.unique(bands)
    }
    return calibrate_by_band(blended, bands, target)


def model_b_accessibility(df: pd.DataFrame) -> np.ndarray:
    """Alt B: locality kernel modulated by field pressure (production parity)."""
    xp_base = df["xp_m4"].to_numpy(float) if "xp_accessibility_mult" not in df.columns else (
        df["xp_m4"].to_numpy(float) / np.maximum(df["xp_accessibility_mult"].to_numpy(float), 1e-6)
    )
    mult = xe.accessibility_multiplier_array(df)
    return xp_base * mult


def model_c_conditional(df: pd.DataFrame, cond_lookup: np.ndarray) -> np.ndarray:
    """Alt C: surplus vs origin-conditional expectation (ratio form)."""
    xp_m4 = df["xp_m4"].to_numpy(float)
    xp_hier = df["xp_hier_od"].to_numpy(float) if "xp_hier_od" in df.columns else xp_m4.copy()
    p_cond = vector_from_lookup(df, cond_lookup)
    baseline = _scale_max_one(1.0 / np.maximum(p_cond, 1e-12))

    # Surplus ratio: global OD rarity relative to local transition ease.
    ratio = xp_hier / np.maximum(baseline, 1e-6)
    ratio = _scale_max_one(ratio)

    progress_mult = (
        df["xp_progress_mult"].to_numpy(float)
        if "xp_progress_mult" in df.columns
        else xse.progress_toward_goal_multiplier(df["progress_ratio"].to_numpy(float))
    )
    out = np.minimum(ratio * progress_mult, xse.XP_PASS_MAX)

    bands = df["distance_band"].astype(str).to_numpy()
    target = {band: float(xp_m4[bands == band].mean()) for band in np.unique(bands)}
    return calibrate_by_band(out, bands, target)


def threat_flags(residual: np.ndarray, bands: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    thr = np.array([thresholds.get(b, np.inf) for b in bands], dtype=float)
    return residual > thr


def summarize_passes(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for label, mask in [
        ("Augusto 6m (SB)", (df["player_name"].str.contains("Augusto", case=False, na=False))
         & df["team"].str.contains("Bernardo", case=False, na=False)
         & df["pass_distance"].between(5.5, 6.5)
         & df["xp_m4"].between(0.45, 0.53)),
        ("Short def. lateral (x<50, y wide)", (df["distance_band"] == "short")
         & (df["x_start"] < 50) & ((df["y_start"] < 25) | (df["y_start"] > 55))),
        ("Short terço final (x_end>72)", (df["distance_band"] == "short") & (df["x_end"] > 72)),
        ("Long >30m progressivo", (df["distance_band"] == "long") & (df["progress_ratio"] > 0.35)),
        ("Todos shorts", df["distance_band"] == "short"),
        ("Todos longs", df["distance_band"] == "long"),
        ("Todos passes", pd.Series(True, index=df.index)),
    ]:
        sub = df.loc[mask]
        if sub.empty:
            continue
        row = {"cohort": label, "n": len(sub)}
        for name, col in cols.items():
            row[f"{name}_mean"] = float(sub[col].mean())
            row[f"{name}_med"] = float(sub[col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def corr_by_band(df: pd.DataFrame, col: str) -> dict[str, float]:
    out = {}
    for band in ("short", "long"):
        sub = df[df["distance_band"] == band]
        if len(sub) > 10:
            out[band] = float(sub["pass_distance"].corr(sub[col]))
    return out


def main() -> None:
    print("Loading Serie B season passes with current xP...")
    season = xe.load_season_passes()
    mask = season["is_won"] & season["has_end"] & (season["ox"] >= 0) & (season["dx"] >= 0)
    sb = attach_cells(season.loc[mask].copy())

    print("Building global transition model (Alt A/C base)...")
    global_completed = attach_cells(xse._league_completed_passes())
    cond_lookup, fake_lookup = build_transition_model(global_completed)

    sb["xp_a"] = model_a_blend(sb, fake_lookup)
    sb["xp_b"] = model_b_accessibility(sb)
    sb["xp_c"] = model_c_conditional(sb, cond_lookup)

    thresholds = xe.load_threat_thresholds()
    bands = sb["distance_band"].astype(str).to_numpy()

    for tag, col in [("m4", "xp_m4"), ("A", "xp_a"), ("B", "xp_b"), ("C", "xp_c")]:
        expected_col = f"xp_expected_{tag}" if tag != "m4" else "xp_expected"
        if tag == "m4":
            sb["res_m4"] = sb["xp_residual"].to_numpy(float)
        else:
            # Re-use same Ridge expected; isolate effect of xP numerator change.
            sb[f"res_{tag.lower()}"] = sb[col].to_numpy(float) - sb["xp_expected"].to_numpy(float)
        sb[f"threat_{tag.lower()}"] = threat_flags(
            sb[f"res_{'m4' if tag == 'm4' else tag.lower()}"].to_numpy(float),
            bands,
            thresholds,
        )

    cols = {"m4": "xp_m4", "A": "xp_a", "B": "xp_b", "C": "xp_c"}
    cohort = summarize_passes(sb, cols)

    print("\n=== COORTES (média xP) ===")
    for _, r in cohort.iterrows():
        print(f"\n{r['cohort']} (n={int(r['n'])})")
        for k in ["m4", "A", "B", "C"]:
            print(f"  {k}: mean={r[f'{k}_mean']:.3f} med={r[f'{k}_med']:.3f}")

    # Augusto specific
    aug = sb[(sb["player_name"].str.contains("Augusto", case=False, na=False))
             & (sb["team"].str.contains("Bernardo", case=False, na=False))
             & sb["pass_distance"].between(5.5, 6.5)
             & sb["xp_m4"].between(0.45, 0.53)].iloc[0]
    print("\n=== PASSE AUGUSTO 6m ===")
    for k, col in [("M4", "xp_m4"), ("A", "xp_a"), ("B", "xp_b"), ("C", "xp_c")]:
        print(f"  {k}: {aug[col]:.3f} | threat={bool(aug[f'threat_{k.lower()}'] if k!='M4' else aug['is_threat_m4'])}")

    print("\n=== DISTRIBUIÇÃO GLOBAL (Série B) ===")
    for k, col in cols.items():
        print(
            f"{k}: mean={sb[col].mean():.3f} med={sb[col].median():.3f} "
            f"p90={sb[col].quantile(0.9):.3f}"
        )

    print("\n=== CORRELAÇÃO distância × xP (viés longo) ===")
    for k, col in cols.items():
        c = corr_by_band(sb, col)
        print(f"  {k}: short={c.get('short', float('nan')):.3f} long={c.get('long', float('nan')):.3f}")

    print("\n=== THREAT PASSES (mesmo limiar de resíduo, esperado fixo) ===")
    for k in ["m4", "a", "b", "c"]:
        col = f"threat_{k}"
        n = int(sb[col].sum())
        print(f"  {k.upper()}: {n} ({100*n/len(sb):.2f}%)")

    # Shift vs M4
    print("\n=== DELTA vs M4 (média) ===")
    for k, col in [("A", "xp_a"), ("B", "xp_b"), ("C", "xp_c")]:
        delta = sb[col] - sb["xp_m4"]
        print(
            f"  {k}: mean_delta={delta.mean():+.4f} | short={delta[sb.distance_band=='short'].mean():+.4f} "
            f"| long={delta[sb.distance_band=='long'].mean():+.4f}"
        )

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    report = {
        "models": {
            "A": "blend 65/35 M4 + origin-conditional fake; band mean calibration",
            "B": "multiplicative ease(locality, field pressure); beta by zone+dist; floor 0.68; band calibration",
            "C": "ratio xp_hier_od / baseline(P(D|O)); progress mult; band calibration",
        },
        "augusto_6m": {
            "xp_m4": float(aug["xp_m4"]),
            "xp_a": float(aug["xp_a"]),
            "xp_b": float(aug["xp_b"]),
            "xp_c": float(aug["xp_c"]),
            "threat_m4": bool(aug["is_threat_m4"]),
            "threat_a": bool(aug["threat_a"]),
            "threat_b": bool(aug["threat_b"]),
            "threat_c": bool(aug["threat_c"]),
        },
        "threat_counts": {k: int(sb[f"threat_{k}"].sum()) for k in ["m4", "a", "b", "c"]},
        "corr_distance": {k: corr_by_band(sb, col) for k, col in cols.items()},
        "cohort_means": cohort.to_dict(orient="records"),
    }
    path = out_dir / "xp_accessibility_model_comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {path}")


if __name__ == "__main__":
    main()
