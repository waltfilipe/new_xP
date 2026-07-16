#!/usr/bin/env python3
"""Build season xP artifacts (ridge model, threat thresholds, Serie B parquet)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xp_engine as xe


def main() -> None:
    print(f"Building xP season artifacts ({xe.XP_MODEL_VERSION})...")
    meta_threat = xe.fit_and_save_artifacts(force=True)
    print("Threat thresholds:", meta_threat.get("residual_threshold_labels"))
    season = xe.build_serie_b_season_passes(force_artifacts=False)
    print(f"Season passes: {len(season):,}")
    print(f"Threat passes: {int(season[xe.THREAT_COL].sum()):,}")
    _, players = xe.build_xp_analytics()
    print(f"Players with xP metrics: {len(players)}")
    if players:
        top = players[0]
        print(f"Top xP: {top['player_name']} ({top['xp_m4_total']:.1f})")


if __name__ == "__main__":
    main()
