"""Split midfielders by action-origin profile (retorno/misto vs campo ofensivo)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from heuristic_scoring import SB_FIELD_X

MIDFIELD_POSITION_CODES: frozenset[str] = frozenset({
    "CM", "CDM", "DM", "RCM", "LCM", "RDM", "LDM", "CAM",
})

LEGACY_MIDFIELD_GROUP = "midfielders"
CENTRAL_MIDFIELD_GROUP = "central_midfielders"
ATTACKING_MIDFIELD_GROUP = "attacking_midfielders"

MIDFIELD_RATING_GROUPS: frozenset[str] = frozenset({
    LEGACY_MIDFIELD_GROUP,
    CENTRAL_MIDFIELD_GROUP,
    ATTACKING_MIDFIELD_GROUP,
})

OFFENSIVE_ORIGIN_THRESHOLD = 58.0
MIN_ACTIONS_FOR_ORIGIN_SPLIT = 50
MIDLINE_X = SB_FIELD_X / 2.0


def is_midfield_position_code(position: str | None) -> bool:
    if not position:
        return False
    return str(position).strip().upper() in MIDFIELD_POSITION_CODES


def default_midfield_group(position: str | None) -> str:
    if str(position or "").strip().upper() == "CAM":
        return ATTACKING_MIDFIELD_GROUP
    return CENTRAL_MIDFIELD_GROUP


def _action_start_x_values(
    passes: pd.DataFrame | None,
    carries: pd.DataFrame | None,
) -> list[float]:
    from passes_engine import filter_live_ball_passes

    xs: list[float] = []
    if passes is not None and not passes.empty:
        work = filter_live_ball_passes(passes)
        if work is not None and not work.empty:
            if "is_won" in work.columns:
                work = work[work["is_won"].astype(bool)]
            if not work.empty and "x_start" in work.columns:
                xs.extend(work["x_start"].astype(float).tolist())
    if carries is not None and not carries.empty:
        work = carries
        if "is_dribble" in work.columns:
            work = work[~work["is_dribble"].astype(bool)]
        if "has_end" in work.columns:
            work = work[work["has_end"].astype(bool)]
        if not work.empty and "x_start" in work.columns:
            xs.extend(work["x_start"].astype(float).tolist())
    return xs


def offensive_origin_pct(
    player_id: str,
    passes_by_id: dict[str, pd.DataFrame],
    carries_by_id: dict[str, pd.DataFrame],
) -> float | None:
    """Share of pass/carry origins in the opponent's half (x >= midfield line)."""
    xs = _action_start_x_values(
        passes_by_id.get(str(player_id)),
        carries_by_id.get(str(player_id)),
    )
    if len(xs) < MIN_ACTIONS_FOR_ORIGIN_SPLIT:
        return None
    arr = np.asarray(xs, dtype=float)
    return float((arr >= MIDLINE_X).mean() * 100.0)


def resolve_midfield_position_group(
    player: dict,
    passes_by_id: dict[str, pd.DataFrame],
    carries_by_id: dict[str, pd.DataFrame],
) -> str:
    """Return central_midfielders or attacking_midfielders for a midfielder."""
    pos = str(player.get("position") or "").strip().upper()
    if not is_midfield_position_code(pos):
        return str(player.get("position_group") or "")

    pct = offensive_origin_pct(str(player["player_id"]), passes_by_id, carries_by_id)
    if pct is None:
        return default_midfield_group(pos)
    if pct >= OFFENSIVE_ORIGIN_THRESHOLD:
        return ATTACKING_MIDFIELD_GROUP
    return CENTRAL_MIDFIELD_GROUP


def apply_midfield_position_groups(
    players: list[dict],
    passes_by_id: dict[str, pd.DataFrame],
    carries_by_id: dict[str, pd.DataFrame],
) -> list[dict]:
    """Assign central vs attacking midfield groups from origin data."""
    enriched: list[dict] = []
    for player in players:
        pos = str(player.get("position") or "").strip().upper()
        current_group = str(player.get("position_group") or "")
        if not is_midfield_position_code(pos) and current_group not in MIDFIELD_RATING_GROUPS:
            enriched.append(dict(player))
            continue
        group = resolve_midfield_position_group(player, passes_by_id, carries_by_id)
        enriched.append({**player, "position_group": group})
    return enriched


def midfield_origin_fields(
    player: dict,
    passes_by_id: dict[str, pd.DataFrame],
    carries_by_id: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Optional diagnostic fields for dashboards."""
    pos = str(player.get("position") or "").strip().upper()
    if not is_midfield_position_code(pos):
        return {}
    pct = offensive_origin_pct(str(player["player_id"]), passes_by_id, carries_by_id)
    if pct is None:
        return {"midfield_offensive_origin_pct": None}
    return {
        "midfield_offensive_origin_pct": round(pct, 1),
        "midfield_origin_profile": (
            "campo_ofensivo" if pct >= OFFENSIVE_ORIGIN_THRESHOLD else "retorno_ou_misto"
        ),
    }
