"""Maps for the xP study tab."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mplsoccer import Pitch

from xp_study_engine import FIELD_X, FIELD_Y, XP_GRID_COLS, XP_GRID_ROWS, XP_PASS_MAX

FIG_W, FIG_H = 8.4, 5.6
FIG_DPI = 220
ARROW_WIDTH = 0.9
ARROW_HEADWIDTH = 1.3
ARROW_HEADLENGTH = 1.3
CMAP_XP = plt.cm.plasma


def _base_pitch(*, figsize: tuple[float, float] = (FIG_W, FIG_H), dpi: int = FIG_DPI):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(dpi)
    return fig, ax, pitch


def _delicate_arrows(pitch, ax, x1, y1, x2, y2, color, *, alpha: float, lw_scale: float = 1.0) -> None:
    pitch.arrows(
        x1, y1, x2, y2,
        color=color,
        width=ARROW_WIDTH * lw_scale,
        headwidth=ARROW_HEADWIDTH * lw_scale,
        headlength=ARROW_HEADLENGTH * lw_scale,
        ax=ax,
        zorder=4,
        alpha=alpha,
    )


def draw_xp_destination_surface(
    xp_grid: np.ndarray,
    count_grid: np.ndarray,
    *,
    title: str,
    dest_cols: int = XP_GRID_COLS,
    dest_rows: int = XP_GRID_ROWS,
):
    """Background heatmap of destination-cell xP weights for the match."""
    fig, ax, pitch = _base_pitch()
    rows, cols = xp_grid.shape
    dest_rows = rows
    dest_cols = cols
    x_bins = np.linspace(0.0, FIELD_X, dest_cols + 1)
    y_bins = np.linspace(0.0, FIELD_Y, dest_rows + 1)
    norm = Normalize(vmin=0.0, vmax=XP_PASS_MAX)
    for iy in range(dest_rows):
        for ix in range(dest_cols):
            if count_grid[iy, ix] <= 0:
                continue
            rect = Rectangle(
                (x_bins[ix], y_bins[iy]),
                x_bins[ix + 1] - x_bins[ix],
                y_bins[iy + 1] - y_bins[iy],
                facecolor=CMAP_XP(norm(float(xp_grid[iy, ix]))),
                edgecolor="#334155",
                linewidth=0.6,
                alpha=0.82,
                zorder=1,
            )
            ax.add_patch(rect)
    sm = ScalarMappable(norm=norm, cmap=CMAP_XP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("xP destino (raridade)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax.set_title(title, color="white", fontsize=10, pad=8)
    return fig


def draw_top_xp_passes_map(
    top_passes,
    *,
    player_name: str,
    match_label: str,
    xp_grid: np.ndarray | None = None,
    dest_cols: int = XP_GRID_COLS,
    dest_rows: int = XP_GRID_ROWS,
):
    """Top-N xP passes for one player, color-coded by xP value."""
    fig, ax, pitch = _base_pitch()

    if xp_grid is not None:
        dest_rows, dest_cols = xp_grid.shape
        x_bins = np.linspace(0.0, FIELD_X, dest_cols + 1)
        y_bins = np.linspace(0.0, FIELD_Y, dest_rows + 1)
        norm = Normalize(vmin=0.0, vmax=XP_PASS_MAX)
        for iy in range(dest_rows):
            for ix in range(dest_cols):
                rect = Rectangle(
                    (x_bins[ix], y_bins[iy]),
                    x_bins[ix + 1] - x_bins[ix],
                    y_bins[iy + 1] - y_bins[iy],
                    facecolor=CMAP_XP(norm(float(xp_grid[iy, ix]))),
                    edgecolor="none",
                    alpha=0.18,
                    zorder=1,
                )
                ax.add_patch(rect)

    if top_passes is None or top_passes.empty:
        ax.text(60, 40, "Sem passes com xP", ha="center", va="center", color="white", fontsize=10)
        ax.set_title(f"{player_name}\nTop passes xP · {match_label}", color="white", fontsize=10, pad=8)
        return fig

    values = top_passes["xp_value"].to_numpy(dtype=float)
    norm = Normalize(vmin=0.0, vmax=XP_PASS_MAX)

    for rank, row in enumerate(top_passes.itertuples(index=False), start=1):
        color = CMAP_XP(norm(float(row.xp_value)))
        lw_scale = 0.85 + 0.35 * (float(row.xp_value) / XP_PASS_MAX)
        _delicate_arrows(
            pitch, ax,
            row.x_start, row.y_start, row.x_end, row.y_end,
            color, alpha=0.92, lw_scale=lw_scale,
        )
        pitch.scatter(
            row.x_start, row.y_start,
            s=28, marker="o", color=color, edgecolors="white", linewidths=0.5, ax=ax, zorder=5,
        )
        pitch.scatter(
            row.x_end, row.y_end,
            s=36, marker="s", color=color, edgecolors="white", linewidths=0.5, ax=ax, zorder=6,
        )
        ax.text(
            row.x_end, row.y_end + 2.5,
            f"#{rank} {row.xp_value:.2f}",
            ha="center", va="bottom", color="white", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#475569", alpha=0.9),
            zorder=7,
        )

    sm = ScalarMappable(norm=norm, cmap=CMAP_XP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("xP do passe", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    legend_handles = [
        Line2D([0], [0], color=CMAP_XP(0.9), lw=2.0, label="Maior xP"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#94a3b8", markersize=5, linestyle="None", label="Origem"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#94a3b8", markersize=5, linestyle="None", label="Destino"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=7,
    )
    for text in leg.get_texts():
        text.set_color("white")

    ax.set_title(
        f"{player_name}\nTop {len(top_passes)} passes xP · {match_label}",
        color="white", fontsize=10, pad=8,
    )
    return fig


def draw_special_passes_season_map(
    passes,
    *,
    player_name: str,
    season_label: str = "temporada",
    category_label: str = "Special pass",
    xp_col: str | None = None,
    threat_col: str | None = None,
):
    """Season map of passes for one special-pass category."""
    fig, ax, pitch = _base_pitch()

    if passes is None or passes.empty:
        ax.text(
            60, 40, "Sem passes para este filtro",
            ha="center", va="center", color="white", fontsize=10,
        )
        ax.set_title(
            f"{player_name}\n{category_label} · {season_label}",
            color="white", fontsize=10, pad=8,
        )
        return fig

    work = passes.copy()
    color_by_xp = xp_col is not None and xp_col in work.columns
    if color_by_xp:
        values = work[xp_col].to_numpy(dtype=float)
        vmax = max(float(np.max(values)), 0.05)
        norm = Normalize(vmin=0.0, vmax=min(vmax, XP_PASS_MAX))
    else:
        norm = None

    for row in work.itertuples(index=False):
        if color_by_xp:
            xp_value = float(getattr(row, xp_col))
            color = CMAP_XP(norm(xp_value))
            lw_scale = 0.85 + 0.35 * min(xp_value / XP_PASS_MAX, 1.0)
        else:
            color = "#60a5fa"
            lw_scale = 1.0
        _delicate_arrows(
            pitch, ax,
            row.x_start, row.y_start, row.x_end, row.y_end,
            color, alpha=0.9, lw_scale=lw_scale,
        )
        pitch.scatter(
            row.x_start, row.y_start,
            s=28, marker="o", color=color, edgecolors="white", linewidths=0.5, ax=ax, zorder=5,
        )
        pitch.scatter(
            row.x_end, row.y_end,
            s=36, marker="s", color=color, edgecolors="white", linewidths=0.5, ax=ax, zorder=6,
        )

    if color_by_xp:
        sm = ScalarMappable(norm=norm, cmap=CMAP_XP)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("xP do passe", color="white", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    legend_handles = [
        Line2D([0], [0], color=CMAP_XP(0.9) if color_by_xp else "#60a5fa", lw=2.0, label="Passe"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#94a3b8", markersize=5, linestyle="None", label="Origem"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#94a3b8", markersize=5, linestyle="None", label="Destino"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=7,
    )
    for text in leg.get_texts():
        text.set_color("white")

    ax.set_title(
        f"{player_name}\n{len(work)} passes · {category_label} · {season_label}",
        color="white", fontsize=10, pad=8,
    )
    return fig


def draw_xp_threat_passes_season_map(
    passes,
    *,
    player_name: str,
    season_label: str = "temporada",
    distance_label: str = "todas as distâncias",
    xp_col: str = "xp_m4",
):
    """Backward-compatible alias for threat-only season maps."""
    return draw_special_passes_season_map(
        passes,
        player_name=player_name,
        season_label=season_label,
        category_label=distance_label,
        xp_col=xp_col,
    )
