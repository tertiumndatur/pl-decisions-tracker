#!/usr/bin/env python3

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
import pandas as pd


def build_delta_frame(conn: sqlite3.Connection, year: int) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz="UTC").timestamp() * 1000)

    sql = """
    SELECT
      d.id               AS decision_id,
      d.count            AS count,
      d.country          AS country,
      u.timestamp        AS ts,
      u.dataUpdated      AS dataUpdated,
      it.name            AS institution,
      ct.type            AS case_type,
      dm.description     AS decision_marker
    FROM decisions d
    JOIN updates u           ON u.dateId  = d.dateId
    JOIN institution it     ON it.id = d.institution
    JOIN caseType ct       ON ct.id = d.caseType
    JOIN decisionMarker dm ON dm.id = d.decisionMarker
    WHERE
      u.timestamp >= :start_ms
      AND u.timestamp <  :end_ms
      AND d.country=241 AND case_type='Ochrona międzynarodowa' AND (it.id=1516 OR it.id=810)
    ORDER BY
      u.timestamp ASC,
      ct.type ASC,
      dm.description ASC
    """

    df = pd.read_sql_query(sql, conn, params={"start_ms": start_ms, "end_ms": end_ms})
    if df.empty:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["date"] = df["ts"].dt.floor("D")

    daily = (
        df.groupby(["date", "decision_marker"])["count"]
        .sum()
        .reset_index()
        .sort_values(["decision_marker", "date"])
    )
    daily["cumulative"] = daily.groupby("decision_marker")["count"].cumsum()
    daily["delta"] = daily.groupby("decision_marker")["cumulative"].diff().fillna(daily["cumulative"])

    pivot = daily.pivot(index="date", columns="decision_marker", values="delta").fillna(0)
    delta = pivot.diff().fillna(pivot)

    delta.index = pd.to_datetime(delta.index)
    if delta.index.tz is not None:
        delta.index = delta.index.tz_convert(None)

    delta = delta.asfreq("D", fill_value=0)

    delta = delta.clip(lower=0)

    return delta


def render_plot(delta: pd.DataFrame, *, title: str | None = None):
    n = len(delta.columns)
    overall_max = float(delta.max().max()) if not delta.empty else 0
    overall_top = max(1, int(math.ceil(overall_max))) + 1
    base_per_subplot = 3
    extra_height = overall_top / 25
    fig_height = (1 + n) * (base_per_subplot + extra_height)

    fig, axes = plt.subplots(
        nrows=1 + n,
        ncols=1,
        figsize=(14, fig_height),
        sharex=True
    )

    if not isinstance(axes, (list, tuple)):
        axes = list(axes)
    else:
        axes = list(axes)

    overall_max = 0
    if not delta.empty:
        overall_max = float(delta.max().max())

    overall_top = max(1, int(math.ceil(overall_max))) + 1

    ax0 = axes[0]
    delta.plot(ax=ax0, marker="o", linewidth=1)
    ax0.set_ylim(bottom=0, top=overall_top)
    ax0.yaxis.set_major_locator(MultipleLocator(1))
    ax0.grid(True, axis="y", alpha=0.2)

    if title:
        ax0.set_title(title)

    color_map = {col: line.get_color() for col, line in zip(delta.columns, ax0.get_lines())}

    for i, col in enumerate(delta.columns):
        ax = axes[i + 1]
        ax.plot(
            delta.index,
            delta[col],
            marker="o",
            linewidth=1.2,
            markersize=4,
            color=color_map.get(col),
        )
        ax.set_title(str(col))
        ax.set_ylim(bottom=0, top=overall_top)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, axis="y", alpha=0.2)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", labelbottom=True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    return fig, axes


def main() -> int:
    db_path = Path("tracker.db").resolve()
    out_path = Path("assets/decision_marker.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        delta = build_delta_frame(conn, 2026)

    fig, _ = render_plot(delta, title=f"Decision marker daily ({2026})")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

