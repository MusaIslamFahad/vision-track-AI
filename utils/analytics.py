"""
analytics.py — Per-session detection statistics and Plotly chart builders.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class SessionStats:
    """Accumulates per-frame detection results for post-session analytics."""

    def __init__(self):
        self.frame_data: list[dict] = []
        self.unique_ids: set[int] = set()

    def update(self, frame_idx: int, detections: list, fps: float) -> None:
        class_counts: dict[str, int] = defaultdict(int)
        for det in detections:
            class_counts[det.class_name] += 1
            self.unique_ids.add(det.track_id)

        self.frame_data.append({
            "frame": frame_idx,
            "fps": fps,
            "total_objects": len(detections),
            **{f"cls_{k}": v for k, v in class_counts.items()},
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.frame_data).fillna(0)

    def summary(self) -> dict:
        if not self.frame_data:
            return {}
        df = self.to_dataframe()
        return {
            "total_frames_processed": len(df),
            "unique_tracks": len(self.unique_ids),
            "avg_objects_per_frame": round(df["total_objects"].mean(), 2),
            "peak_objects": int(df["total_objects"].max()),
            "avg_fps": round(df["fps"].mean(), 2),
        }


# ── Chart builders ────────────────────────────────────────────────────────────

def objects_over_time_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.area(
        df, x="frame", y="total_objects",
        title="Objects Detected Per Frame",
        labels={"frame": "Frame", "total_objects": "Object Count"},
        color_discrete_sequence=["#185FA5"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAF8",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def fps_over_time_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        df, x="frame", y="fps",
        title="Inference FPS Over Time",
        labels={"frame": "Frame", "fps": "FPS"},
        color_discrete_sequence=["#27AE60"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAF8",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def class_distribution_chart(class_counts: dict[str, int]) -> Optional[go.Figure]:
    if not class_counts:
        return None
    df = pd.DataFrame(
        sorted(class_counts.items(), key=lambda x: x[1], reverse=True),
        columns=["Class", "Count"]
    )
    fig = px.bar(
        df, x="Count", y="Class", orientation="h",
        title="Detections by Class",
        color="Count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAF8",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def track_id_timeline(df: pd.DataFrame) -> Optional[go.Figure]:
    cls_cols = [c for c in df.columns if c.startswith("cls_")]
    if not cls_cols:
        return None
    melted = df[["frame"] + cls_cols].melt(
        id_vars="frame", var_name="class", value_name="count"
    )
    melted["class"] = melted["class"].str.replace("cls_", "", regex=False)
    melted = melted[melted["count"] > 0]
    fig = px.area(
        melted, x="frame", y="count", color="class",
        title="Class Breakdown Over Time",
        labels={"frame": "Frame", "count": "Count", "class": "Class"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAF8",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig
