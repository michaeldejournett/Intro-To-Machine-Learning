"""Load cab rides and weather, merge on pickup neighborhood and nearest timestamp."""

from __future__ import annotations

import pandas as pd

from config import CAB_CSV, WEATHER_CSV


def load_raw_cab(path: str | None = None) -> pd.DataFrame:
    p = path or str(CAB_CSV)
    df = pd.read_csv(p)
    df = df.rename(columns={"source": "location"})
    # Cab timestamps are epoch milliseconds; weather.csv uses epoch seconds.
    df["time_stamp"] = (df["time_stamp"] // 1000).astype("int64")
    return df


def load_raw_weather(path: str | None = None) -> pd.DataFrame:
    p = path or str(WEATHER_CSV)
    df = pd.read_csv(p)
    # Strip BOM if present in first column name
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def merge_cab_weather(
    cab: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Nearest-time weather reading per (location, cab time_stamp)."""
    parts: list[pd.DataFrame] = []
    for loc, cab_loc in cab.groupby("location", sort=False):
        w_loc = weather[weather["location"] == loc]
        if w_loc.empty:
            continue
        cab_loc = cab_loc.sort_values("time_stamp")
        w_loc = w_loc.sort_values("time_stamp")
        m = pd.merge_asof(
            cab_loc,
            w_loc,
            on="time_stamp",
            direction="nearest",
            suffixes=("", "_wx"),
        )
        parts.append(m)
    merged = pd.concat(parts, axis=0, ignore_index=True)
    return merged


def load_merged_dataframe(
    cab_path: str | None = None,
    weather_path: str | None = None,
) -> pd.DataFrame:
    cab = load_raw_cab(cab_path)
    w = load_raw_weather(weather_path)
    return merge_cab_weather(cab, w)
