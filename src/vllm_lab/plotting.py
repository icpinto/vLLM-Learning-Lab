"""Plotting helpers for notebooks.

    This module uses matplotlib directly and intentionally avoids custom colour
    choices so charts work in different themes.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import pandas as pd


def bar_from_records(records: Iterable[Mapping], x: str, y: str, title: str, ylabel: str | None = None):
    df = pd.DataFrame(records)
    ax = df.plot(kind="bar", x=x, y=y, legend=False)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    plt.tight_layout()
    return ax


def line_from_dataframe(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str | None = None):
    ax = df.plot(kind="line", x=x, y=y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    plt.tight_layout()
    return ax
