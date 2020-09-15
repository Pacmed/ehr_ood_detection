"""
Module defining functions to plot the results of experiments.
"""

# STD
from typing import Optional, Dict, Tuple

# EXT
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# PROJECT
from uncertainty_estimation.utils.types import ResultDict


def plot_results_as_heatmap(
    result_dict: ResultDict,
    name: str,
    lower_cmap_limit: float = 0,
    save_dir: Optional[str] = None,
    rel_sizes: Optional[Dict[str, float]] = None,
    percentage_sigs: Optional[Dict[str, float]] = None,
):
    """
    Plot results compactly inside a heatmap.

    Parameters
    ----------
    result_dict: ResultDict
        Results of experiments
    name: str
        Name of plot.
    lower_cmap_limit: float
        Metric score that corresponds to the lower limit color on the colorbar (e.g. for AUC-ROC, 0.5 is considered to
        be as good for random guessing - thus color everything below that the same way).
    save_dir: Optional[str]
        Path that figure should be saved to if given.
    rel_sizes: Optional[Dict[str, float]]
        Dictionary containing the relative sizes of groups models were tested on.
    percentage_sigs: Optional[Dict[str, float]]
        Dictionary containing the percentage of statistically significant different features of tested group compared to
        reference group.
    """
    df = pd.DataFrame.from_dict(result_dict)

    if rel_sizes is not None or percentage_sigs is not None:
        new_indices = []

        for row_idx in df.index:
            suffix = (
                f"\n(size: {rel_sizes[row_idx] * 100:.2f} %"
                if rel_sizes is not None
                else "("
            )
            suffix += (
                ", " if rel_sizes is not None and percentage_sigs is not None else ""
            )
            suffix += (
                f"diff: {percentage_sigs[row_idx] * 100:.2f} %)"
                if percentage_sigs is not None
                else ")"
            )
            new_indices.append(f"{row_idx} {suffix}")

        df.index = new_indices

    df = df.sort_index()
    df = df.applymap(lambda l: np.array(l).mean())
    df = df.reindex(sorted(df.columns), axis=1)
    annotations = df.applymap(lambda l: f"{np.array(l).mean():.2f}".lstrip("0"))

    _, ax = plt.subplots(figsize=(12, 12))
    sns.set(font_scale=0.8)
    sns.heatmap(
        data=df,
        annot=annotations,
        fmt="",
        linewidths=0.5,
        vmin=lower_cmap_limit,
        vmax=1,
        square=True,
        ax=ax,
        cmap="OrRd",
        cbar=False,
    )
    plt.title(name)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    if save_dir:
        plt.savefig(save_dir, dpi=300, bbox_inches="tight", pad=0)
        plt.close()
    else:
        plt.show()


def plot_results_as_boxplot(
    result_dict: ResultDict,
    name: str,
    hline: Optional[float] = None,
    vline: Optional[float] = None,
    xlim: Optional[Tuple[float, float]] = (0.0, 1.0),
    ylim: Optional[Tuple[float, float]] = (0.0, 1.0),
    dummy_group_name: Optional[str] = None,
    x_name: str = "scale",
    horizontal: bool = False,
    legend_out: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    save_dir: Optional[str] = None,
    kind: str = "box",
    rel_sizes: Optional[Dict[str, float]] = None,
    percentage_sigs: Optional[Dict[str, float]] = None,
    **kwargs,
):
    """
    Plot results as a bar plot.

    Parameters
    ----------
    result_dict: ResultDict
        Results of experiments
    name: str
        Name of plot.
    hline: Optional[float]
        Plot a horizontal line at the given y position.
    vline: Optional[float]
        Plot a vertical line at the given x position.
    xlim: Tuple[float, float]
        Set the limits on the x-axis of the current plot.
    ylim: TUple[float, float]
        Set the limits on the y-axis of the current plot.
    dummy_group_name: Optional[str]
        Name of a dummy group used for padding.
    x_name: str
        Labels for the x-axis.
    horizontal: bool
        Whether to arrange the bars horizontally or vertically.
    legend_out: bool
        Whether the legend should be placed outside the figure.
    figsize: Optional[Tuple[int, int]]
        Size of the figure.
    save_dir: Optional[str]
        Path that figure should be saved to if given.
    rel_sizes: Optional[Dict[str, float]]
        Dictionary containing the relative sizes of groups models were tested on.
    percentage_sigs: Optional[Dict[str, float]]
        Dictionary containing the percentage of statistically significant different features of tested group compared to
        reference group.
    kind: str
        The kind of plot to draw.
    """
    sns.set_palette("Set1", 10)
    sns.set_style("whitegrid")
    df = pd.DataFrame.from_dict(result_dict, orient="columns")

    if rel_sizes is not None or percentage_sigs is not None:
        new_indices = []

        for row_idx in df.index:
            suffix = (
                f"\n(size: {rel_sizes[row_idx] * 100:.2f} %"
                if rel_sizes is not None
                else "("
            )
            suffix += (
                ", " if rel_sizes is not None and percentage_sigs is not None else ""
            )
            suffix += (
                f"diff: {percentage_sigs[row_idx] * 100:.2f} %)"
                if percentage_sigs is not None
                else ")"
            )
            new_indices.append(f"{row_idx} {suffix}")

        df.index = new_indices

    df = df.stack().reset_index()

    df.columns = [x_name, "", name]
    df = df.explode(name)
    df = df.sort_index()
    df = df.sort_values(by="")
    df = df.sort_values(by=" ")
    df = df.reindex(sorted(df.columns), axis=1)

    if dummy_group_name:
        methods = [m for m in df[""].unique()]
        fake_df = pd.DataFrame(
            {
                x_name: [dummy_group_name] * len(methods),
                "": methods,
                name: [0] * len(methods),
            }
        )
        df = pd.concat([fake_df, df])

    plt.figure(figsize=figsize)

    if horizontal:
        sns.catplot(
            x=name,
            y=x_name,
            hue="",
            data=df,
            kind=kind,
            legend_out=legend_out,
            facet_kws=dict(despine=False),
            **kwargs,
        )

    else:
        sns.catplot(
            x=x_name,
            y=name,
            hue="",
            data=df,
            kind=kind,
            legend_out=legend_out,
            facet_kws=dict(despine=False),
            **kwargs,
        )

    plt.ylim(ylim)
    plt.xlim(xlim)

    if hline:
        plt.axhline(hline, linestyle="--")
    if vline:
        plt.axvline(vline, linestyle="--")
    if save_dir:
        plt.savefig(save_dir, dpi=300, bbox_inches="tight", pad=0)
        plt.close()
    else:
        plt.show()
