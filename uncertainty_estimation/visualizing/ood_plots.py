"""
Module defining functions to plot the results of experiments.
"""

# STD
from typing import Optional

# EXT
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# PROJECT
from uncertainty_estimation.utils.types import ResultDict


def heatmap_result_plot(
    result_dict: ResultDict,
    name: str,
    lower_cmap_limit: float = 0,
    save_dir: Optional[str] = None,
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
        Metric score that corresponds to the lower limit color on the colorbar (e.g. for ROC-AUC, 0.5 is considered to
        be as good for random guessing - thus color everything below that the same way).
    save_dir: Optional[str]
        Path that figure should be saved to if given.
    """
    df = pd.DataFrame.from_dict(result_dict)
    df = df.sort_index()
    df = df.applymap(lambda l: np.array(l).mean())
    annotations = df.applymap(lambda l: f"{np.array(l).mean():.2f}".lstrip("0"))

    _, ax = plt.subplots(figsize=(12, 12))
    sns.set(font_scale=0.8)
    sns.heatmap(
        data=df,
        annot=annotations,
        fmt="",
        linewidths=0.5,
        vmin=lower_cmap_limit,
        square=True,
        ax=ax,
        cmap="viridis",
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


def barplot_from_nested_dict(
    nested_dict,
    metric_name="OOD detection AUC",
    group_name="OOD group",
    vline=None,
    xlim=(0, 1.0),
    save_dir=None,
    height=6,
    aspect=1.5,
    legend_out=False,
):
    sns.set_style("whitegrid")
    sns.set_palette("Set1", 10)
    df = pd.DataFrame.from_dict(nested_dict)

    df = df.stack().reset_index()
    df.columns = [group_name, "", metric_name]

    sns.catplot(
        x=metric_name,
        y=group_name,
        hue="",
        data=df,
        kind="bar",
        height=height,
        aspect=aspect,
        facet_kws=dict(despine=False),
        alpha=0.9,
        legend_out=legend_out,
    )
    plt.xlim(xlim)
    if vline:
        plt.axvline(vline, linestyle="--")
    if save_dir:
        plt.savefig(save_dir, dpi=300, bbox_inches="tight", pad=0)
        plt.close()
    else:
        plt.show()


def boxplot_from_nested_listdict(
    nested_dict,
    name,
    hline=None,
    vline=None,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    dummy_group_name=None,
    x_name="scale",
    horizontal=False,
    legend_out=False,
    figsize=None,
    save_dir=None,
    kind="box",
    **kwargs,
):
    sns.set_palette("Set1", 10)
    sns.set_style("whitegrid")
    df = pd.DataFrame.from_dict(nested_dict, orient="columns")

    df = df.stack().reset_index()
    df.columns = [x_name, "", name]
    df = df.explode(name)
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
