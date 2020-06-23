from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def barplot_from_nested_dict(nested_dict: dict, xlim: Tuple[float, float],
                             figsize: Tuple[float, float], title: str, save_dir: str,
                             nested_std_dict: dict = None,
                             remove_yticks: bool = False, legend: bool = True):
    """Plot and save a grouped barplot from a nested dictionary.

    Parameters
    ----------
    nested_dict: dict
        The data represented in a nested dictionary.
    nested_std_dict: dict
        The standard deviations, also in a nested dictionary, to be used as error bars.
    xlim: Tuple[float, float]
        The limits on the x-axis to use.
    figsize: Tuple[float, float]
        The figure size to use.
    title: str
        The title of the plot.
    save_dir: str
        Where to save the file.
    remove_yticks: bool
        Whether to remove the yticks.
    """
    sns.set_palette("Set1", 10)
    sns.set_style('whitegrid')
    df = pd.DataFrame.from_dict(nested_dict,
                                orient='index').iloc[::-1]
    if nested_std_dict:
        std_df = pd.DataFrame.from_dict(nested_std_dict,
                                        orient='index')  # .iloc[::-1]
        df.plot(kind='barh', alpha=0.9, xerr=std_df, figsize=figsize, fontsize=12,
                title=title, xlim=xlim, legend=False)
    else:
        df.plot(kind='barh', alpha=0.9, figsize=figsize, fontsize=12,
                title=title, xlim=xlim, legend=False)
    if legend:
        plt.legend(loc='lower right')
    if remove_yticks:
        plt.yticks([], [])
    plt.savefig(save_dir, dpi=300,
                bbox_inches='tight', pad=0)
    plt.close()
