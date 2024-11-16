import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from src.visualization.utils import subplot_dim_optm


def plot_violin(X, y_tags: List[str], x_tag: str):
    # optimal subplot window
    m, n = subplot_dim_optm(len(y_tags))

    # create the violin plot
    fig, axes = plt.subplots(n, m, figsize=(10, 6), sharex=True)

    # flat 
    axes = axes.reshape(n*m) if n*m > 1 else axes

    # loop over axes to display subplots
    for ax, y_tag in zip(axes, y_tags):
        # violin plot
        sns.violinplot(
            x=x_tag, 
            y=y_tag, 
            data=X, 
            inner='box', 
            ax=ax,
            width=1.1
        )
        ax.grid()

    plt.tight_layout()

    return fig

def plot_boxplot(X, y_tags: List[str], x_tag: str):
    # optimal subplot window
    m, n = subplot_dim_optm(len(y_tags))

    # create the violin plot
    fig, axes = plt.suplots(n, m, figsize=(10, 6), sharex=True)

    # flat 
    axes = axes.reshape(n*m) if n*m == 1 else axes

    # loop over axes to display subplots
    for ax, y_tag in zip(axes, y_tags):
        # violin plot
        sns.boxplot(
            x=x_tag, 
            y=y_tag, 
            data=X, 
            ax=ax,
            width=0.4
        )

    plt.tight_layout()

    return fig
