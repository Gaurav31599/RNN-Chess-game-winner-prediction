"""
Exploratory Data Analysis — correlation heatmap and distribution plots.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.linalg import DenseMatrix
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler


def pearson_correlation(df):
    """Compute and return a pandas DataFrame with Pearson correlation coefficients."""
    continuous_cols = [
        "turns", "white_rating", "black_rating",
        "opening_ply", "fix_time", "variable_time",
    ]
    df_corr = df.select(continuous_cols)
    assembler = VectorAssembler(inputCols=continuous_cols, outputCol="features")
    out = assembler.transform(df_corr)
    pearson = Correlation.corr(out, "features", "pearson").collect()[0][0]
    arr = pearson.toArray()
    return pd.DataFrame(arr, index=continuous_cols, columns=continuous_cols)


def plot_correlation_heatmap(corr_df, save_path: str = None):
    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    sns.heatmap(corr_df, annot=True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_winner_distribution(df_pandas):
    sns.histplot(data=df_pandas, x="winner", color="black")
    plt.xlabel("Winner", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.show()


def plot_target_distribution(df_pandas):
    ax = sns.histplot(data=df_pandas, x="Target", color="black")
    ax.tick_params(axis="x", rotation=45)
    plt.show()


def plot_opening_ply_vs_rating(df_pandas):
    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.violinplot(data=df_pandas, x="opening_ply", y="white_rating", ax=axes[0])
    axes[0].set_title("White Rating vs Opening Ply")
    sns.violinplot(data=df_pandas, x="opening_ply", y="black_rating", ax=axes[1])
    axes[1].set_title("Black Rating vs Opening Ply")
    plt.tight_layout()
    plt.show()


def plot_time_boxplots(df_pandas):
    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(
        data=df_pandas, x="Target", y="fix_time", color="black", ax=axes[0],
        medianprops=dict(color="red"), showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": "10"},
    )
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylim(-10, 170)
    axes[0].set_title("Fix Time by Outcome")

    sns.boxplot(
        data=df_pandas, x="Target", y="variable_time", color="black", ax=axes[1],
        medianprops=dict(color="white"), showmeans=True, showfliers=False,
        meanprops={"marker": "o", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": "10"},
    )
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylim(-5, 30)
    axes[1].set_title("Variable Time by Outcome")

    plt.tight_layout()
    plt.show()
