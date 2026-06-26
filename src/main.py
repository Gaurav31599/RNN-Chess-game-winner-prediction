"""
Entry point — runs the full pipeline end-to-end.

Usage:
    python src/main.py --data data/games.csv
"""
import argparse
from setup_spark import configure_env, create_spark_session
from data_preprocessing import load_data, engineer_features
from eda import (
    pearson_correlation, plot_correlation_heatmap,
    plot_winner_distribution, plot_target_distribution,
    plot_opening_ply_vs_rating, plot_time_boxplots,
)
from random_forest import run_case1, run_case2


def main(data_path: str):
    configure_env()
    spark = create_spark_session()

    # Load & preprocess
    raw = load_data(spark, data_path)
    df = engineer_features(raw)

    # EDA
    df_pd = df.toPandas()
    corr_df = pearson_correlation(df)
    plot_correlation_heatmap(corr_df, save_path="outputs/correlation_heatmap.png")
    plot_winner_distribution(df_pd)
    plot_target_distribution(df_pd)
    plot_opening_ply_vs_rating(df_pd)
    plot_time_boxplots(df_pd)

    # Models
    run_case1(df)
    run_case2(df)

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess winner prediction with PySpark RF")
    parser.add_argument("--data", default="data/games.csv", help="Path to games.csv")
    args = parser.parse_args()
    main(args.data)
