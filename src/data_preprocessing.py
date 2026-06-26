"""
Data loading and preprocessing for the Lichess chess games dataset.
Input:  games.csv  (from https://www.kaggle.com/datasnaek/chess)
Output: cleaned Spark DataFrame `df` ready for modelling.
"""
from pyspark.sql.functions import concat_ws, split, isnull, when, count, col


def load_data(spark, path: str = "../data/games.csv"):
    return (
        spark.read.format("csv")
        .load(path, header="true", inferSchema="true", multiline=True)
    )


def check_missing(df):
    return df.select(
        [count(when(isnull(c), c)).alias(c) for c in df.columns]
    )


def engineer_features(df):
    # Combined target: winner + victory_status
    df = df.withColumn("Target", concat_ws("-", "winner", "victory_status"))

    # Split increment_code into fixed and variable time
    df = (
        df.withColumn("fix_time", split(df.increment_code, "\\+")[0])
          .withColumn("variable_time", split(df.increment_code, "\\+")[1])
    )

    # Drop unneeded / inconsistent columns
    to_drop = ("created_at", "last_move_at", "victory_status", "increment_code", "moves")
    df = df.drop(*to_drop)

    # Cast new time columns
    df = (
        df.withColumn("fix_time", df.fix_time.cast("int"))
          .withColumn("variable_time", df.variable_time.cast("int"))
    )

    # Remove draw-outoftime outliers
    df = df.where(df.Target != "draw-outoftime")
    return df
