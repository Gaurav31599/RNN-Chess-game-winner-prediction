"""
Spark environment setup for Kaggle / Colab.
Run this before importing PySpark.
"""
import os
import subprocess

def install_dependencies():
    subprocess.run(["apt-get", "install", "openjdk-8-jdk-headless", "-qq"], check=True)
    subprocess.run(["wget", "-q",
                    "https://downloads.apache.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz"],
                   check=True)
    subprocess.run(["tar", "xf", "spark-3.1.2-bin-hadoop2.7.tgz"], check=True)
    subprocess.run(["pip", "install", "pyspark==3.2.1", "py4j==0.10.9.3", "findspark"], check=True)

    subprocess.run(["wget",
                    "https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/21.12.0/rapids-4-spark_2.12-21.12.0.jar"],
                   check=True)
    subprocess.run(["wget",
                    "https://repo1.maven.org/maven2/ai/rapids/cudf/21.12.2/cudf-21.12.2-cuda11.jar"],
                   check=True)


def configure_env(spark_home: str = "/kaggle/working/spark-3.1.2-bin-hadoop2.7"):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["SPARK_HOME"] = spark_home
    rapids_jar = f"{os.path.dirname(spark_home)}/rapids-4-spark_2.12-21.12.0.jar"
    cudf_jar = f"{os.path.dirname(spark_home)}/cudf-21.12.2-cuda11.jar"
    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        f"--jars {rapids_jar},{cudf_jar} --master local[*] pyspark-shell"
    )

    import findspark
    findspark.init()


def create_spark_session():
    from pyspark.sql import SparkSession
    rapids_jar = "/kaggle/working/rapids-4-spark_2.12-21.12.0.jar"
    cudf_jar = "/kaggle/working/cudf-21.12.2-cuda11.jar"

    spark = (
        SparkSession.builder.appName("ChessWinnerPrediction")
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .getOrCreate()
    )
    spark.sparkContext.addPyFile(rapids_jar)
    spark.sparkContext.addPyFile(cudf_jar)
    spark.conf.set("spark.rapids.sql.enabled", "true")
    spark.conf.set("spark.rapids.sql.incompatibleOps.enabled", "true")
    spark.conf.set("spark.rapids.sql.format.csv.read.enabled", "true")
    spark.conf.set("spark.rapids.sql.format.csv.enabled", "true")
    return spark
