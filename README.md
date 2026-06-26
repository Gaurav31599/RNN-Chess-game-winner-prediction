# Chess Winner Prediction — Big Data with PySpark

Predict the outcome of online chess games using Apache Spark (PySpark) with GPU acceleration via NVIDIA RAPIDS.

> **Note:** The current implementation uses a **Random Forest** classifier (two prediction targets). An **RNN-based model** is planned as a future extension for sequence-based move prediction.

---

## Dataset

[Lichess Chess Games](https://www.kaggle.com/datasnaek/chess) — `games.csv`  
~20,000 rated games with features: player ratings, opening codes, time controls, moves, and outcomes.

Place the downloaded file at `data/games.csv` (excluded from version control by `.gitignore`).

---

## Project Structure

```
.
├── notebooks/
│   └── chess_winner_prediction.ipynb   # Full analysis notebook (EDA + modelling)
├── src/
│   ├── setup_spark.py                  # Spark + RAPIDS environment setup
│   ├── data_preprocessing.py           # Load, clean, feature engineering
│   ├── eda.py                          # Correlation heatmap & distribution plots
│   ├── random_forest.py                # RF classifier — Case 1 (7-class) & Case 2 (3-class)
│   └── main.py                         # End-to-end pipeline entry point
├── data/                               # Place games.csv here (gitignored)
├── outputs/                            # Saved plots / confusion matrices (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Models

### Random Forest (PySpark MLlib)

| Case | Target | Classes | Best Accuracy |
|------|--------|---------|--------------|
| 1 | `winner-victory_status` | 7 | ~38% |
| 2 | `winner` only | 3 (white / black / draw) | ~77% (white), ~52% (black) |

- 5-fold Cross Validation
- Grid search over `maxDepth ∈ {5, 15}`, `numTrees ∈ {5, 15, 30, 60}`, `maxBins = 365`
- GPU-accelerated via [RAPIDS Accelerator for Apache Spark](https://nvidia.github.io/spark-rapids/)

### RNN (planned)
Sequence model over the `moves` column to capture game dynamics beyond aggregate statistics.

---

## Setup

### Kaggle / Colab (GPU runtime recommended)

Open `notebooks/chess_winner_prediction.ipynb` — the first cell handles all dependency installation.

### Local

```bash
# Java 8+ required
pip install -r requirements.txt
python src/main.py --data data/games.csv
```

---

## Tech Stack

| Tool | Role |
|------|------|
| Apache Spark 3.1 / PySpark 3.2 | Distributed data processing |
| Hadoop 2.7 | Underlying file system layer |
| NVIDIA RAPIDS | GPU-accelerated Spark SQL |
| PySpark MLlib | Random Forest, cross-validation, evaluation |
| Pandas / Seaborn / Matplotlib | EDA and visualisation |

---

*Originally developed during the II level Master in Big Data Analytics and Social Mining, University of Pisa.*
