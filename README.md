#  Bank Marketing Classification with PySpark ML

> **Big Data Processing · Pontificia Universidad Javeriana**  
> **Author:** Diego Alejandro Sarmiento Rodriguez  
> **Period:** April 28 – May 25, 2026

---

##  Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Methodology](#methodology)
   - [1. Spark Session Setup](#1-spark-session-setup)
   - [2. Data Loading](#2-data-loading)
   - [3. Data Understanding & Description](#3-data-understanding--description)
   - [4. Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
   - [5. Data Cleaning & Preparation](#5-data-cleaning--preparation)
   - [6. Feature Engineering](#6-feature-engineering)
   - [7. Machine Learning Models](#7-machine-learning-models)
   - [8. Model Evaluation](#8-model-evaluation)
6. [Results Summary](#results-summary)
7. [Conclusions](#conclusions)
8. [How to Run](#how-to-run)

---

## Project Overview

This project implements a complete **supervised binary classification pipeline** using **Apache Spark (PySpark)** to predict whether a bank client will subscribe to a fixed-term deposit (`y = yes/no`).

The data corresponds to direct marketing phone campaigns conducted by a **Portuguese banking institution**. The goal is to use the full data science lifecycle — from ingestion and EDA to model training and evaluation — entirely within a distributed Spark environment.

Five classification algorithms are trained and compared:

| # | Algorithm |
|---|-----------|
| 1 | Logistic Regression |
| 2 | Decision Tree |
| 3 | Random Forest |
| 4 | Gradient Boosted Trees (GBT) |
| 5 | Support Vector Machine (SVM) |

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **File** | `bank-full.csv` |
| **Records** | ~45,211 rows |
| **Features** | 16 input variables + 1 target |
| **Target** | `y` — Has the client subscribed to a term deposit? (`yes` / `no`) |
| **Storage** | Loaded from Hadoop HDFS cluster (`hdfs://10.195.34.34:9000/csv/bank-full.csv`) |

### Variable Dictionary

| Variable | Type | Description |
|---|---|---|
| `age` | Integer | Client's age in years |
| `job` | Categorical | Type of job (e.g., blue-collar, management, technician) |
| `marital` | Categorical | Marital status (married, single, divorced) |
| `education` | Categorical | Education level (primary, secondary, tertiary) |
| `default` | Binary | Has credit in default? |
| `balance` | Integer | Average yearly bank balance (euros) |
| `housing` | Binary | Has a housing loan? |
| `loan` | Binary | Has a personal loan? |
| `contact` | Categorical | Communication type (cellular, telephone, unknown) |
| `day` | Integer | Last contact day of the month |
| `month` | Categorical | Last contact month of the year |
| `duration` | Integer | Last contact duration in seconds |
| `campaign` | Integer | Number of contacts during this campaign |
| `pdays` | Integer | Days since client was last contacted (-1 = never) |
| `previous` | Integer | Number of contacts before this campaign |
| `poutcome` | Categorical | Outcome of previous marketing campaign |
| `y` | **Target** | Subscribed to a term deposit? (yes / no) |

---

## Tech Stack

```
Python 3.x
Apache Spark / PySpark (MLlib + SQL)
Hadoop HDFS
pandas
numpy
matplotlib
seaborn
scikit-learn (roc_curve utility)
findspark
```

---

## Project Structure

```
.
├── Lab_Clasification_Sarmiento.ipynb   # Main notebook
├── pipeModel/                          # Saved PySpark Pipeline model
├── output.parquet                      # Processed feature dataset
└── README.md                           # This file
```

---

## Methodology

### 1. Spark Session Setup

A `SparkSession` is initialized using `SparkConf` with the application name `"Metrics_Lab"`. The session is configured to run against the university's distributed cluster. `findspark` is used to locate and add Spark to the Python path before any PySpark imports.

```python
configura = SparkConf()
configura.setAppName("Metrics_Lab")
sparkS = SparkSession.builder.config(conf=configura).getOrCreate()
```

---

### 2. Data Loading

The `bank-full.csv` file is read directly from **Hadoop HDFS** into a Spark DataFrame. The file uses semicolons (`;`) as field delimiters and includes a header row.

```python
df00 = sparkS.read.format("csv") \
    .option("header", "true") \
    .option("sep", ";") \
    .load("hdfs://10.195.34.34:9000/csv/bank-full.csv")
```

---

### 3. Data Understanding & Description

After loading, the schema is printed and descriptive statistics are computed for every column individually using `.describe()` in a loop. This reveals the data types (all loaded as strings initially) and the value distributions across all 17 variables.

**Key finding — Class Imbalance:**

| Class | Count | Percentage |
|---|---|---|
| `no` | 39,922 | **88.3%** |
| `yes` | 5,289 | **11.7%** |

> ⚠️ This severe imbalance (88.3% vs 11.7%) would introduce significant bias into any ML model trained without correction.

**Variable casting:** Numeric columns loaded as strings are explicitly cast to `IntegerType` using `.withColumn()` and `.cast("int")`. The affected columns are: `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, and `previous`.

---

### 4. Exploratory Data Analysis (EDA)

A thorough EDA is conducted using PySpark's `.groupBy()`, converted to pandas, and visualized with `matplotlib` and `seaborn`.

#### 4.1 Frequency Histograms (Numeric Variables)

A 4×2 grid of histograms is generated for all 7 numeric variables:

- **Age:** Unimodal, slightly right-skewed. Most clients are between 25–60 years old, peaking at 30–40.
- **Balance:** Extremely right-skewed. Centered near 0–5,000€ with an extreme tail past 80,000€. Negative values (overdrafts) also present.
- **Day:** Relatively uniform with cyclic contact peaks throughout the month.
- **Duration:** Heavily right-skewed. Most calls last under 500 seconds.
- **Campaign:** Heavily right-skewed. Majority of clients were contacted 1–3 times; some extreme cases show 50+ contacts.
- **Pdays:** Bimodal — a massive spike at -1 (never contacted) and a secondary cluster between 100–400 days.
- **Previous:** Similar to pdays; the overwhelming majority shows 0 prior contacts.

#### 4.2 Boxplots (Numeric Variables vs. Target)

Side-by-side boxplots for each numeric variable split by the target class `y` reveal:

- **Duration:** The clearest separator. Clients who subscribed had noticeably longer call durations — a strong predictor.
- **Campaign:** Subscribing clients were reached in fewer attempts; non-subscribers accumulate a long tail of 60+ contacts.
- **Pdays / Previous:** Clients who subscribed show a visible history of prior contact, while non-subscribers are predominantly first-time prospects.
- **Age / Day / Balance:** Similar distributions across both classes, making them weaker individual predictors.

#### 4.3 Correlation Matrix

A Pearson correlation matrix is computed natively in PySpark using `Correlation.corr()` after assembling numeric variables into a `VectorAssembler`. The result is visualized as a heatmap.

**Strongest correlations:**

| Pair | Correlation | Interpretation |
|---|---|---|
| `pdays` ↔ `previous` | **+0.45** | More past contact days = more past contacts |
| `duration` ↔ `y` | **+0.39** | Longer calls strongly tied to subscription |
| `day` ↔ `campaign` | **+0.16** | Campaign activity increases at certain points in the month |
| `pdays` ↔ `y` | **+0.10** | Prior contact recency has a mild positive effect on conversion |
| `previous` ↔ `y` | **+0.09** | Prior contact history weakly increases conversion likelihood |

#### 4.4 Pair Plot

A full `sns.pairplot()` is generated on all numeric variables, colored by the target class `y`, to visually inspect pairwise relationships and identify any natural class separation.

#### 4.5 Pie Charts (Binary Variables)

Individual pie charts are produced for `default`, `housing`, and `loan`:

- **Default:** Extreme imbalance — over 98% of clients have no credit in default.
- **Housing:** Relatively balanced — a large share of clients carry a housing loan.
- **Loan:** Imbalanced — the majority of clients have no personal loan.

#### 4.6 Count Plots (Categorical Variables vs. Target)

A 5×2 grid of `sns.countplot()` charts breaks down 9 categorical variables by the target variable `y`:

- **Job:** Blue-collar dominates by count; management and retired segments show higher conversion rates.
- **Marital:** Married clients are the largest group; conversion rates are fairly consistent across statuses.
- **Education:** Secondary education is most represented; tertiary-educated clients show slightly higher conversion.
- **Contact:** Cellular is the dominant method; unknown contact type is very common (potential data quality issue).
- **Month:** May has the highest contact volume; low-activity months (Sep, Dec) paradoxically show higher relative conversion rates.
- **Poutcome:** Mostly unknown (new clients); where recorded, prior successes strongly predict current subscription.

#### 4.7 Age vs. Marital Boxplot

A dedicated boxplot reveals the expected demographic structure:

- **Single** → Lowest median age (~32 years)
- **Married** → Mid median age (~42 years)
- **Divorced** → Highest median age (~45 years)

#### 4.8 Null Value Check

A null value audit is run across all columns using `F.col(column).isNull()`. No null values are found in the dataset, confirming data integrity.

---

### 5. Data Cleaning & Preparation

Three transformation steps are applied sequentially to produce a clean, model-ready dataset:

#### Step 1 — Remove `pdays` column

`pdays` is dropped because 82%+ of its values are -1 (never contacted), making it a sparse, biased variable. Its information is also already partially captured by `previous`, creating multicollinearity risk.

```python
df03 = df02.drop('pdays')
```

#### Step 2 — Remove outliers in `previous`

Records where `previous > 30` are identified as extreme outliers (clients contacted more than 30 times before the current campaign). These are filtered out to prevent distortion of model boundaries.

```python
df02 = df01.filter(F.col('previous') <= 30)
```

#### Step 3 — Oversampling to Fix Class Imbalance

The minority class (`y = 'yes'`) is oversampled using PySpark's `.sample(True, ...)` with replacement until it matches the size of the majority class (`y = 'no'`). The two DataFrames are then combined with `.union()`.

```python
dfOverSampledMinor = dfMenorDependiente.sample(True, cantMayor / dfMenorDependiente.count(), seed=42)
df04 = dfMayorDependiente.union(dfOverSampledMinor)
```

**Result after oversampling:** Both classes achieve ~50% balance.

---

### 6. Feature Engineering

A full **PySpark ML Pipeline** is built to automate all preprocessing steps before model training:

#### 6.1 Categorical Encoding

For each of the 9 categorical variables (`job`, `marital`, `education`, `default`, `month`, `housing`, `loan`, `contact`, `poutcome`), two stages are added to the pipeline:

1. **`StringIndexer`** — Converts string categories to numeric indices.
2. **`OneHotEncoder`** — Converts numeric indices into sparse binary vectors (one-hot).

#### 6.2 Target Label Encoding

The target variable `y` (`yes`/`no`) is encoded into a numeric `label` column using a `StringIndexer` with `alphabetAsc` ordering (mapping `no → 0`, `yes → 1`).

#### 6.3 Feature Vector Assembly

A `VectorAssembler` combines all one-hot encoded categorical vectors and the 6 numeric features into a single `features` dense/sparse vector column:

```python
NumCol = ['age', 'balance', 'duration', 'day', 'campaign', 'previous']
ensInput = [c + '_oneHot' for c in col_cat] + NumCol
```

#### 6.4 Pipeline Execution

The pipeline is fitted on `df04` and the transformed dataset is saved:

- **Trained pipeline model** → saved to `pipeModel/`
- **Final feature dataset** (`label` + `features`) → saved as `output.parquet`

#### 6.5 Train / Test Split

The final dataset `df05` is split with an **80/20 ratio** using a fixed seed:

```python
trainData, testData = df05.randomSplit([.8, .2], seed=4321)
```

Both splits are verified to confirm balanced class distribution after oversampling.

---

### 7. Machine Learning Models

Two reusable helper functions are defined before training:

- **`plotConfusionMat()`** — Renders a heatmap confusion matrix using seaborn.
- **`plotROC()` / `plotROC_SVM()`** — Plots the ROC curve using `sklearn.metrics.roc_curve`, adapting for models with or without a `probability` column.

All five models are evaluated using:

- `MulticlassClassificationEvaluator` → Accuracy, Weighted Precision, Weighted Recall, F1-Score
- `BinaryClassificationEvaluator` → ROC-AUC

---

#### Model 1: Logistic Regression

A linear probabilistic classifier trained with a maximum of 10 iterations.

```python
LRinstance = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)
LRmodel = LRinstance.fit(trainData)
```

---

#### Model 2: Decision Tree Classifier

A single decision tree, trained without depth constraints (default settings).

```python
DTinstance = DecisionTreeClassifier(labelCol='label', featuresCol='features')
modelDT = DTinstance.fit(trainData)
```

---

#### Model 3: Random Forest Classifier

An ensemble of decision trees trained in parallel with randomized feature selection.

```python
RFinstance = RandomForestClassifier(labelCol='label', featuresCol='features')
modelRF = RFinstance.fit(trainData)
```

---

#### Model 4: Gradient Boosted Tree (GBT)

An ensemble model that trains trees sequentially, each correcting the residuals of the previous one.

```python
GBTinstance = GBTClassifier(labelCol='label', featuresCol='features')
modelGBT = GBTinstance.fit(trainData)
```

---

#### Model 5: Support Vector Machine (SVM)

A Linear SVM (`LinearSVC`) that finds the optimal hyperplane separating the two classes. Note: SVM does not output a `probability` column; ROC is computed from `rawPrediction` scores instead.

```python
SVMinstance = LinearSVC(labelCol='label', featuresCol='features')
modelSVM = SVMinstance.fit(trainData)
```

---

### 8. Model Evaluation

A final comparative bar chart is produced showing **Accuracy**, **F1-Score**, and **ROC-AUC** for all five models, sorted by ROC-AUC performance.

**Evaluation Summary:**

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Gradient Boosted Tree | ~84–85% | ~84–85% | **> 0.90** |
| Logistic Regression | ~83–84% | ~83–84% | **> 0.90** |
| SVM | ~82–84% | ~82–84% | **> 0.90** |
| Random Forest | ~82–83% | ~82–83% | ~0.88–0.89 |
| Decision Tree | ~82–83% | ~82–83% | ~0.82–0.84 |

> **Note:** Exact metric values depend on the Spark execution environment and cluster state at runtime.

---

## Results Summary

- All five models achieve **competitive Accuracy and F1-Scores** (82–85%), demonstrating that the feature engineering and oversampling pipeline was effective.
- The key differentiator between models is the **ROC-AUC score**, which measures the model's ability to distinguish between the two classes across all decision thresholds.
- **GBT, Logistic Regression, and SVM** all exceed a ROC-AUC of 0.90, making them the top candidates for production deployment.
- **Decision Tree** performs the weakest in ROC-AUC despite similar accuracy, suggesting it overfits to specific thresholds.

---

## Conclusions

**1. Call duration is the single strongest predictor of term deposit subscription.**
The correlation matrix and boxplot analysis consistently highlight `duration` as the variable most linearly associated with the target (`r = 0.39`). Clients who eventually subscribed showed significantly longer call durations than those who declined. This suggests that the quality and depth of engagement during a phone call is a far more reliable signal than demographic or financial variables. For future campaigns, this finding supports prioritizing conversation quality over contact volume.

**2. Gradient Boosted Trees (GBT) provides the best overall classification performance for this banking use case.**
While all models converge at similar Accuracy and F1-Score levels (~82–85%), GBT consistently leads in ROC-AUC (exceeding 0.90). This means GBT minimizes false positives most effectively — a critical property in a marketing context where incorrectly targeting non-converting clients wastes campaign resources. Combined with the oversampling strategy applied during data preparation, GBT proves to be the most robust and reliable model for predicting term deposit subscription from historical campaign data.

---

## How to Run

### Prerequisites

- Apache Spark cluster with HDFS access
- Python 3.x with the following packages:
  ```
  pyspark, findspark, pandas, numpy, matplotlib, seaborn, scikit-learn
  ```
- `bank-full.csv` uploaded to HDFS at `hdfs://<your-cluster-ip>:9000/csv/`

### Steps

1. Clone this repository and open the notebook in JupyterLab or Jupyter Notebook.
2. Update the HDFS path in the data loading cell to match your cluster IP.
3. Run all cells in order from top to bottom.
4. Model outputs (pipeline, parquet) will be saved to the working directory.

---

> **Course:** Big Data Processing — Pontificia Universidad Javeriana  
> **Dataset License:** UCI Machine Learning Repository (open access for academic use)
