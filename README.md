# MLOps Predictive Maintenance System: Pipeline & Model Governance

A comprehensive, end-to-end Machine Learning Operations (MLOps) system demonstrating automated experiment tracking, model governance, and production lifecycle management using **MLflow**. The project builds a robust predictive maintenance engine designed to identify potential machinery failure before it occurs using synthetic multi-sensor equipment readings.

---

## 🏗️ System Architecture

The project is split into two foundational operational phases:

### Phase 1: Experimentation & Tracking (Lab 13)
* **Data Synthesization:** Simulates 10,000 equipment matrices (Temperature, Vibration, Pressure, RPM, Operational Age) with built-in realistic anomalies.
* **Statistical Profiling:** Conducts deep Exploratory Data Analysis (EDA) to establish correlation matrices and failure distributions.
* **Parallel Architecture Benchmarking:** Automatically logs parameters, metrics, and serialized model artifacts for `LogisticRegression`, `RandomForestClassifier`, and `XGBClassifier`.

### Phase 2: Governance & Versioning (Lab 14)
* **Model Registry Client:** Queries the programmatic tracking database using the `MlflowClient` to isolate the champion run based on the highest $ROC\ AUC$ metric.
* **Metadata Lifecycle Governance:** Enriches the registered champion with metadata tags (`validation_status`, `framework`, `team`) and transfers it through lifecycle phases (`Staging` ➔ `Production`).
* **Production Inference Pipeline:** Exposes a dynamic operational interface that continuously binds to the active production model URI.
* **Disaster Recovery Rollback:** Simulates minor-version drift scenarios and verifies instant rollback capabilities to a stable baseline version.

---

## 📊 Evaluation Baseline & Expectations

Based on system parameters, the algorithms perform within the following historical boundaries:

| Model | Logged Parameters | Expected ROC AUC Range |
| :--- | :--- | :--- |
| **XGBoost** | `n_estimators: 100`, `max_depth: 6`, `learning_rate: 0.1` | **0.92 - 0.95** (Champion) |
| **Random Forest** | `n_estimators: 100`, `max_depth: 10`, `min_samples_split: 5` | **0.90 - 0.93** |
| **Logistic Regression** | `C: 1.0`, `max_iter: 1000` | **0.85 - 0.88** |

---

## 💻 Technical Setup

### 1. Environment Initialization
Clone the repository and install the required engineering dependencies:
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
cd YOUR_REPOSITORY
pip install mlflow scikit-learn pandas numpy matplotlib seaborn xgboost
