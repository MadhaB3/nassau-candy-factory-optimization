# Factory Reallocation & Shipping Optimization — Nassau Candy Distributor

A decision-intelligence system that predicts shipping lead time under different factory assignments, simulates factory-reassignment scenarios before execution, and recommends configurations that improve delivery speed **without** sacrificing profitability — packaged as an interactive Streamlit dashboard.

📄 [Research Paper (PDF)](docs/Nassau_Candy_Research_Paper.pdf) · 📄 [Executive Summary (PDF)](docs/Nassau_Candy_Executive_Summary.pdf)

## Problem

Nassau Candy assigns products to its five factories using static, legacy rules. This creates suboptimal shipping distances, elevated lead times in certain regions, and margin erosion from logistics inefficiency — with no system to simulate reassignment scenarios or quantify impact before execution.

## What this project does

- **Predictive modeling** — Linear Regression, Random Forest, and Gradient Boosting regressors predict shipping lead time from product, factory, region, and ship mode. Best model (Gradient Boosting): **R² = 0.977, RMSE = 0.50 days**.
- **Route & product clustering** — K-Means clustering surfaces consistently slow routes vs. high-performing ones.
- **Scenario simulation engine** — every product is simulated against all five factories, predicting lead time and profit impact of each candidate reassignment.
- **Recommendation engine** — 125 reassignment opportunities ranked by a composite lead-time/profit score, averaging a **41% lead-time reduction** and **~$3,300 combined estimated profit impact**.
- **Streamlit dashboard** — Factory Optimization Simulator, What-If Scenario Analysis, Recommendation Dashboard, and Risk & Impact Panel (with a speed-vs-profit priority slider).

## ⚠️ Key data-quality finding

The raw `Ship Date` column in the source file is corrupted: its year drifts 2–4+ years ahead of `Order Date`, growing across the file, making the literal `Ship Date − Order Date` unusable (mean ≈ 1,320 days). This is documented in full in the research paper. **Modeling fix:** an `Estimated Lead Time` was engineered from great-circle distance (factory → customer state/province centroid) and ship-mode speed, used consistently across modeling, clustering, and simulation, and flagged transparently in the dashboard as an estimate rather than an observed fact.

## Tech stack

Python (pandas, NumPy, scikit-learn), Streamlit, Plotly, Matplotlib

## Project structure

```
├── app.py                  # Streamlit dashboard
├── data_prep.py             # Cleaning, distance engineering, lead-time reconstruction
├── modeling.py               # Model training + route/product clustering
├── scenario_engine.py        # Scenario simulation + recommendation ranking
├── eda_charts.py              # Chart generation for the research paper
├── data/                       # Raw + processed datasets, simulation outputs
├── models/                      # Trained model + evaluation metrics
├── charts/                       # Generated EDA/results charts
├── docs/                          # Research paper & executive summary (PDF)
└── requirements.txt
```

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Data and the trained model are pre-generated, so the dashboard launches instantly. To regenerate the pipeline from scratch:

```bash
python data_prep.py
python modeling.py
python scenario_engine.py
python eda_charts.py
```

## Key results

| Metric | Value |
|---|---|
| Product/region/ship-mode combinations reviewed | 154 |
| Recommendations with positive projected gain | 125 |
| Avg. lead-time reduction (recommended set) | 41.1% |
| Combined estimated profit impact | $3,333 |
| Best model R² | 0.977 |

---
Built by Madhav Bhatnagar as a portfolio data analysis project.
