import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans

df = pd.read_csv("data/nassau_clean.csv")

# ---------------------------------------------------------------
# Predictive modeling: Estimated Lead Time ~ Product, Factory, Region, Ship Mode
# ---------------------------------------------------------------
features_cat = ["Factory", "Region", "Ship Mode", "Division"]
features_num = ["DistanceMiles", "Units"]
target = "EstimatedLeadTimeDays"

X = df[features_cat + features_num]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
    ("num", StandardScaler(), features_num),
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
}

results = {}
best_name, best_r2, best_pipe = None, -np.inf, None
for name, model in models.items():
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    results[name] = {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 4)}
    if r2 > best_r2:
        best_name, best_r2, best_pipe = name, r2, pipe

print(pd.DataFrame(results).T)
print("Best model:", best_name)

joblib.dump(best_pipe, "models/lead_time_model.joblib")
with open("models/model_results.json", "w") as f:
    json.dump({"results": results, "best_model": best_name}, f, indent=2)

# ---------------------------------------------------------------
# Route & product clustering
# route = Factory -> Region combination; cluster by performance similarity
# ---------------------------------------------------------------
route_perf = df.groupby(["Factory", "Region"]).agg(
    AvgLeadTime=("EstimatedLeadTimeDays", "mean"),
    AvgDistance=("DistanceMiles", "mean"),
    AvgMargin=("Gross Profit", lambda x: x.sum() / df.loc[x.index, "Sales"].sum()),
    OrderVolume=("Order ID", "count"),
    TotalSales=("Sales", "sum"),
).reset_index()

clust_features = route_perf[["AvgLeadTime", "AvgDistance", "AvgMargin", "OrderVolume"]]
scaler = StandardScaler()
clust_X = scaler.fit_transform(clust_features)

k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
route_perf["Cluster"] = km.fit_predict(clust_X)

# Label clusters by lead time / margin profile
cluster_summary = route_perf.groupby("Cluster")[["AvgLeadTime", "AvgDistance", "AvgMargin", "OrderVolume"]].mean()
cluster_summary["Rank_LeadTime"] = cluster_summary["AvgLeadTime"].rank()
labels = {}
for c, row in cluster_summary.iterrows():
    if row["AvgLeadTime"] == cluster_summary["AvgLeadTime"].max():
        labels[c] = "Consistently Slow Route"
    elif row["AvgLeadTime"] == cluster_summary["AvgLeadTime"].min():
        labels[c] = "High-Performing Route"
    elif row["AvgMargin"] == cluster_summary["AvgMargin"].min():
        labels[c] = "Margin-Risk Route"
    else:
        labels[c] = "Moderate-Performance Route"
route_perf["ClusterLabel"] = route_perf["Cluster"].map(labels)

route_perf.to_csv("data/route_clusters.csv", index=False)
print(route_perf.sort_values("AvgLeadTime", ascending=False).head(10))
print(cluster_summary)
