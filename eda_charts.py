import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/nassau_clean.csv")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# 1. Raw vs reconstructed lead time (data quality issue)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(df["RawLeadTimeDays"], bins=40, color="#c0392b", alpha=0.8)
ax.set_title("Raw Ship Date - Order Date (corrupted field)")
ax.set_xlabel("Days"); ax.set_ylabel("Order count")
fig.tight_layout(); fig.savefig("charts/01_raw_leadtime_defect.png"); plt.close(fig)

# 2. Distance distribution by factory
fig, ax = plt.subplots(figsize=(6.5, 3.8))
df.boxplot(column="DistanceMiles", by="Factory", ax=ax, rot=25)
ax.set_title("Shipping Distance by Factory"); ax.set_ylabel("Miles")
plt.suptitle("")
fig.tight_layout(); fig.savefig("charts/02_distance_by_factory.png"); plt.close(fig)

# 3. Estimated lead time by ship mode
fig, ax = plt.subplots(figsize=(6, 3.5))
df.boxplot(column="EstimatedLeadTimeDays", by="Ship Mode", ax=ax, rot=15)
ax.set_title("Estimated Lead Time by Ship Mode"); ax.set_ylabel("Days")
plt.suptitle("")
fig.tight_layout(); fig.savefig("charts/03_leadtime_by_shipmode.png"); plt.close(fig)

# 4. Sales and margin by region
reg = df.groupby("Region").agg(Sales=("Sales", "sum"), Margin=("Gross Profit", "sum"))
reg["MarginPct"] = reg["Margin"] / reg["Sales"]
fig, ax1 = plt.subplots(figsize=(6, 3.8))
reg["Sales"].plot(kind="bar", ax=ax1, color="#2980b9", alpha=0.8)
ax1.set_ylabel("Total Sales ($)", color="#2980b9")
ax2 = ax1.twinx()
reg["MarginPct"].plot(ax=ax2, color="#e67e22", marker="o", linewidth=2)
ax2.set_ylabel("Gross Margin %", color="#e67e22")
ax1.set_title("Sales & Margin by Region")
fig.tight_layout(); fig.savefig("charts/04_sales_margin_region.png"); plt.close(fig)

# 5. Model comparison
import json
with open("models/model_results.json") as f:
    mr = json.load(f)["results"]
mdf = pd.DataFrame(mr).T
fig, ax = plt.subplots(figsize=(6, 3.5))
mdf["R2"].plot(kind="bar", ax=ax, color="#27ae60")
ax.set_title("Model R\u00b2 - Predicting Estimated Lead Time")
ax.set_ylabel("R\u00b2"); ax.set_xticklabels(mdf.index, rotation=15)
fig.tight_layout(); fig.savefig("charts/05_model_comparison.png"); plt.close(fig)

# 6. Route cluster scatter
rc = pd.read_csv("data/route_clusters.csv")
fig, ax = plt.subplots(figsize=(6.5, 4))
colors = {"Consistently Slow Route": "#c0392b", "High-Performing Route": "#27ae60",
          "Moderate-Performance Route": "#f39c12", "Margin-Risk Route": "#8e44ad"}
for label, grp in rc.groupby("ClusterLabel"):
    ax.scatter(grp["AvgDistance"], grp["AvgLeadTime"], label=label,
               color=colors.get(label, "#333"), s=60, alpha=0.8)
ax.set_xlabel("Avg Distance (miles)"); ax.set_ylabel("Avg Lead Time (days)")
ax.set_title("Route Clustering: Factory \u2192 Region Combinations")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("charts/06_route_clusters.png"); plt.close(fig)

# 7. Top recommendations - profit impact
tr = pd.read_csv("data/top_recommendations.csv").sort_values("ProfitImpact", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(7, 4))
labels = tr["Product Name"].str.slice(0, 18) + " (" + tr["Region"] + ")"
ax.barh(labels, tr["ProfitImpact"], color="#16a085")
ax.set_xlabel("Estimated Profit Impact ($)")
ax.set_title("Top 10 Reassignment Recommendations by Profit Impact")
ax.invert_yaxis()
fig.tight_layout(); fig.savefig("charts/07_top_recommendations.png"); plt.close(fig)

print("Charts generated:")
import os
for f in sorted(os.listdir("charts")):
    print(" ", f)
