import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("data/nassau_clean.csv")
model = joblib.load("models/lead_time_model.joblib")

factories = pd.DataFrame([
    ("Lot's O' Nuts",     32.881893, -111.768036),
    ("Wicked Choccy's",   32.076176,  -81.088371),
    ("Sugar Shack",       48.119140,  -96.181150),
    ("Secret Factory",    41.446333,  -90.565487),
    ("The Other Factory", 35.117500,  -89.971107),
], columns=["Factory", "F_Lat", "F_Lon"])

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# Per-product-region-shipmode current baseline
grp = df.groupby(["Product Name", "Division", "Factory", "Region", "Ship Mode"]).agg(
    C_Lat=("C_Lat", "mean"), C_Lon=("C_Lon", "mean"),
    Units=("Units", "mean"), OrderCount=("Order ID", "count"),
    Sales=("Sales", "sum"), Cost=("Cost", "sum"), GrossProfit=("Gross Profit", "sum"),
    CurrentLeadTime=("EstimatedLeadTimeDays", "mean"),
    CurrentDistance=("DistanceMiles", "mean"),
).reset_index()

rows = []
for _, r in grp.iterrows():
    current_margin = r["GrossProfit"] / r["Sales"] if r["Sales"] else 0
    best_alt = None
    for _, f in factories.iterrows():
        dist = haversine(f["F_Lat"], f["F_Lon"], r["C_Lat"], r["C_Lon"])
        pred_input = pd.DataFrame([{
            "Factory": f["Factory"], "Region": r["Region"], "Ship Mode": r["Ship Mode"],
            "Division": r["Division"], "DistanceMiles": dist, "Units": r["Units"],
        }])
        pred_lead = model.predict(pred_input)[0]
        # Simple logistics-cost proxy: shipping cost scales with distance;
        # estimate a per-mile cost rate from current cost/sales structure so
        # reassignment profit impact reflects distance change only.
        # Assume 0.6% of current unit cost per 100 miles as a shipping-sensitive slice.
        cost_per_mile = (r["Cost"] / max(r["CurrentDistance"], 1)) * 0.15
        new_cost = r["Cost"] - (cost_per_mile * r["CurrentDistance"]) + (cost_per_mile * dist)
        new_profit = r["Sales"] - new_cost
        new_margin = new_profit / r["Sales"] if r["Sales"] else 0
        rows.append({
            "Product Name": r["Product Name"], "Division": r["Division"], "Region": r["Region"],
            "Ship Mode": r["Ship Mode"], "Current Factory": r["Factory"],
            "Candidate Factory": f["Factory"], "IsCurrent": f["Factory"] == r["Factory"],
            "OrderCount": r["OrderCount"], "Sales": round(r["Sales"], 2),
            "CurrentLeadTime": round(r["CurrentLeadTime"], 2),
            "PredictedLeadTime": round(pred_lead, 2),
            "LeadTimeChangeDays": round(pred_lead - r["CurrentLeadTime"], 2),
            "LeadTimeReductionPct": round(100 * (r["CurrentLeadTime"] - pred_lead) / max(r["CurrentLeadTime"], 0.01), 1),
            "CurrentMargin": round(current_margin, 4),
            "PredictedMargin": round(new_margin, 4),
            "ProfitImpact": round(new_profit - r["GrossProfit"], 2),
        })

sim = pd.DataFrame(rows)
sim.to_csv("data/scenario_simulation_full.csv", index=False)

# ---------------------------------------------------------------
# Recommendation ranking: for each current (non-optimal) assignment,
# find the best alternate factory candidate (excluding current)
# ---------------------------------------------------------------
alts = sim[~sim["IsCurrent"]].copy()

# Composite score: weight lead-time reduction and profit impact
alts["LeadScore"] = alts["LeadTimeReductionPct"].clip(lower=0)
alts["ProfitScore"] = alts["ProfitImpact"].clip(lower=0)
# normalize
alts["LeadScoreNorm"] = alts["LeadScore"] / (alts["LeadScore"].max() or 1)
alts["ProfitScoreNorm"] = alts["ProfitScore"] / (alts["ProfitScore"].max() or 1)
alts["CompositeScore"] = 0.5 * alts["LeadScoreNorm"] + 0.5 * alts["ProfitScoreNorm"]

top_recs = (alts.sort_values("CompositeScore", ascending=False)
            .groupby(["Product Name", "Region", "Ship Mode"], as_index=False)
            .first())

top_recs = top_recs[top_recs["CompositeScore"] > 0].sort_values("CompositeScore", ascending=False)
top_recs.to_csv("data/top_recommendations.csv", index=False)

print("Total scenario rows:", len(sim))
print("Recommendations with positive composite gain:", len(top_recs))
print(top_recs[["Product Name","Region","Ship Mode","Current Factory","Candidate Factory",
                 "LeadTimeReductionPct","ProfitImpact","CompositeScore"]].head(15).to_string())

# Aggregate KPIs
kpi = {
    "avg_lead_time_reduction_pct_top_recs": round(top_recs["LeadTimeReductionPct"].mean(), 2),
    "total_profit_impact_top_recs": round(top_recs["ProfitImpact"].sum(), 2),
    "recommendation_coverage_pct": round(100 * len(top_recs) / grp.shape[0], 2),
    "n_product_region_shipmode_combos": int(grp.shape[0]),
    "n_recommendations": int(len(top_recs)),
}
import json
with open("data/kpi_summary.json", "w") as f:
    json.dump(kpi, f, indent=2)
print(kpi)
