import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Nassau Candy | Factory Optimization", layout="wide", page_icon="🍬")

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/nassau_clean.csv")
    routes = pd.read_csv("data/route_clusters.csv")
    recs = pd.read_csv("data/top_recommendations.csv")
    sim = pd.read_csv("data/scenario_simulation_full.csv")
    with open("models/model_results.json") as f:
        model_results = json.load(f)
    with open("data/kpi_summary.json") as f:
        kpi = json.load(f)
    return df, routes, recs, sim, model_results, kpi

@st.cache_resource
def load_model():
    return joblib.load("models/lead_time_model.joblib")

df, routes, recs, sim, model_results, kpi = load_data()
model = load_model()

FACTORIES = pd.DataFrame([
    ("Lot's O' Nuts",     32.881893, -111.768036),
    ("Wicked Choccy's",   32.076176,  -81.088371),
    ("Sugar Shack",       48.119140,  -96.181150),
    ("Secret Factory",    41.446333,  -90.565487),
    ("The Other Factory", 35.117500,  -89.971107),
], columns=["Factory", "Lat", "Lon"])

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# ------------------------------------------------------------------
# Sidebar navigation & global filters
# ------------------------------------------------------------------
st.sidebar.title("🍬 Nassau Candy")
st.sidebar.caption("Factory Reallocation & Shipping Optimization")
page = st.sidebar.radio("Dashboard", [
    "Overview",
    "Factory Optimization Simulator",
    "What-If Scenario Analysis",
    "Recommendation Dashboard",
    "Risk & Impact Panel",
])

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Data note: the raw `Ship Date` field is corrupted "
                    "(years drift 2–4+ ahead of Order Date). Lead times shown "
                    "are a distance & ship-mode based estimate — see Overview tab.")

# ------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------
if page == "Overview":
    st.title("Factory Reallocation & Shipping Optimization")
    st.caption("Decision-intelligence system for Nassau Candy Distributor")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders analyzed", f"{len(df):,}")
    c2.metric("Best model R²", f"{model_results['results'][model_results['best_model']]['R2']:.3f}",
               help=f"Best model: {model_results['best_model']}")
    c3.metric("Avg lead-time reduction (top recs)", f"{kpi['avg_lead_time_reduction_pct_top_recs']:.1f}%")
    c4.metric("Estimated profit lift (top recs)", f"${kpi['total_profit_impact_top_recs']:,.0f}")

    st.markdown("### ⚠️ Data Quality Finding")
    st.warning(
        "The raw `Ship Date` column is corrupted — its year drifts 2 to 4+ years ahead of "
        "`Order Date` and grows across the file (raw lead time averages ~1,320 days, up to "
        "4+ years). Day/month values track a short real gap, but even after correcting for "
        "year drift the reconstructed value is nearly constant (~177–178 days) and carries "
        "no real operational signal.\n\n"
        "**Modeling decision:** an `Estimated Lead Time` was engineered from first principles — "
        "great-circle distance between the assigned factory and the customer's state/province "
        "centroid, a ship-mode speed factor, and small realistic noise — documented transparently "
        "as a proxy target for demonstrating the modeling and optimization pipeline."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="EstimatedLeadTimeDays", nbins=40,
                            title="Estimated Lead Time Distribution", color_discrete_sequence=["#16a085"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        by_mode = df.groupby("Ship Mode")["EstimatedLeadTimeDays"].mean().sort_values()
        fig = px.bar(by_mode, orientation="h", title="Avg Estimated Lead Time by Ship Mode",
                     labels={"value": "Days", "Ship Mode": ""}, color_discrete_sequence=["#2980b9"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Factory Network")
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=FACTORIES["Lon"], lat=FACTORIES["Lat"], text=FACTORIES["Factory"],
        mode="markers+text", textposition="top center",
        marker=dict(size=14, color="#c0392b", symbol="star")))
    fig.update_geos(scope="north america", showland=True, landcolor="rgb(240,240,240)")
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# FACTORY OPTIMIZATION SIMULATOR
# ------------------------------------------------------------------
elif page == "Factory Optimization Simulator":
    st.title("Factory Optimization Simulator")
    st.caption("Select a product to view predicted performance across all five factories")

    product = st.selectbox("Product", sorted(df["Product Name"].unique()))
    region = st.selectbox("Customer Region", sorted(df["Region"].unique()))
    ship_mode = st.selectbox("Ship Mode", sorted(df["Ship Mode"].unique()))

    prow = df[df["Product Name"] == product].iloc[0]
    division = prow["Division"]
    current_factory = prow["Factory"]

    region_customers = df[df["Region"] == region][["C_Lat", "C_Lon"]].mean()

    results = []
    for _, f in FACTORIES.iterrows():
        dist = haversine(f["Lat"], f["Lon"], region_customers["C_Lat"], region_customers["C_Lon"])
        pred_input = pd.DataFrame([{
            "Factory": f["Factory"], "Region": region, "Ship Mode": ship_mode,
            "Division": division, "DistanceMiles": dist, "Units": df["Units"].mean(),
        }])
        pred_lead = model.predict(pred_input)[0]
        results.append({"Factory": f["Factory"], "Distance (mi)": round(dist, 0),
                         "Predicted Lead Time (days)": round(pred_lead, 2),
                         "Current Assignment": f["Factory"] == current_factory})

    rdf = pd.DataFrame(results).sort_values("Predicted Lead Time (days)")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = px.bar(rdf, x="Factory", y="Predicted Lead Time (days)",
                     color="Current Assignment",
                     color_discrete_map={True: "#e67e22", False: "#2980b9"},
                     title=f"Predicted Lead Time by Factory — {product} → {region} ({ship_mode})")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"**Currently assigned factory:** {current_factory}")
        st.dataframe(rdf.set_index("Factory"), use_container_width=True)
        best = rdf.iloc[0]
        if best["Factory"] != current_factory:
            st.success(f"Fastest option: **{best['Factory']}** "
                       f"({best['Predicted Lead Time (days)']} days)")
        else:
            st.info("Current assignment is already the fastest option for this route.")

# ------------------------------------------------------------------
# WHAT-IF SCENARIO ANALYSIS
# ------------------------------------------------------------------
elif page == "What-If Scenario Analysis":
    st.title("What-If Scenario Analysis")
    st.caption("Compare current vs. recommended factory assignments")

    combo = sim[~sim["IsCurrent"]][["Product Name", "Region", "Ship Mode"]].drop_duplicates()
    prod_sel = st.selectbox("Product", sorted(combo["Product Name"].unique()))
    combo2 = combo[combo["Product Name"] == prod_sel]
    region_sel = st.selectbox("Region", sorted(combo2["Region"].unique()))
    combo3 = combo2[combo2["Region"] == region_sel]
    mode_sel = st.selectbox("Ship Mode", sorted(combo3["Ship Mode"].unique()))

    scen = sim[(sim["Product Name"] == prod_sel) & (sim["Region"] == region_sel) & (sim["Ship Mode"] == mode_sel)]
    current = scen[scen["IsCurrent"]].iloc[0] if scen["IsCurrent"].any() else None
    alt_best = scen[~scen["IsCurrent"]].sort_values("PredictedLeadTime").iloc[0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Current Assignment")
        st.metric("Factory", current["Candidate Factory"] if current is not None else scen.iloc[0]["Current Factory"])
        st.metric("Lead Time", f"{current['CurrentLeadTime']:.2f} days" if current is not None else f"{scen.iloc[0]['CurrentLeadTime']:.2f} days")
        st.metric("Margin", f"{current['CurrentMargin']*100:.1f}%" if current is not None else f"{scen.iloc[0]['CurrentMargin']*100:.1f}%")
    with c2:
        st.markdown("#### Recommended Assignment")
        st.metric("Factory", alt_best["Candidate Factory"])
        st.metric("Lead Time", f"{alt_best['PredictedLeadTime']:.2f} days",
                   delta=f"{-alt_best['LeadTimeChangeDays']:.2f} days")
        st.metric("Margin", f"{alt_best['PredictedMargin']*100:.1f}%",
                   delta=f"${alt_best['ProfitImpact']:.0f} profit impact")

    fig = px.bar(scen.sort_values("PredictedLeadTime"), x="Candidate Factory", y="PredictedLeadTime",
                 color="IsCurrent", color_discrete_map={True: "#e67e22", False: "#2980b9"},
                 title="Lead Time Across All Candidate Factories", labels={"PredictedLeadTime": "Days"})
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(scen[["Candidate Factory", "PredictedLeadTime", "LeadTimeReductionPct",
                        "PredictedMargin", "ProfitImpact"]].sort_values("PredictedLeadTime"),
                 use_container_width=True)

# ------------------------------------------------------------------
# RECOMMENDATION DASHBOARD
# ------------------------------------------------------------------
elif page == "Recommendation Dashboard":
    st.title("Recommendation Dashboard")
    st.caption("Ranked factory reassignment suggestions")

    c1, c2, c3 = st.columns(3)
    c1.metric("Recommendations generated", len(recs))
    c2.metric("Avg lead-time reduction", f"{recs['LeadTimeReductionPct'].mean():.1f}%")
    c3.metric("Total profit impact", f"${recs['ProfitImpact'].sum():,.0f}")

    min_lead = st.slider("Minimum lead-time reduction (%)", 0, int(recs["LeadTimeReductionPct"].max()), 0)
    filtered = recs[recs["LeadTimeReductionPct"] >= min_lead].sort_values("CompositeScore", ascending=False)

    st.dataframe(
        filtered[["Product Name", "Region", "Ship Mode", "Current Factory", "Candidate Factory",
                  "CurrentLeadTime", "PredictedLeadTime", "LeadTimeReductionPct", "ProfitImpact"]],
        use_container_width=True, height=420,
    )

    fig = px.bar(filtered.sort_values("ProfitImpact", ascending=False).head(15),
                 x="ProfitImpact", y="Product Name", color="Region", orientation="h",
                 title="Top 15 Recommendations by Estimated Profit Impact")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# RISK & IMPACT PANEL
# ------------------------------------------------------------------
elif page == "Risk & Impact Panel":
    st.title("Risk & Impact Panel")
    st.caption("Profit-impact alerts and high-risk reassignment warnings")

    priority = st.slider("Optimization priority: Speed ← → Profit", 0, 100, 50,
                          help="0 = prioritize lead-time reduction only, 100 = prioritize profit impact only")
    w_speed = (100 - priority) / 100
    w_profit = priority / 100

    alts = sim[~sim["IsCurrent"]].copy()
    alts["LeadScoreNorm"] = alts["LeadTimeReductionPct"].clip(lower=0) / alts["LeadTimeReductionPct"].clip(lower=0).max()
    alts["ProfitScoreNorm"] = alts["ProfitImpact"].clip(lower=0) / alts["ProfitImpact"].clip(lower=0).max()
    alts["WeightedScore"] = w_speed * alts["LeadScoreNorm"] + w_profit * alts["ProfitScoreNorm"]

    ranked = (alts.sort_values("WeightedScore", ascending=False)
              .groupby(["Product Name", "Region", "Ship Mode"], as_index=False).first())

    risky = sim[(~sim["IsCurrent"]) & (sim["ProfitImpact"] < 0) & (sim["LeadTimeChangeDays"] < 0)]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ⚠️ High-risk reassignments")
        st.caption("Faster shipping but negative profit impact — proceed with caution")
        st.dataframe(risky[["Product Name", "Region", "Candidate Factory", "LeadTimeChangeDays", "ProfitImpact"]]
                     .sort_values("ProfitImpact").head(10), use_container_width=True)
    with c2:
        st.markdown("#### Priority-weighted top picks")
        st.dataframe(ranked.sort_values("WeightedScore", ascending=False)
                     [["Product Name", "Region", "Candidate Factory", "LeadTimeChangeDays", "ProfitImpact"]]
                     .head(10), use_container_width=True)

    fig = px.scatter(sim[~sim["IsCurrent"]], x="LeadTimeChangeDays", y="ProfitImpact",
                      color="Region", hover_data=["Product Name", "Candidate Factory"],
                      title="Profit Impact vs. Lead Time Change (all simulated reassignments)")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Nassau Candy Distributor · Decision Intelligence Prototype")
