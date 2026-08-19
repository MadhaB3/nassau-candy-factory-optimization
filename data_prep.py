"""
Nassau Candy Distributor — Data Preparation
=============================================
Handles a critical data-quality defect in the raw file (see README/paper):
the `Ship Date` field's YEAR component is corrupted (drifts 2-4+ years ahead
of Order Date, growing over the file), while month/day roughly track a
short real shipping gap. Raw (Ship Date - Order Date) is therefore useless
(mean ~1320 days). We reconstruct a realistic lead time from the month-day
delta, then, because the reconstructed value is itself nearly constant
(~177-178 days for every row, i.e. also not real operational signal), we
build an engineered "Estimated Lead Time" target from first-principles
logistics factors: great-circle distance from the assigned factory to the
customer's state/province centroid, shipping mode speed, and small
realistic noise. This is documented transparently as a modeling proxy.
"""
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------
df = pd.read_csv('data/raw.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
df['Ship Date_raw'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y')
df['RawLeadTimeDays'] = (df['Ship Date_raw'] - df['Order Date']).dt.days

# ---------------------------------------------------------------
# 2. Factory coordinates & product->factory correlation (from brief)
# ---------------------------------------------------------------
factories = pd.DataFrame([
    ("Lot's O' Nuts",     32.881893, -111.768036),
    ("Wicked Choccy's",   32.076176,  -81.088371),
    ("Sugar Shack",       48.119140,  -96.181150),
    ("Secret Factory",    41.446333,  -90.565487),
    ("The Other Factory", 35.117500,  -89.971107),
], columns=["Factory", "F_Lat", "F_Lon"])

product_factory = pd.DataFrame([
    ("Chocolate", "Wonka Bar - Nutty Crunch Surprise", "Lot's O' Nuts"),
    ("Chocolate", "Wonka Bar - Fudge Mallows", "Lot's O' Nuts"),
    ("Chocolate", "Wonka Bar -Scrumdiddlyumptious", "Lot's O' Nuts"),
    ("Chocolate", "Wonka Bar - Milk Chocolate", "Wicked Choccy's"),
    ("Chocolate", "Wonka Bar - Triple Dazzle Caramel", "Wicked Choccy's"),
    ("Sugar", "Laffy Taffy", "Sugar Shack"),
    ("Sugar", "SweeTARTS", "Sugar Shack"),
    ("Sugar", "Nerds", "Sugar Shack"),
    ("Sugar", "Fun Dip", "Sugar Shack"),
    ("Other", "Fizzy Lifting Drinks", "Sugar Shack"),
    ("Sugar", "Everlasting Gobstopper", "Secret Factory"),
    ("Sugar", "Hair Toffee", "The Other Factory"),
    ("Other", "Lickable Wallpaper", "Secret Factory"),
    ("Other", "Wonka Gum", "Secret Factory"),
    ("Other", "Kazookles", "The Other Factory"),
], columns=["Division_ref", "Product Name", "Factory"])

df = df.merge(product_factory[["Product Name", "Factory"]], on="Product Name", how="left")
df = df.merge(factories, on="Factory", how="left")

# ---------------------------------------------------------------
# 3. Customer location centroid lookup (state/province -> lat/lon)
#    Standard public geographic centroids, US states + Canadian provinces.
# ---------------------------------------------------------------
state_coords = {
    "Alabama": (32.806671, -86.791130), "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221), "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564), "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371), "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.897438, -77.026817), "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074), "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137), "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526), "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067), "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927), "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106), "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192), "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368), "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082), "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896), "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482), "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419), "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915), "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938), "Pennsylvania": (40.590752, -77.209755),
    "Rhode Island": (41.680893, -71.511780), "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828), "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461), "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686), "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494), "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508), "Wyoming": (42.755966, -107.302490),
    # Canadian provinces
    "Alberta": (53.933327, -116.576504), "British Columbia": (53.726669, -127.647621),
    "Manitoba": (53.760860, -98.813873), "New Brunswick": (46.565314, -66.461914),
    "Newfoundland and Labrador": (53.135509, -57.660435), "Nova Scotia": (44.681488, -63.744311),
    "Ontario": (51.253775, -85.323214), "Prince Edward Island": (46.510712, -63.416817),
    "Quebec": (52.939916, -73.549136), "Saskatchewan": (52.935162, -106.450864),
}
sc = pd.DataFrame([(k, v[0], v[1]) for k, v in state_coords.items()],
                   columns=["State/Province", "C_Lat", "C_Lon"])
df = df.merge(sc, on="State/Province", how="left")

# ---------------------------------------------------------------
# 4. Great-circle distance (miles), factory -> customer
# ---------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

df["DistanceMiles"] = haversine(df["F_Lat"], df["F_Lon"], df["C_Lat"], df["C_Lon"])

# ---------------------------------------------------------------
# 5. Engineered "Estimated Lead Time" target (days)
#    base handling + distance/speed by ship mode + small noise
# ---------------------------------------------------------------
mode_speed_mph = {"Same Day": 1400, "First Class": 550, "Second Class": 340, "Standard Class": 220}
mode_base_days = {"Same Day": 0.4, "First Class": 1.0, "Second Class": 1.6, "Standard Class": 2.2}

df["ShipSpeed"] = df["Ship Mode"].map(mode_speed_mph)
df["ShipBase"] = df["Ship Mode"].map(mode_base_days)
noise = np.random.normal(0, 0.5, len(df))
df["EstimatedLeadTimeDays"] = (df["ShipBase"] + df["DistanceMiles"] / df["ShipSpeed"] + noise).clip(lower=0.5).round(2)

# ---------------------------------------------------------------
# 6. Save cleaned dataset
# ---------------------------------------------------------------
keep_cols = ["Row ID", "Order ID", "Order Date", "Ship Mode", "Customer ID",
             "Country/Region", "City", "State/Province", "Postal Code",
             "Division", "Region", "Product ID", "Product Name",
             "Sales", "Units", "Gross Profit", "Cost",
             "Factory", "F_Lat", "F_Lon", "C_Lat", "C_Lon",
             "DistanceMiles", "EstimatedLeadTimeDays", "RawLeadTimeDays"]
df[keep_cols].to_csv("data/nassau_clean.csv", index=False)

print("Rows:", len(df))
print("Missing factory assignment:", df["Factory"].isna().sum())
print("Missing coords:", df["C_Lat"].isna().sum())
print(df["EstimatedLeadTimeDays"].describe())
print(df.groupby("Ship Mode")["EstimatedLeadTimeDays"].mean())
