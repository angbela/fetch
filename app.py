import streamlit as st
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from geopy.distance import geodesic
import plotly.graph_objects as go

# ======================================================
# Streamlit config
# ======================================================
st.set_page_config(page_title="Directional Fetch Analyzer", layout="wide")
st.title("Directional Fetch to Nearest Coastline")
st.caption("Single coastline · Optional land/sea background · Directional fetch")

# ======================================================
# Sidebar inputs
# ======================================================
st.sidebar.header("Input Parameters")

lat = st.sidebar.number_input("Latitude", value=-2.375, format="%.6f")
lon = st.sidebar.number_input("Longitude", value=133.3125, format="%.6f")

bearing_step = st.sidebar.slider("Bearing step (degrees)", 1, 15, 5)
max_km = st.sidebar.slider("Max search distance (km)", 100, 2000, 1000, step=100)

buffer_distance = st.sidebar.slider(
    "Coast detection tolerance (degrees)",
    0.001, 0.02, 0.005, step=0.001
)

zoom_deg = st.sidebar.slider("Initial zoom window (degrees)", 1, 20, 6)

# ✅ NEW: background toggle (BEFORE Run)
show_bg = st.sidebar.checkbox(
    "Enable land/sea background",
    value=True
)

run = st.sidebar.button("🚀 Run Analysis")

# ======================================================
# Direction helpers
# ======================================================
def bearing_to_sector(b):
    if b >= 337.5 or b < 22.5:
        return "N"
    elif b < 67.5:
        return "NE"
    elif b < 112.5:
        return "E"
    elif b < 157.5:
        return "SE"
    elif b < 202.5:
        return "S"
    elif b < 247.5:
        return "SW"
    elif b < 292.5:
        return "W"
    else:
        return "NW"

sector_colors = {
    "N": "#1f77b4",
    "NE": "#17becf",
    "E": "#2ca02c",
    "SE": "#98df8a",
    "S": "#ffdf00",
    "SW": "#ff7f0e",
    "W": "#d62728",
    "NW": "#9467bd"
}

# ======================================================
# Load coastline
# ======================================================
@st.cache_data(show_spinner=False)
def load_coastline():
    gdf = gpd.read_file("data/coastline/ne_10m_coastline.shp")
    return unary_union(gdf.geometry), gdf

coastline, coastline_gdf = load_coastline()

# ======================================================
# Run analysis
# ======================================================
if run:
    origin = Point(lon, lat)
    bearings = np.arange(0, 360, bearing_step)

    rows = []
    trimmed_lines = {}

    with st.spinner("Computing directional fetch..."):
        for bearing in bearings:
            hit = None

            for d in range(1, max_km + 1):
                dest = geodesic(kilometers=d).destination((lat, lon), bearing)
                pt = Point(dest.longitude, dest.latitude)

                if coastline.distance(pt) <= buffer_distance:
                    hit = pt
                    break

            if hit:
                line = LineString([origin, hit])
                fetch_km = geodesic((lat, lon), (hit.y, hit.x)).kilometers
            else:
                pts = []
                for d in range(0, max_km + 1):
                    dest = geodesic(kilometers=d).destination((lat, lon), bearing)
                    pts.append((dest.longitude, dest.latitude))
                line = LineString(pts)
                fetch_km = max_km

            sector = bearing_to_sector(bearing)
            trimmed_lines[bearing] = line

            rows.append({
                "Bearing (deg)": bearing,
                "Direction": sector,
                "Fetch (km)": fetch_km
            })

    df_fetch = pd.DataFrame(rows)

    # ======================================================
    # Plot
    # ======================================================
    fig = go.Figure()

    # --- Natural Earth coastline (authoritative) ---
    for geom in coastline_gdf.geometry:
        if geom.geom_type == "LineString":
            x, y = geom.xy
            fig.add_trace(go.Scattergeo(
                lon=list(x),
                lat=list(y),
                mode="lines",
                line=dict(color="black", width=1),
                showlegend=False
            ))
        elif geom.geom_type == "MultiLineString":
            for g in geom.geoms:
                x, y = g.xy
                fig.add_trace(go.Scattergeo(
                    lon=list(x),
                    lat=list(y),
                    mode="lines",
                    line=dict(color="black", width=1),
                    showlegend=False
                ))

    # --- Fetch lines ---
    for bearing, line in trimmed_lines.items():
        sector = bearing_to_sector(bearing)
        x, y = line.xy
        fig.add_trace(go.Scattergeo(
            lon=list(x),
            lat=list(y),
            mode="lines",
            line=dict(width=1.4, color=sector_colors[sector]),
            opacity=0.85,
            showlegend=False
        ))

    # --- Origin ---
    fig.add_trace(go.Scattergeo(
        lon=[lon],
        lat=[lat],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="Origin"
    ))

    # ======================================================
    # GEO SETTINGS (conditional background)
    # ======================================================
    geo_kwargs = dict(
        projection_type="mercator",
        center=dict(lat=lat, lon=lon),
        lataxis_range=[lat - zoom_deg, lat + zoom_deg],
        lonaxis_range=[lon - zoom_deg, lon + zoom_deg],
        showcoastlines=False,
        showcountries=False,
        showlakes=False,
        showrivers=False,
        showframe=False
    )

    if show_bg:
        geo_kwargs.update(
            showland=True,
            landcolor="#efe8d8",
            showocean=True,
            oceancolor="#dcecf7"
        )
    else:
        geo_kwargs.update(
            showland=False,
            showocean=False,
            bgcolor="white"
        )

    fig.update_geos(**geo_kwargs)

    fig.update_layout(
        height=750,
        margin=dict(l=0, r=0, t=40, b=0),
        title="Directional Fetch"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # Fetch table
    # ======================================================
    st.subheader("Fetch Length per Bearing")
    st.dataframe(
        df_fetch.style.format({"Fetch (km)": "{:.2f}"}),
        use_container_width=True
    )

    # ======================================================
    # Effective fetch
    # ======================================================
    dir_order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    eff_rows = []

    for sector in dir_order:
        sub = df_fetch[df_fetch["Direction"] == sector]
        if sub.empty:
            continue

        theta = np.deg2rad(sub["Bearing (deg)"] - sub["Bearing (deg)"].mean())
        w = np.cos(theta) ** 2
        eff = np.sum(sub["Fetch (km)"] * w) / np.sum(w)

        eff_rows.append({
            "Direction": sector,
            "Effective Fetch (km)": eff
        })

    df_eff = pd.DataFrame(eff_rows)

    st.subheader("Effective Fetch per Direction")
    st.dataframe(
        df_eff.style.format({"Effective Fetch (km)": "{:.2f}"}),
        use_container_width=True
    )

    st.success("Analysis complete ✔️")

else:
    st.info("👈 Enter coordinates and click **Run Analysis**")
