import base64
import io
import os  # Added to check for local file existence
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt

# ============================
# 1. BASIC CONFIG
# ============================

DEFAULT_EXCEL_FILE = "Water sample results.xlsx"
DEFAULT_SHEET_NAME = "Water samples_Dry&Wet Season"

POLLUTANT_MAPPING = {
    "B": "Boron", "Al": "Aluminum", "V": "Vanadium", "Cr": "Chromium",
    "Mn": "Manganese", "Fe": "Iron", "Co": "Cobalt", "Ni": "Nickel",
    "Cu": "Copper", "Zn": "Zinc", "As": "Arsenic", "Se": "Selenium",
    "Sr": "Strontium", "Mo": "Molybdenum", "Cd": "Cadmium", "Sn": "Tin",
    "Sb": "Antimony", "Ba": "Barium", "Hg": "Mercury", "Pb": "Lead",
    "NO3": "Nitrate", "PO4": "Phosphate"
}

METHODOLOGY_ICON_SVG = """
<svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M13 21 A22 22 0 0 1 43 13" stroke="#8D3EBC" stroke-width="8" stroke-linecap="round" fill="none"/>
  <path d="M43 13 A22 22 0 0 1 51 43" stroke="#2E9BDE" stroke-width="8" stroke-linecap="round" fill="none"/>
  <path d="M51 43 A22 22 0 0 1 13 43" stroke="#E43D30" stroke-width="8" stroke-linecap="round" fill="none"/>
  <path d="M13 43 A22 22 0 0 1 13 21" stroke="#F2A032" stroke-width="8" stroke-linecap="round" fill="none"/>
  <path d="M10 44 H38 V39 L54 48 L38 57 V52 H10Z" fill="#41B649"/>
  <g transform="translate(32 32)">
    <circle r="12" fill="#4D5B65"/>
    <circle r="6" fill="#C7D2DA"/>
    <rect x="-2.5" y="-15" width="5" height="8" rx="2" fill="#4D5B65"/>
    <rect x="-2.5" y="-15" width="5" height="8" rx="2" fill="#4D5B65" transform="rotate(60)"/>
    <rect x="-2.5" y="-15" width="5" height="8" rx="2" fill="#4D5B65" transform="rotate(120)"/>
    <rect x="-2.5" y="-15" width="5" height="8" rx="2" fill="#4D5B65" transform="rotate(180)"/>
    <rect x="-2.5" y="-15" width="5" height="8" rx="2" fill="#4D5B65" transform="rotate(240)"/>
    <rect x="-2.5" y="-15" width="5" height="8" rx="2" fill="#4D5B65" transform="rotate(300)"/>
  </g>
</svg>
""".strip()

METHODOLOGY_ICON_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(METHODOLOGY_ICON_SVG.encode("utf-8")).decode("utf-8")

# ============================
# 2. NUMERICAL METHODS
# ============================

def extract_site(sample_id: str) -> str:
    try:
        parts = str(sample_id).split(" - ")
        return parts[1].strip()
    except Exception:
        return None

def trapezoidal_rule(y: np.ndarray, h: float) -> float:
    n = len(y) - 1
    if n < 1: return 0.0
    return h * (0.5 * y[0] + y[1:n].sum() + 0.5 * y[n])

def simpson_one_third_rule(y: np.ndarray, h: float) -> float:
    n = len(y) - 1
    if n < 1: return 0.0
    if n % 2 == 0:
        return (h/3.0) * (y[0] + y[n] + 4*y[1:n:2].sum() + 2*y[2:n-1:2].sum())
    else:
        return simpson_one_third_rule(y[:-1], h) + trapezoidal_rule(y[-2:], h)

def simpson_three_eighth_rule(y: np.ndarray, h: float) -> float:
    n = len(y) - 1
    if n < 1: return 0.0
    m = n - (n % 3)
    total = 0.0
    for j in range(0, m, 3):
        total += (3*h/8) * (y[j] + 3*y[j+1] + 3*y[j+2] + y[j+3])
    if m < n:
        total += trapezoidal_rule(y[m:], h)
    return total

def boole_rule(y: np.ndarray, h: float) -> float:
    n = len(y) - 1
    if n < 4: return trapezoidal_rule(y, h)
    m = n - (n % 4)
    total = 0.0
    for j in range(0, m, 4):
        total += (2*h/45) * (7*y[j] + 32*y[j+1] + 12*y[j+2] + 32*y[j+3] + 7*y[j+4])
    if m < n:
        total += trapezoidal_rule(y[m:], h)
    return total

def weddle_rule(y: np.ndarray, h: float) -> float:
    n = len(y) - 1
    if n < 6: return trapezoidal_rule(y, h)
    m = n - (n % 6)
    total = 0.0
    for j in range(0, m, 6):
        total += (3*h/10) * (y[j] + 5*y[j+1] + y[j+2] + 6*y[j+3] + y[j+4] + 5*y[j+5] + y[j+6])
    if m < n:
        total += trapezoidal_rule(y[m:], h)
    return total

# ============================
# 3. PROCESSING
# ============================

@st.cache_data
def load_data(excel_file, sheet_name):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    id_col = "Sample ID - Dry season"
    df = df.dropna(subset=[id_col])
    df = df[~df[id_col].astype(str).str.contains(r"\*BDL", na=False)]
    
    pollutant_cols_raw = df.columns[1:]
    df[pollutant_cols_raw] = df[pollutant_cols_raw].replace("BDL", 0)
    df[pollutant_cols_raw] = df[pollutant_cols_raw].replace("mg/l", np.nan)
    df[pollutant_cols_raw] = df[pollutant_cols_raw].apply(lambda col: pd.to_numeric(col, errors="coerce"))
    df.rename(columns=POLLUTANT_MAPPING, inplace=True)
    return df, id_col, list(df.columns[1:])

def process_season(df, id_col, pollutant_cols, season_prefix, H):
    df_season = df[df[id_col].astype(str).str.startswith(season_prefix)].copy()
    df_season["Site"] = df_season[id_col].astype(str).apply(extract_site)
    df_season = df_season.dropna(subset=["Site"])
    
    def sort_key(s):
        if isinstance(s, str) and s.startswith("S") and s[1:].isdigit():
            return int(s[1:])
        return 0
        
    site_means = df_season.groupby("Site")[pollutant_cols].mean()
    site_means = site_means.sort_index(key=lambda idx: [sort_key(s) for s in idx])
    
    n_sites = len(site_means)
    distances = np.arange(n_sites) * H
    site_means.insert(0, "Distance_km", distances)
    river_length = distances.max() if n_sites > 0 else 0.0
    return site_means, river_length

def compute_integrals(site_means, pollutant_cols, H, length, area):
    results = []
    for col in pollutant_cols:
        if col not in site_means.columns: continue
        y = np.nan_to_num(site_means[col].values.astype(float), nan=0.0)
        
        trap = trapezoidal_rule(y, H)
        simp13 = simpson_one_third_rule(y, H)
        simp38 = simpson_three_eighth_rule(y, H)
        boole = boole_rule(y, H)
        weddle = weddle_rule(y, H)
        
        # Using Weddle's rule for the total load estimation
        mass = (1e6 * area * weddle) if area > 0 else np.nan
        
        results.append({
            "Pollutant": col,
            "Trapezoidal": trap,
            "Simpson 1/3": simp13,
            "Simpson 3/8": simp38,
            "Boole": boole,
            "Weddle": weddle,
            "Total Load (kg)": mass / 1e6 if pd.notna(mass) else 0
        })
    return pd.DataFrame(results)

def compute_hotspots_df(site_means, pollutants):
    records = []
    for p in pollutants:
        if p in site_means.columns and not site_means[p].isna().all():
            idx_max = site_means[p].idxmax()
            val_max = site_means.loc[idx_max, p]
            dist_max = site_means.loc[idx_max, "Distance_km"]
            records.append({
                "Pollutant": p, "Max Site": idx_max, 
                "Distance (km)": dist_max, "Max Conc. (mg/L)": val_max
            })
    return pd.DataFrame(records)

# ============================
# 4. REPORTING (PDF)
# ============================

def create_static_plot_for_pdf(x, y, pollutant, season):
    plt.figure(figsize=(6, 3))
    plt.plot(x, y, marker='o', linestyle='-', color='tab:blue')
    plt.title(f"{pollutant} Concentration - {season}")
    plt.xlabel("Distance (km)")
    plt.ylabel("mg/L")
    plt.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def generate_advanced_pdf(mode, H, width, depth, data_pack):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w_page, h_page = A4
    y_pos = h_page - 50

    def write_line(text, offset=14, bold=False, font_size=10):
        nonlocal y_pos
        if y_pos < 50:
            c.showPage()
            y_pos = h_page - 50
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, font_size)
        c.drawString(40, y_pos, text)
        y_pos -= offset

    write_line("Environmental Impact Assessment Report", bold=True, font_size=16, offset=25)
    write_line(f"Analysis Mode: {mode}", bold=True)
    write_line(f"Parameters: Step H={H}km, Width={width}m, Depth={depth}m", offset=20)

    if mode == "Comparison":
        merged_df = data_pack['merged']
        write_line("Dry vs Wet Season Comparison (Weddle's Rule):", bold=True, offset=15)
        headers = ["Pollutant", "Dry (Weddle)", "Wet (Weddle)", "Diff"]
        x_offsets = [40, 150, 250, 350]
        for i, h in enumerate(headers): c.drawString(x_offsets[i], y_pos, h)
        y_pos -= 15
        for _, row in merged_df.iterrows():
            c.drawString(40, y_pos, str(row['Pollutant']))
            c.drawString(150, y_pos, f"{row['Weddle_Dry']:.3f}")
            c.drawString(250, y_pos, f"{row['Weddle_Wet']:.3f}")
            c.drawString(350, y_pos, f"{row['Difference_Wet_minus_Dry']:.3f}")
            y_pos -= 15
    else:
        results_df = data_pack['results']
        site_means = data_pack['sites']
        selected = data_pack['selected']
        write_line("Integration Results (Weddle's Rule):", bold=True, offset=15)
        headers = ["Pollutant", "Weddle Integral", "Total Load (kg)"]
        x_offsets = [40, 200, 350]
        for i, h in enumerate(headers): c.drawString(x_offsets[i], y_pos, h)
        y_pos -= 15
        for _, row in results_df.iterrows():
            if row['Pollutant'] in selected:
                c.drawString(40, y_pos, str(row['Pollutant']))
                c.drawString(200, y_pos, f"{row['Weddle']:.4f}")
                c.drawString(350, y_pos, f"{row['Total Load (kg)']:.2f}")
                y_pos -= 15
        y_pos -= 20
        write_line("Concentration Profiles:", bold=True, offset=15)
        x_vals = site_means["Distance_km"].values
        for pol in selected:
            if pol in site_means.columns:
                if y_pos < 250:
                    c.showPage(); y_pos = h_page - 50
                img_buf = create_static_plot_for_pdf(x_vals, site_means[pol].values, pol, mode)
                c.drawImage(ImageReader(img_buf), 40, y_pos - 200, width=400, height=200)
                y_pos -= 220
    c.save()
    buffer.seek(0)
    return buffer

# ============================
# 5. STREAMLIT APP MAIN
# ============================

def main():
    st.set_page_config(page_title="EcoNumerics: River Analysis", layout="wide", page_icon="🌊")
    st.markdown(
        f"""
        <style>
        .stTabs [data-baseweb="tab-list"] button:nth-child(3) p {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-weight: 600;
        }}
        .stTabs [data-baseweb="tab-list"] button:nth-child(3) p::before {{
            content: "";
            width: 26px;
            height: 26px;
            background-image: url('{METHODOLOGY_ICON_DATA_URI}');
            background-size: contain;
            background-repeat: no-repeat;
            display: inline-block;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("🌊 EcoNumerics: River Pollution Modeling")
        st.markdown("*Advanced Numerical Integration & Environmental Analysis Tool*")
    with col2:
        st.caption("Project: Numerical Analysis Lab")
        st.caption("Status: **v4.8 Select All Feature**")

    # --- SIMPLIFIED SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info(f"📁 Source: `{DEFAULT_EXCEL_FILE}`")
        
        if not os.path.exists(DEFAULT_EXCEL_FILE):
            st.error("⚠️ File missing!")
            st.warning(f"Please place '{DEFAULT_EXCEL_FILE}' in the app directory.")
            st.stop()

        st.divider()
        H = st.number_input("Step Size H (km)", 0.1, 5.0, 1.0)
        WIDTH = st.number_input("River Width (m)", 1.0, 100.0, 10.0)
        DEPTH = st.number_input("River Depth (m)", 0.5, 20.0, 2.0)
        area = WIDTH * DEPTH

    # --- DATA LOADING (LOCAL ONLY) ---
    try:
        df, id_col, pollutant_cols = load_data(DEFAULT_EXCEL_FILE, DEFAULT_SHEET_NAME)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    site_dry, len_dry = process_season(df, id_col, pollutant_cols, "D - ", H)
    site_wet, len_wet = process_season(df, id_col, pollutant_cols, "W - ", H)
    res_dry = compute_integrals(site_dry, pollutant_cols, H, len_dry, area)
    res_wet = compute_integrals(site_wet, pollutant_cols, H, len_wet, area)

    # --- MAIN TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Season Analysis", "⚖️ Comparison & Deviation", "Methodology"])

    # ---------------- TAB 1: SINGLE SEASON ----------------
    with tab1:
        season_choice = st.radio("Select Season", ["Dry Season", "Wet Season"], horizontal=True)
        current_res = res_dry if season_choice == "Dry Season" else res_wet
        current_site = site_dry if season_choice == "Dry Season" else site_wet
        
        m1, m2, m3 = st.columns(3)
        top_pollutant = current_res.loc[current_res["Total Load (kg)"].idxmax()]
        m1.metric("River Length", f"{len_dry:.2f} km")
        m2.metric("Highest Load", top_pollutant['Pollutant'], f"{top_pollutant['Total Load (kg)']:.2f} kg")
        m3.metric("Section Area", f"{area:.2f} m²")

        col_sel1, col_sel2 = st.columns([1, 2])
        with col_sel1:
            # UPDATED: Select All Logic
            select_all = st.checkbox("Select All Pollutants", value=False)
            if select_all:
                selected_pol = st.multiselect("Select Pollutants", pollutant_cols, default=pollutant_cols)
            else:
                selected_pol = st.multiselect("Select Pollutants", pollutant_cols, default=pollutant_cols[:2])
            
            st.markdown("##### 📍 Hotspot Details")
            if selected_pol:
                hotspots = compute_hotspots_df(current_site, selected_pol)
                st.dataframe(hotspots, hide_index=True)
        
        with col_sel2:
            if selected_pol:
                fig = go.Figure()
                x_axis = current_site["Distance_km"]
                for p in selected_pol:
                    y_vals = current_site[p]
                    fig.add_trace(go.Scatter(
                        x=x_axis, y=y_vals, mode='lines+markers', name=p,
                        fill='tozeroy', fillcolor='rgba(0,100,255,0.0)' 
                    ))
                    max_idx = y_vals.idxmax()
                    fig.add_annotation(
                        x=current_site.loc[max_idx, "Distance_km"],
                        y=y_vals[max_idx],
                        text="Max", showarrow=True, arrowhead=1, ax=0, ay=-30,
                        bgcolor="red", bordercolor="red", font=dict(color="white")
                    )
                
                # Dark template applied
                fig.update_layout(
                    title=f"Concentration Profile & Hotspots ({season_choice})",
                    xaxis_title="Distance (km)", yaxis_title="mg/L",
                    hovermode="x unified", template="plotly_dark", height=450
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Numerical Integration Results")
        
        numeric_cols = ["Trapezoidal", "Simpson 1/3", "Simpson 3/8", "Boole", "Weddle", "Total Load (kg)"]
        st.dataframe(current_res.style.format(subset=numeric_cols, formatter="{:.4f}"))
        
        if st.button("📄 Generate PDF Report"):
            pack = {'results': current_res, 'sites': current_site, 'selected': selected_pol}
            pdf_data = generate_advanced_pdf(season_choice, H, WIDTH, DEPTH, pack)
            st.download_button("Download Report", pdf_data, f"Report_{season_choice}.pdf", "application/pdf")

    # ---------------- TAB 2: COMPARISON ----------------
    with tab2:
        st.subheader("Dry vs Wet Season Comparison")
        comp_pollutant = st.selectbox("Select Pollutant to Compare", pollutant_cols)
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            c_fig = go.Figure()
            c_fig.add_trace(go.Scatter(x=site_dry["Distance_km"], y=site_dry[comp_pollutant], 
                                       name="Dry", line=dict(color='orange')))
            c_fig.add_trace(go.Scatter(x=site_wet["Distance_km"], y=site_wet[comp_pollutant], 
                                       name="Wet", line=dict(color='blue')))
            diff_vals = site_wet[comp_pollutant] - site_dry[comp_pollutant]
            c_fig.add_trace(go.Scatter(x=site_dry["Distance_km"], y=diff_vals, 
                                       name="Diff (Wet-Dry)", line=dict(color='gray', dash='dot')))
            
            # Dark template applied
            c_fig.update_layout(title=f"Seasonal Variation: {comp_pollutant}", template="plotly_dark", height=400)
            st.plotly_chart(c_fig, use_container_width=True)

        with col_c2:
            st.markdown("##### 🔍 Accuracy Deviation Matrix")
            st.caption("Comparison of methods against Weddle's Rule (Assumed True). Values are % Error.")
            dev_df = res_dry.copy()
            baseline = dev_df["Weddle"].replace(0, np.nan)
            cols = ["Trapezoidal", "Simpson 1/3", "Simpson 3/8", "Boole"]
            for c in cols: dev_df[c] = ((dev_df[c] - baseline) / baseline) * 100
            st.dataframe(dev_df[["Pollutant"] + cols].set_index("Pollutant").style.background_gradient(cmap="RdYlGn_r", axis=None).format("{:.2f}%"))

        # Comparison logic uses Weddle's Rule
        wed_dry = res_dry[["Pollutant", "Weddle"]].rename(columns={"Weddle": "Weddle_Dry"})
        wed_wet = res_wet[["Pollutant", "Weddle"]].rename(columns={"Weddle": "Weddle_Wet"})
        merged = pd.merge(wed_dry, wed_wet, on="Pollutant")
        merged["Difference_Wet_minus_Dry"] = merged["Weddle_Wet"] - merged["Weddle_Dry"]
        
        st.markdown("##### Total Load Difference (Weddle's Rule)")
        st.dataframe(merged)

        if st.button("📄 Generate Comparison Report"):
            pack = {'merged': merged}
            pdf_data = generate_advanced_pdf("Comparison", H, WIDTH, DEPTH, pack)
            st.download_button("Download Comparison PDF", pdf_data, "Comparison_Report.pdf", "application/pdf")

    # ---------------- TAB 3: METHODOLOGY ----------------
    with tab3:
        st.header("Mathematical Methodology")
        st.markdown("The following Newton-Cotes integration formulas are used to calculate total pollutant load:")

        st.subheader("1. Trapezoidal Rule (Degree 1)")
        st.markdown("Connects data points with straight lines.")
        st.latex(r"\int_{x_0}^{x_n} f(x) dx \approx \frac{h}{2} [y_0 + 2(y_1 + y_2 + \dots + y_{n-1}) + y_n]")

        st.divider()

        st.subheader("2. Simpson's 1/3 Rule (Degree 2)")
        st.markdown("Fits parabolas (quadratic curves) through groups of 3 points.")
        st.latex(r"\int_{x_0}^{x_n} f(x) dx \approx \frac{h}{3} [y_0 + 4(y_1 + y_3 + \dots) + 2(y_2 + y_4 + \dots) + y_n]")

        st.divider()

        st.subheader("3. Simpson's 3/8 Rule (Degree 3)")
        st.markdown("Fits cubic curves through groups of 4 points.")
        st.latex(r"\int_{x_0}^{x_n} f(x) dx \approx \frac{3h}{8} [y_0 + 3y_1 + 3y_2 + 2y_3 + 3y_4 + \dots + y_n]")

        st.divider()

        st.subheader("4. Boole's Rule (Degree 4)")
        st.markdown("Uses 5 points at a time with a degree 4 polynomial.")
        st.latex(r"\int_{x_0}^{x_4} f(x) dx \approx \frac{2h}{45} [7y_0 + 32y_1 + 12y_2 + 32y_3 + 7y_4]")

        st.divider()

        st.subheader("5. Weddle's Rule (Degree 6)")
        st.markdown("Uses 7 points at a time. Highly accurate for smooth data.")
        st.latex(r"\int_{x_0}^{x_6} f(x) dx \approx \frac{3h}{10} [y_0 + 5y_1 + y_2 + 6y_3 + y_4 + 5y_5 + y_6]")

if __name__ == "__main__":
    main()