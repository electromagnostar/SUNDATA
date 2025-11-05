
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime
from zoneinfo import ZoneInfo

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Solar Performance Analyzer", layout="wide")
st.title("🔆 Solar Performance Analyzer")
st.caption("Upload your solar dataset and sunrise/sunset file to calculate generation metrics and visualize results.")

# ============================================
# SIDEBAR PARAMETERS
# ============================================
with st.sidebar:
    st.header("⚙️ Parameters")
    capacity_kw = st.number_input("Site Capacity (kW)", value=1500.4, min_value=0.0, step=0.1)
    min_irr = st.number_input("CPR: Min Irradiance (W/m²)", value=100.0, min_value=0.0, step=10.0)
    max_ctrl = st.number_input("CPR: Max Control Meter (kW)", value=665.0, min_value=0.0, step=5.0)
    min_pr = st.number_input("CPR: Min Performance Ratio", value=0.6, min_value=0.0, max_value=2.0, step=0.05)
    apply_iqr = st.checkbox("Apply per-day 3×IQR outlier cleaning", value=True)

# ============================================
# FILE UPLOAD
# ============================================
col1, col2 = st.columns(2)
with col1:
    solar_file = st.file_uploader("📄 Upload Solar Data (Excel or CSV)", type=["xlsx", "xls", "csv"])
with col2:
    sun_file = st.file_uploader("🌅 Upload Sunrise/Sunset Data (Excel or CSV)", type=["xlsx", "xls", "csv"])

run = st.button("🚀 Run Analysis")

# ============================================
# FUNCTIONS
# ============================================
def per_day_3x_iqr_clean(df, numeric_cols, date_col="Date"):
    df_out = df.copy()
    for col in numeric_cols:
        if col not in df_out.columns:
            continue
        grp = df_out.groupby(date_col)[col]
        q1 = grp.transform(lambda x: x.quantile(0.25))
        q3 = grp.transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1
        low, high = q1 - 3 * iqr, q3 + 3 * iqr
        df_out.loc[(df_out[col] < low) | (df_out[col] > high), col] = np.nan
    return df_out

def to_excel(df_data, df_avg, df_excl):
    with pd.ExcelWriter(io.BytesIO(), engine="xlsxwriter") as writer:
        df_data.to_excel(writer, index=False, sheet_name="Processed Data")
        df_avg.to_excel(writer, index=False, sheet_name="Daily Averages")
        df_excl.to_excel(writer, index=False, sheet_name="Exclusion Reasons")
        return writer.book.filename.getvalue()

# ============================================
# MAIN ANALYSIS
# ============================================
if run:
    if not solar_file or not sun_file:
        st.error("Please upload both the solar data and the sunrise/sunset data files.")
        st.stop()

    try:
        # ---- Load solar data ----
        if solar_file.name.endswith(".csv"):
            df = pd.read_csv(solar_file)
        else:
            df = pd.read_excel(solar_file)

        # Normalize headers
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns={
            "Date Time (UTC)": "DateTime (UTC)",
            "Panel Irradiance": "Panel Irradiance",
            "Energy GEN (kWh)": "Energy GEN (kWh)",
            "Control Meter (kW Total)": "Control Meter (kW Total)",
        })

        if "DateTime (UTC)" not in df.columns:
            if "Date" in df.columns and "Time (UTC)" in df.columns:
                df["DateTime (UTC)"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time (UTC)"].astype(str), errors="coerce")
            else:
                raise ValueError("File must contain 'DateTime (UTC)' or separate 'Date' and 'Time (UTC)' columns.")

        df["Date"] = pd.to_datetime(df["DateTime (UTC)"], errors="coerce").dt.date
        df["Time (UTC)"] = pd.to_datetime(df["DateTime (UTC)"], errors="coerce").dt.time
        df["timestamp_utc"] = pd.to_datetime(df["DateTime (UTC)"], errors="coerce")

        # ---- Load sunrise/sunset ----
        if sun_file.name.endswith(".csv"):
            sun = pd.read_csv(sun_file)
        else:
            sun = pd.read_excel(sun_file)

        sun.columns = [str(c).strip().lower() for c in sun.columns]
        sun["date"] = pd.to_datetime(sun["date"], errors="coerce").dt.date
        sun["sunrise (bst)"] = pd.to_datetime(sun["sunrise (bst)"].astype(str), errors="coerce").dt.time
        sun["sunset (bst)"] = pd.to_datetime(sun["sunset (bst)"].astype(str), errors="coerce").dt.time

        # ---- Daylight detection ----
        df["time_bst"] = df.apply(lambda r: datetime.combine(r["Date"], r["Time (UTC)"]).replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/London")).time(), axis=1)
        df = df.merge(sun, left_on="Date", right_on="date", how="left")
        df["is_daylight"] = df.apply(lambda r: r["sunrise (bst)"] <= r["time_bst"] <= r["sunset (bst)"] if pd.notna(r["sunrise (bst)"]) and pd.notna(r["sunset (bst)"]) else False, axis=1)

        # ---- Apply daylight rules ----
        df.loc[~df["is_daylight"], "Panel Irradiance"] = 0.0
        df.loc[df["Panel Irradiance"] < 0, "Panel Irradiance"] = 0.0

        # Flatten energy at night
        energy = df["Energy GEN (kWh)"].fillna(0).astype(float).values
        for i in range(1, len(energy)):
            if not df.loc[i, "is_daylight"] and energy[i] > energy[i-1]:
                energy[i] = energy[i-1]
            if energy[i] < 0:
                energy[i] = 0
        df["Energy GEN (kWh)"] = energy

        # ---- Control Meter recalculation ----
        delta_e = np.insert(np.diff(energy), 0, 0)
        hours = df["timestamp_utc"].diff().dt.total_seconds().fillna(300) / 3600.0  # default 5-min step
        df["Control Meter (kW Total)"] = np.maximum(delta_e / hours, 0)

        # ---- Actual + Expected generation ----
        df["Actual Generation (kWh)"] = np.maximum(delta_e, 0)
        df["Expected Generation (kWh)"] = capacity_kw * (df["Panel Irradiance"].clip(lower=0) / 1000.0) * hours

        # ---- PR & CPR ----
        df["Performance Ratio"] = np.where(df["Expected Generation (kWh)"] > 0, df["Actual Generation (kWh)"] / df["Expected Generation (kWh)"], np.nan)
        valid_cpr = (df["Panel Irradiance"] >= min_irr) & (df["Control Meter (kW Total)"] <= max_ctrl) & (df["Performance Ratio"] >= min_pr)
        df["Contractual Performance Ratio"] = np.where(valid_cpr, df["Performance Ratio"], np.nan)

        # ---- Exclusion reasons ----
        def exclusion_reason(row):
            reasons = []
            if not (row["Expected Generation (kWh)"] > 0):
                reasons.append("Expected Gen ≤ 0")
            if row["Panel Irradiance"] < min_irr:
                reasons.append(f"Irradiance < {min_irr}")
            if row["Control Meter (kW Total)"] > max_ctrl:
                reasons.append(f"Meter > {max_ctrl}")
            if row["Performance Ratio"] < min_pr:
                reasons.append(f"PR < {min_pr}")
            return "; ".join(reasons)
        df["Exclusion Reason"] = df.apply(exclusion_reason, axis=1)

        # ---- Outlier cleaning ----
        if apply_iqr:
            numeric_cols = ["Panel Irradiance","Control Meter (kW Total)","Actual Generation (kWh)","Expected Generation (kWh)","Performance Ratio","Contractual Performance Ratio"]
            df = per_day_3x_iqr_clean(df, numeric_cols)

        # ---- Averages ----
        avg = df.groupby("Time (UTC)")[["Panel Irradiance","Control Meter (kW Total)","Actual Generation (kWh)","Expected Generation (kWh)","Performance Ratio","Contractual Performance Ratio"]].mean().reset_index()
        excl = df["Exclusion Reason"].replace("", "No Exclusion").value_counts().reset_index()
        excl.columns = ["Exclusion Reason","Count"]

        # ---- Graphs ----
        st.subheader("📊 Results")
        fig1 = px.line(avg, x="Time (UTC)", y="Panel Irradiance", title="Average Panel Irradiance (W/m²)")
        fig2 = px.line(avg, x="Time (UTC)", y="Control Meter (kW Total)", title="Average Control Meter (kW)")
        fig3 = px.line(avg, x="Time (UTC)", y=["Actual Generation (kWh)","Expected Generation (kWh)"], title="Actual vs Expected Generation")
        fig4 = px.line(avg, x="Time (UTC)", y=["Performance Ratio","Contractual Performance Ratio"], title="Performance Ratios (PR & CPR)")
        fig5 = px.bar(excl, x="Exclusion Reason", y="Count", title="Exclusion Reasons")

        for fig in [fig1, fig2, fig3, fig4, fig5]:
            st.plotly_chart(fig, use_container_width=True)

        # ---- Download ----
        excel_data = to_excel(df, avg, excl)
        st.download_button("⬇️ Download Excel Report", data=excel_data, file_name="Solar_Performance_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.success("✅ Analysis complete!")

    except Exception as e:
        st.error(f"Processing failed: {e}")
