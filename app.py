# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# IMPORT THE LOGIC FROM YOUR COLLEAGUE'S FILE
from logic import extract_text_from_uploaded_file, extract_pnl_with_llm

st.set_page_config(page_title="How X Makes Money", layout="wide")

# -------------------------------------------------------------------
# 1. Aesthetics & UI Configuration
# -------------------------------------------------------------------

# Define your colors here (UI Job)
CATEGORY_COLORS = {
    "Revenue": "#4285F4",       # Blue
    "COGS": "#DB4437",          # Red
    "Gross Profit": "#BDBDBD",  # Grey
    "R&D": "#AB47BC",           # Purple
    "Sales & Marketing": "#F4B400", # Yellow
    "G&A": "#00ACC1",           # Teal
    "Other Opex": "#8D6E63",    # Brown
    "Tax": "#E91E63",           # Pink
    "Net Income": "#0F9D58"     # Green
}

st.sidebar.header("Design Settings")
brand_color = st.sidebar.color_picker("Brand Color", "#4285F4")
min_share = st.sidebar.slider("Min Revenue Share", 0.0, 0.2, 0.05, 0.01)

st.title("How X Makes Money (Collaborative Edition)")

# -------------------------------------------------------------------
# 2. Input Logic (Connecting UI to Logic)
# -------------------------------------------------------------------

col1, col2 = st.columns([1, 2])
with col1:
    input_mode = st.radio("Input Mode", ["Upload File", "Paste Text", "Use Sample Data"])

raw_text = ""

if input_mode == "Upload File":
    f = st.file_uploader("Upload 10-K / Annual Report (PDF)", type=["pdf", "txt"])
    if f: 
        # Call the logic function
        raw_text = extract_text_from_uploaded_file(f)
        if raw_text.startswith("ERROR"):
            st.error(raw_text)
            raw_text = ""

elif input_mode == "Paste Text":
    raw_text = st.text_area("Paste Income Statement text")

elif input_mode == "Use Sample Data":
    st.session_state.raw_df = pd.DataFrame({
        "Item": ["Search", "YouTube", "Cloud", "TAC", "R&D", "S&M", "Tax"],
        "Amount": [50000, 8000, 6000, 12000, 9000, 4000, 3000],
        "Category": ["Revenue", "Revenue", "Revenue", "COGS", "R&D", "Sales & Marketing", "Tax"]
    })

# The "Run" Button
if (input_mode != "Use Sample Data") and raw_text:
    if st.button("Analyze with AI"):
        with st.spinner("Scanning document and analyzing financials..."):
            try:
                # Call the logic function
                df_result, company, currency = extract_pnl_with_llm(raw_text)
                st.session_state.raw_df = df_result
                if company: st.success(f"Identified: {company} ({currency})")
            except Exception as e:
                st.error(f"AI Error: {e}")

# -------------------------------------------------------------------
# 3. Visualization Logic (UI Job)
# -------------------------------------------------------------------

if "raw_df" in st.session_state and st.session_state.raw_df is not None:
    df = st.session_state.raw_df.copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
    
    st.subheader("1. Data Review")
    edited_df = st.data_editor(
        df,
        column_config={
            "Category": st.column_config.SelectboxColumn(
                "Category",
                options=list(CATEGORY_COLORS.keys()) + ["Ignore"],
                required=True
            )
        },
        use_container_width=True,
        num_rows="dynamic"
    )

    clean_df = edited_df[edited_df["Category"] != "Ignore"].copy()
    
    # Logic to build the Sankey numbers
    grp = clean_df.groupby("Category")["Amount"].sum()
    total_revenue = grp.get("Revenue", 0)
    total_cogs = grp.get("COGS", 0)
    gross_profit = total_revenue - total_cogs
    
    opex_cats = ["R&D", "Sales & Marketing", "G&A", "Other Opex"]
    total_opex = sum(grp.get(c, 0) for c in opex_cats)
    operating_profit = gross_profit - total_opex
    tax = grp.get("Tax", 0)
    net_income = operating_profit - tax

    # Display Metrics
    k1, k2, k3, k4 = st.columns(4)
    if total_revenue > 0:
        k1.metric("Revenue", f"{total_revenue:,.0f}")
        k2.metric("Gross Margin", f"{(gross_profit/total_revenue)*100:.1f}%")
        k3.metric("Op Margin", f"{(operating_profit/total_revenue)*100:.1f}%")
        k4.metric("Net Margin", f"{(net_income/total_revenue)*100:.1f}%")

    # Generate Chart
    if st.button("Generate Sankey Diagram"):
        labels, sources, targets, values, colors = [], [], [], [], []
        
        label_idx = {}
        def get_idx(name):
            if name not in label_idx:
                label_idx[name] = len(labels)
                labels.append(name)
                if name in CATEGORY_COLORS: colors.append(CATEGORY_COLORS[name])
                elif name == "Total Revenue": colors.append(brand_color)
                else: colors.append("rgba(180,180,180,0.5)")
            return label_idx[name]

        # Build Flows
        # 1. Revenue Sources -> Total Revenue
        for _, row in clean_df[clean_df["Category"] == "Revenue"].iterrows():
            if total_revenue > 0 and (row["Amount"] < (total_revenue * min_share)):
                s_idx = get_idx("Other Revenue")
            else:
                s_idx = get_idx(row["Item"])
            sources.append(s_idx); targets.append(get_idx("Total Revenue")); values.append(row["Amount"])

        # 2. Total Revenue -> COGS & Gross Profit
        if total_cogs > 0:
            sources.append(get_idx("Total Revenue")); targets.append(get_idx("Cost of Revenue")); values.append(total_cogs)
        sources.append(get_idx("Total Revenue")); targets.append(get_idx("Gross Profit")); values.append(gross_profit)

        # 3. Gross Profit -> Opex
        for cat in opex_cats:
            amt = grp.get(cat, 0)
            if amt > 0:
                sources.append(get_idx("Gross Profit")); targets.append(get_idx(cat)); values.append(amt)
        sources.append(get_idx("Gross Profit")); targets.append(get_idx("Operating Profit")); values.append(operating_profit)

        # 4. Operating Profit -> Tax & Net Income
        if tax > 0:
            sources.append(get_idx("Operating Profit")); targets.append(get_idx("Tax")); values.append(tax)
        sources.append(get_idx("Operating Profit")); targets.append(get_idx("Net Income")); values.append(net_income)

        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=20, thickness=20, line=dict(color="black", width=0.5), label=labels, color=colors),
            link=dict(source=sources, target=targets, value=values, color="rgba(200,200,200,0.3)")
        )])
        
        fig.update_layout(title_text=f"Financial Flow: {st.session_state.get('detected_company', '')}", font_size=14, height=600)
        st.plotly_chart(fig, use_container_width=True)
