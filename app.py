import streamlit as st

st.write("### DEBUG: Vision Auditor build 2025-11-28")


# app.py
# "How X Makes Money" - Vision Auditor Edition
# VERSION: Single-File, Multi-Screenshot Reconciliation

import os
import json
import base64
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Try importing OpenAI (Handles missing library gracefully)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# -------------------------------------------------------------------
# 0. App Configuration & Styles
# -------------------------------------------------------------------
st.set_page_config(page_title="Financial Flow Auditor", layout="wide")

# Color Palette for Sankey Diagram
CATEGORY_COLORS = {
    "Revenue": "#4285F4",       # Blue (Sources)
    "COGS": "#DB4437",          # Red (Direct Costs)
    "Gross Profit": "#BDBDBD",  # Grey (Calculated Node)
    "R&D": "#AB47BC",           # Purple
    "Sales & Marketing": "#F4B400", # Yellow
    "G&A": "#00ACC1",           # Teal
    "Other Opex": "#8D6E63",    # Brown
    "Tax": "#E91E63",           # Pink
    "Net Income": "#0F9D58",    # Green
    "Unallocated": "#9E9E9E",   # Grey (Reconciliation Gaps)
    "Eliminations": "#5f6368"   # Dark Grey (Inter-segment)
}

# -------------------------------------------------------------------
# 1. Backend Logic: The AI "Vision Auditor"
# -------------------------------------------------------------------

def get_openai_client():
    """Initializes OpenAI client from Secrets or Environment."""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        api_key = os.getenv("OPENAI_API_KEY")
        
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

def encode_image(image_file):
    """Converts uploaded image file to Base64 string for the API."""
    if image_file is None: return None
    return base64.b64encode(image_file.read()).decode('utf-8')

def audit_financials_with_vision(pnl_image_b64, segment_image_b64):
    """
    The Brain: Sends images to GPT-4o-mini to extract and reconcile data.
    """
    client = get_openai_client()
    if client is None:
        return None, None, None, "Error: OpenAI API Key not found."

    system_prompt = """
    You are an expert Financial Auditor. You do not just extract numbers; you RECONCILE them.
    
    YOUR GOAL: Create a clean dataset for a 'Sankey Diagram' that maps Revenue Sources -> Gross Profit -> Net Income.

    INPUTS:
    1. IMAGE A (P&L): The Consolidated Income Statement. This is the SOURCE OF TRUTH for Total Revenue, Total Expenses, and Net Income.
    2. IMAGE B (Segments): The Revenue Breakdown by Product or Geography.

    --- AUDIT ALGORITHM ---

    STEP 1: ESTABLISH TOTALS (From Image A)
    - Extract "Total Revenue" (or "Net Sales"). Let's call this [TR].
      *CRITICAL:* If the report lists "Gross Revenue" (with Excise/GST) and "Net Revenue", YOU MUST USE "NET REVENUE".
    - Identify the Business Model to find Direct Costs (COGS):
      * Retail/Mfg (e.g. Reliance): Sum of "Cost of Materials", "Purchase of Stock", "Changes in Inventories", "Excise Duty".
      * Tech (e.g. Google): "Cost of Revenues" (TAC, Data Centers).
    - Extract Operating Expenses (R&D, S&M, G&A, Depreciation) and Tax.

    STEP 2: RECONCILE REVENUE (From Image B)
    - Extract revenue segments (e.g. "Digital", "Retail"). Sum them up = [Sum_Seg].
    - Compare [Sum_Seg] vs [TR]:
      * MATCH: If they are close (<5% diff), use the segments.
      * GAP: If [Sum_Seg] < [TR], add a segment "Unallocated Revenue" = [TR] - [Sum_Seg].
      * OVERFLOW: If [Sum_Seg] > [TR] (usually due to Inter-segment sales), look for "Eliminations". If not found, proportionally scale down segments to match [TR].

    STEP 3: OUTPUT
    - Return a JSON list. 
    - Normalize category names to: "Revenue", "COGS", "R&D", "Sales & Marketing", "G&A", "Other Opex", "Tax".

    JSON FORMAT:
    {
        "company": "Company Name",
        "currency": "USD/INR/EUR",
        "audit_note": "Short explanation of how you reconciled the revenue.",
        "lines": [
            {"item": "Segment A", "amount": 100, "category": "Revenue"},
            {"item": "Cost of Goods", "amount": 60, "category": "COGS"},
            {"item": "Tax", "amount": 10, "category": "Tax"}
        ]
    }
    """

    # Construct the visual payload
    user_content = [{"type": "text", "text": "Perform the financial audit on these images."}]
    
    if pnl_image_b64:
        user_content.append({"type": "text", "text": "IMAGE A: Master P&L (Source of Truth)"})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pnl_image_b64}"}})
    
    if segment_image_b64:
        user_content.append({"type": "text", "text": "IMAGE B: Segment Breakdown (Detail)"})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{segment_image_b64}"}})

    # API Call
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content)
        
        # Post-processing to DataFrame
        df = pd.DataFrame(data.get("lines", []))
        if "category" not in df.columns: df["category"] = "Other Opex"
        df = df.rename(columns={"item": "Item", "amount": "Amount", "category": "Category"})
        
        return df, data.get("company"), data.get("currency"), data.get("audit_note")
        
    except Exception as e:
        return None, None, None, str(e)

# -------------------------------------------------------------------
# 2. Frontend UI: The "Place for Screenshots"
# -------------------------------------------------------------------

st.title("Financial Flow Auditor 🕵️‍♂️")
st.markdown("""
**How it works:**
1. Upload the **Income Statement** (so we get the correct Profit & Margins).
2. Upload the **Segment Breakdown** (so we know where the money comes from).
3. The AI reconciles them into one clean diagram.
""")

# --- THE TWO-COLUMN UPLOADER ---
col_input1, col_input2 = st.columns(2)

with col_input1:
    st.subheader("1. Master P&L")
    st.caption("Upload the Consolidated Income Statement.")
    pnl_file = st.file_uploader("Drop P&L Screenshot", type=["png", "jpg", "jpeg"], key="pnl")
    if pnl_file:
        st.image(pnl_file, use_container_width=True)

with col_input2:
    st.subheader("2. Revenue Splits")
    st.caption("Upload the Revenue by Segment/Product table.")
    seg_file = st.file_uploader("Drop Segment Screenshot", type=["png", "jpg", "jpeg"], key="seg")
    if seg_file:
        st.image(seg_file, use_container_width=True)

# --- ACTION BUTTON ---
if pnl_file:
    if st.button("Audit & Visualize", type="primary", use_container_width=True):
        with st.spinner("AI Auditor is analyzing the images..."):
            # 1. Encode
            pnl_b64 = encode_image(pnl_file)
            seg_b64 = encode_image(seg_file) if seg_file else None
            
            # 2. Analyze
            df_result, company, currency, note = audit_financials_with_vision(pnl_b64, seg_b64)
            
            if df_result is not None:
                st.session_state.raw_df = df_result
                st.session_state.company_info = f"{company} ({currency})"
                st.session_state.audit_note = note
                st.success("Audit Complete!")
            else:
                st.error(f"Analysis Failed: {note}")
else:
    st.info("👆 Please upload at least the Master P&L to begin.")

# -------------------------------------------------------------------
# 3. Visualization & Results
# -------------------------------------------------------------------

if "raw_df" in st.session_state:
    st.divider()
    
    # Header & Note
    st.header(f"Results: {st.session_state.company_info}")
    if st.session_state.audit_note:
        st.info(f"📝 **Auditor's Finding:** {st.session_state.audit_note}")

    # Layout: Data on Left, Chart on Right
    col_data, col_viz = st.columns([1, 2])

    with col_data:
        st.subheader("Reconciled Data")
        df = st.session_state.raw_df.copy()
        
        # Editable Table
        edited_df = st.data_editor(
            df,
            column_config={
                "Category": st.column_config.SelectboxColumn(
                    "Category",
                    options=list(CATEGORY_COLORS.keys()),
                    required=True
                ),
                "Amount": st.column_config.NumberColumn(format="%.0f")
            },
            use_container_width=True,
            num_rows="dynamic"
        )
        clean_df = edited_df.copy()

    with col_viz:
        # --- CALCULATE SANKEY FLOWS ---
        grp = clean_df.groupby("Category")["Amount"].sum()
        
        # 1. Revenue
        rev_segments = clean_df[clean_df["Category"] == "Revenue"]
        total_revenue = grp.get("Revenue", 0)
        
        # 2. Costs
        cogs = grp.get("COGS", 0)
        gross_profit = total_revenue - cogs
        
        # 3. Opex
        opex_cats = ["R&D", "Sales & Marketing", "G&A", "Other Opex"]
        total_opex = sum(grp.get(c, 0) for c in opex_cats)
        
        # 4. Profit
        operating_profit = gross_profit - total_opex
        tax = grp.get("Tax", 0)
        net_income = operating_profit - tax

        # --- METRICS ROW ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Revenue", f"{total_revenue:,.0f}")
        if total_revenue > 0:
            m2.metric("Gross Margin", f"{(gross_profit/total_revenue)*100:.1f}%")
            m3.metric("Op Margin", f"{(operating_profit/total_revenue)*100:.1f}%")
            m4.metric("Net Margin", f"{(net_income/total_revenue)*100:.1f}%")

        # --- DRAW SANKEY ---
        labels, sources, targets, values, colors = [], [], [], [], []
        label_idx = {}

        def get_idx(name):
            if name not in label_idx:
                label_idx[name] = len(labels)
                labels.append(name)
                if name in CATEGORY_COLORS: colors.append(CATEGORY_COLORS[name])
                elif name == "Total Revenue": colors.append("#000000")
                else: colors.append("rgba(180,180,180,0.5)")
            return label_idx[name]

        # Node: Segments -> Total Revenue
        for _, row in rev_segments.iterrows():
            sources.append(get_idx(row["Item"]))
            targets.append(get_idx("Total Revenue"))
            values.append(row["Amount"])

        # Node: Total Revenue -> COGS & Gross Profit
        if cogs > 0:
            sources.append(get_idx("Total Revenue")); targets.append(get_idx("COGS")); values.append(cogs)
        
        sources.append(get_idx("Total Revenue")); targets.append(get_idx("Gross Profit")); values.append(gross_profit)

        # Node: Gross Profit -> Opex & Op Profit
        for cat in opex_cats:
            amt = grp.get(cat, 0)
            if amt > 0:
                sources.append(get_idx("Gross Profit")); targets.append(get_idx(cat)); values.append(amt)
        
        sources.append(get_idx("Gross Profit")); targets.append(get_idx("Operating Profit")); values.append(operating_profit)

        # Node: Op Profit -> Tax & Net Income
        if tax > 0:
            sources.append(get_idx("Operating Profit")); targets.append(get_idx("Tax")); values.append(tax)
        
        sources.append(get_idx("Operating Profit")); targets.append(get_idx("Net Income")); values.append(net_income)

        # Render
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=20, thickness=20, line=dict(color="black", width=0.5), label=labels, color=colors),
            link=dict(source=sources, target=targets, value=values, color="rgba(200,200,200,0.3)")
        )])
        
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=600)
        st.plotly_chart(fig, use_container_width=True)
