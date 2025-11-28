# app.py
# "How X Makes Money" - Screenshot Auditor Edition
# VERSION: Multi-Image Reconciliation (P&L + Segments)

import os
import json
import base64
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Try OpenAI import
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

st.set_page_config(page_title="Financial Flow Auditor", layout="wide")

# -------------------------------------------------------------------
# 1. Configuration & Styles
# -------------------------------------------------------------------
CATEGORY_COLORS = {
    "Revenue": "#4285F4",       # Blue
    "COGS": "#DB4437",          # Red
    "Gross Profit": "#BDBDBD",  # Grey
    "R&D": "#AB47BC",           # Purple
    "Sales & Marketing": "#F4B400", # Yellow
    "G&A": "#00ACC1",           # Teal
    "Other Opex": "#8D6E63",    # Brown
    "Tax": "#E91E63",           # Pink
    "Net Income": "#0F9D58",    # Green
    "Eliminations": "#5f6368"   # Dark Grey
}

# -------------------------------------------------------------------
# 2. Logic: The AI Auditor
# -------------------------------------------------------------------

def get_openai_client():
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        api_key = os.getenv("OPENAI_API_KEY")
        
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

def encode_image(image_file):
    """Encodes uploaded image to Base64 for the API."""
    if image_file is None: return None
    return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_dual_screenshots(pnl_image_b64, segment_image_b64):
    """
    Sends TWO images to the AI:
    1. The Master P&L (Source of Truth for Totals)
    2. The Segment Breakdown (Source of Detail)
    The AI reconciles them.
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not configured.")

    system_prompt = """
    You are an expert Financial Auditor performing a reconciliation.
    
    INPUTS:
    1. Image A: The Consolidated Income Statement (P&L). This is the SOURCE OF TRUTH for Total Revenue and Net Income.
    2. Image B: The Segment Revenue Breakdown (Product/Geo cuts).

    YOUR ALGORITHM:
    
    --- STEP 1: ESTABLISH TOTALS (From Image A) ---
    - Extract the "Total Revenue" (or "Net Sales") from Image A. Let's call this [TR].
    - Extract "Cost of Sales" (COGS), "Operating Expenses", and "Tax" from Image A.
    
    --- STEP 2: EXTRACT SEGMENTS (From Image B) ---
    - Extract revenue segments (e.g., "Cloud", "Retail"). Sum them up. Let's call this [Sum_Seg].
    
    --- STEP 3: RECONCILIATION LOGIC (Crucial!) ---
    - Compare [Sum_Seg] vs [TR].
    - **Scenario A (Match):** If [Sum_Seg] ≈ [TR] (within 5%), use the segments as the Revenue sources.
    - **Scenario B (Gap):** If [Sum_Seg] < [TR], create a new segment called "Other/Unallocated" equal to ([TR] - [Sum_Seg]).
    - **Scenario C (Double Count):** If [Sum_Seg] > [TR] (significantly), look for an "Eliminations" or "Inter-segment" line in Image B. If not found, assume Image B is Gross and Image A is Net; scale the segments down proportionally so they match [TR].
    
    --- STEP 4: COSTS & MARGINS ---
    - Map the costs from Image A to the flows.
    - **Reliance Specific Check:** If you see "Cost of Materials" + "Purchases of Stock" + "Inventory Changes", sum them as 'COGS'.
    
    OUTPUT JSON:
    {
        "company": "Name",
        "currency": "Symbol",
        "total_revenue_audit": 100000, 
        "lines": [
            {"item": "Segment A", "amount": 60000, "category": "Revenue"},
            {"item": "Segment B", "amount": 40000, "category": "Revenue"},
            {"item": "Cost of Materials", "amount": 50000, "category": "COGS"}
        ]
    }
    """

    # Build content payload
    content = [{"type": "text", "text": "Reconcile these two financial reports."}]
    
    if pnl_image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pnl_image_b64}", "detail": "high"}})
        content.append({"type": "text", "text": "IMAGE A: Consolidated Income Statement (P&L)"})
    
    if segment_image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{segment_image_b64}", "detail": "high"}})
        content.append({"type": "text", "text": "IMAGE B: Segment / Product Breakdown"})

    response = client.chat.completions.create(
        model="gpt-4o-mini", # Vision capable and cheap
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    data = json.loads(response.choices[0].message.content)
    df = pd.DataFrame(data.get("lines", []))
    
    # Normalize columns
    if "category" not in df.columns: df["category"] = "Other Opex"
    df = df.rename(columns={"item": "Item", "amount": "Amount", "category": "Category"})
    
    return df, data.get("company"), data.get("currency"), data.get("total_revenue_audit")

# -------------------------------------------------------------------
# 3. UI & Application
# -------------------------------------------------------------------

st.title("Financial Flow Auditor (Screenshot-Based)")
st.markdown("Upload screenshots of the **Income Statement** and **Segment Breakdown** to auto-reconcile the data.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Master P&L")
    st.info("Upload the main Consolidated Income Statement here.")
    pnl_file = st.file_uploader("Upload P&L", type=["png", "jpg", "jpeg"], key="pnl")
    if pnl_file:
        st.image(pnl_file, caption="Source of Truth (Totals)", use_container_width=True)

with col2:
    st.subheader("2. Segment Splits")
    st.info("Upload the Revenue by Product/Geography table here.")
    seg_file = st.file_uploader("Upload Breakdown", type=["png", "jpg", "jpeg"], key="seg")
    if seg_file:
        st.image(seg_file, caption="Source of Detail (Segments)", use_container_width=True)

# Run Analysis
if pnl_file and seg_file:
    if st.button("Audit & Generate Diagram", type="primary"):
        with st.spinner("AI is reconciling the two reports..."):
            try:
                # Encode
                pnl_b64 = encode_image(pnl_file)
                seg_b64 = encode_image(seg_file)
                
                # Analyze
                df_result, company, currency, audit_rev = analyze_dual_screenshots(pnl_b64, seg_b64)
                
                st.session_state.raw_df = df_result
                st.session_state.company_info = f"{company} ({currency})"
                st.session_state.audit_rev = audit_rev
                
                st.success("Reconciliation Complete!")
                
            except Exception as e:
                st.error(f"Error: {e}")

# -------------------------------------------------------------------
# 4. Visualization & Review
# -------------------------------------------------------------------

if "raw_df" in st.session_state:
    st.divider()
    df = st.session_state.raw_df.copy()
    
    # --- Metrics Check ---
    st.subheader(f"Financials: {st.session_state.company_info}")
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.caption("Edit the reconciled data if needed:")
        edited_df = st.data_editor(
            df,
            column_config={
                "Category": st.column_config.SelectboxColumn(
                    "Category",
                    options=list(CATEGORY_COLORS.keys()),
                    required=True
                )
            },
            use_container_width=True,
            num_rows="dynamic"
        )
        clean_df = edited_df.copy()
    
    with col_b:
        # Calculate Flows
        grp = clean_df.groupby("Category")["Amount"].sum()
        
        rev_segments = clean_df[clean_df["Category"] == "Revenue"]["Amount"].sum()
        reported_total = st.session_state.audit_rev or rev_segments
        
        # Delta Check
        delta = rev_segments - reported_total
        if abs(delta) > (reported_total * 0.01):
            st.warning(f"⚠️ Segment Sum ({rev_segments:,.0f}) differs from P&L Total ({reported_total:,.0f}) by {delta:,.0f}.")
        else:
            st.success("✅ Revenue Reconciled")
            
        # Margins
        cogs = grp.get("COGS", 0)
        gross_profit = rev_segments - cogs
        net_income = gross_profit - sum(grp.get(c, 0) for c in ["R&D", "Sales & Marketing", "G&A", "Other Opex"]) - grp.get("Tax", 0)
        
        st.metric("Total Revenue", f"{rev_segments:,.0f}")
        st.metric("Gross Margin", f"{(gross_profit/rev_segments)*100:.1f}%")
        st.metric("Net Margin", f"{(net_income/rev_segments)*100:.1f}%")

    # --- Sankey Diagram ---
    st.divider()
    if st.button("Draw Chart"):
        labels, sources, targets, values, colors = [], [], [], [], []
        label_idx = {}

        def get_idx(name):
            if name not in label_idx:
                label_idx[name] = len(labels)
                labels.append(name)
                if name in CATEGORY_COLORS: colors.append(CATEGORY_COLORS[name])
                elif name == "Total Revenue": colors.append("#000000") # Black for central node
                else: colors.append("rgba(180,180,180,0.5)")
            return label_idx[name]

        # 1. Segments -> Total Revenue
        for _, row in clean_df[clean_df["Category"] == "Revenue"].iterrows():
            sources.append(get_idx(row["Item"]))
            targets.append(get_idx("Total Revenue"))
            values.append(row["Amount"])

        # 2. Total Revenue -> COGS & Gross Profit
        if cogs > 0:
            sources.append(get_idx("Total Revenue")); targets.append(get_idx("COGS")); values.append(cogs)
        
        sources.append(get_idx("Total Revenue")); targets.append(get_idx("Gross Profit")); values.append(gross_profit)

        # 3. Gross Profit -> Opex
        opex_cats = ["R&D", "Sales & Marketing", "G&A", "Other Opex"]
        total_opex = sum(grp.get(c, 0) for c in opex_cats)
        
        for cat in opex_cats:
            amt = grp.get(cat, 0)
            if amt > 0:
                sources.append(get_idx("Gross Profit")); targets.append(get_idx(cat)); values.append(amt)
        
        operating_profit = gross_profit - total_opex
        sources.append(get_idx("Gross Profit")); targets.append(get_idx("Operating Profit")); values.append(operating_profit)

        # 4. Operating Profit -> Tax & Net Income
        tax = grp.get("Tax", 0)
        if tax > 0:
            sources.append(get_idx("Operating Profit")); targets.append(get_idx("Tax")); values.append(tax)
        
        sources.append(get_idx("Operating Profit")); targets.append(get_idx("Net Income")); values.append(net_income)

        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=20, thickness=20, line=dict(color="black", width=0.5), label=labels, color=colors),
            link=dict(source=sources, target=targets, value=values, color="rgba(200,200,200,0.3)")
        )])
        
        fig.update_layout(title_text=f"Financial Flow: {st.session_state.company_info}", font_size=14, height=600)
        st.plotly_chart(fig, use_container_width=True)
