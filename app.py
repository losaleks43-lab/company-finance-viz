# app.py
# "How X Makes Money" - The AI Auditor Edition
# VERSION: Multi-Screenshot Reconciliation

import os
import json
import base64
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# OpenAI Client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

st.set_page_config(page_title="Financial Flow Auditor", layout="wide")

# -------------------------------------------------------------------
# 0. Configuration
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
    "Unallocated": "#9E9E9E"    # Neutral for reconciliation gaps
}

# -------------------------------------------------------------------
# 1. Logic & AI Functions
# -------------------------------------------------------------------

def get_openai_client():
    # Try secrets first, then env
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

def audit_financials_with_vision(pnl_image_b64, segment_image_b64):
    """
    The Core Logic: Sends both images to GPT-4o-mini with a strict Audit Prompt.
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not configured.")

    system_prompt = """
    You are an expert Financial Auditor. You do not just extract data; you RECONCILE it.
    
    INPUTS:
    1. Image A: Consolidated Income Statement (P&L). This is the SOURCE OF TRUTH for Total Revenue, Costs, and Net Income.
    2. Image B: Revenue Breakdown (Segments/Geography). This provides the detail for the Revenue side.

    YOUR ALGORITHM:

    --- STEP 1: ESTABLISH THE TOTALS (From Image A) ---
    1. Identify the reporting period and currency.
    2. Extract "Net Revenue" (or "Revenue from Operations"). Let's call this [TR].
       - DANGER: Ignore "Gross Revenue" if it includes taxes like GST/Excise. Use the NET figure.
    3. Extract the Cost Structure:
       - If Retail/Mfg: Sum "Cost of Materials", "Purchase of Stock", "Inventory Changes", "Excise Duty" -> Tag as "COGS".
       - If Tech: Extract "Cost of Revenue" -> Tag as "COGS".
    4. Extract Operating Expenses (R&D, SG&A, etc.) and Tax.

    --- STEP 2: EXTRACT & RECONCILE REVENUE (From Image B) ---
    1. Extract the segment/product revenue items. Sum them up. Call this [Sum_Seg].
    2. **Reconciliation Check:**
       - **Case A (Perfect Match):** [Sum_Seg] ≈ [TR]. Use segments as is.
       - **Case B (Gap):** [Sum_Seg] < [TR]. Create a new item "Unallocated Revenue" = [TR] - [Sum_Seg].
       - **Case C (Double Counting):** [Sum_Seg] > [TR] (e.g. Inter-segment sales included). 
         - Action: Look for "Eliminations" in Image B. If found, include it as a negative revenue item.
         - Action: If no eliminations found, Scale down all segments proportionally so they equal [TR].

    --- STEP 3: OUTPUT ---
    Generate a single JSON list representing the flows.

    OUTPUT JSON FORMAT:
    {
        "company": "Company Name",
        "currency": "Currency Symbol",
        "audit_note": "Revenue matched perfectly" or "Scaled down segments by 10% to match P&L",
        "lines": [
            {"item": "Segment A", "amount": 60000, "category": "Revenue"},
            {"item": "Segment B", "amount": 40000, "category": "Revenue"},
            {"item": "Cost of Materials", "amount": 50000, "category": "COGS"},
            {"item": "Tax Expense", "amount": 10000, "category": "Tax"}
        ]
    }
    """

    # Build content payload
    content = [{"type": "text", "text": "Perform the financial audit on these documents."}]
    
    if pnl_image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pnl_image_b64}"}})
        content.append({"type": "text", "text": "IMAGE A: Master P&L (Source of Truth)"})
    
    if segment_image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{segment_image_b64}"}})
        content.append({"type": "text", "text": "IMAGE B: Revenue Segmentation"})

    response = client.chat.completions.create(
        model="gpt-4o-mini", 
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
    
    return df, data.get("company"), data.get("currency"), data.get("audit_note")

# -------------------------------------------------------------------
# 3. UI Layout
# -------------------------------------------------------------------

st.title("Financial Flow Auditor 🕵️‍♂️")
st.markdown("""
**Instructions:**
1. Take a screenshot of the **Consolidated Income Statement** (P&L).
2. (Optional) Take a screenshot of the **Revenue by Segment/Product** table.
3. The AI will reconcile the two and build the flow.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Master P&L (Required)")
    pnl_file = st.file_uploader("Upload P&L Screenshot", type=["png", "jpg", "jpeg"], key="pnl")
    if pnl_file:
        st.image(pnl_file, caption="P&L Preview", use_container_width=True)

with col2:
    st.subheader("2. Revenue Splits (Optional)")
    seg_file = st.file_uploader("Upload Segment Screenshot", type=["png", "jpg", "jpeg"], key="seg")
    if seg_file:
        st.image(seg_file, caption="Segments Preview", use_container_width=True)

# Analysis Trigger
if pnl_file:
    if st.button("Audit & Visualize", type="primary"):
        with st.spinner("AI Auditor is reconciling the numbers..."):
            try:
                # Encode images
                pnl_b64 = encode_image(pnl_file)
                seg_b64 = encode_image(seg_file) if seg_file else None
                
                # Run Analysis
                df_result, company, currency, note = audit_financials_with_vision(pnl_b64, seg_b64)
                
                # Store in Session
                st.session_state.raw_df = df_result
                st.session_state.company_info = f"{company} ({currency})"
                st.session_state.audit_note = note
                
            except Exception as e:
                st.error(f"Analysis Failed: {e}")

# -------------------------------------------------------------------
# 4. Results & Visualization
# -------------------------------------------------------------------

if "raw_df" in st.session_state:
    st.divider()
    df = st.session_state.raw_df.copy()
    
    # --- Header Info ---
    st.subheader(f"Results for {st.session_state.company_info}")
    if st.session_state.audit_note:
        st.info(f"📝 **Auditor Note:** {st.session_state.audit_note}")

    col_data, col_viz = st.columns([1, 2])

    # --- Data Editor ---
    with col_data:
        st.markdown("### Review Data")
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

    # --- Sankey Logic ---
    with col_viz:
        # Calculate Aggregates
        grp = clean_df.groupby("Category")["Amount"].sum()
        
        # Revenue Side
        rev_segments = clean_df[clean_df["Category"] == "Revenue"]
        total_revenue = grp.get("Revenue", 0)
        
        # Cost Side
        total_cogs = grp.get("COGS", 0)
        gross_profit = total_revenue - total_cogs
        
        opex_cats = ["R&D", "Sales & Marketing", "G&A", "Other Opex"]
        total_opex = sum(grp.get(c, 0) for c in opex_cats)
        
        operating_profit = gross_profit - total_opex
        tax = grp.get("Tax", 0)
        net_income = operating_profit - tax

        # Display Key Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Revenue", f"{total_revenue:,.0f}")
        m2.metric("Gross Margin", f"{(gross_profit/total_revenue)*100:.1f}%")
        m3.metric("Net Margin", f"{(net_income/total_revenue)*100:.1f}%")
        
        # Build Sankey
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

        # 1. Segments -> Total Revenue
        for _, row in rev_segments.iterrows():
            sources.append(get_idx(row["Item"]))
            targets.append(get_idx("Total Revenue"))
            values.append(row["Amount"])

        # 2. Total Revenue -> Costs
        if total_cogs > 0:
            sources.append(get_idx("Total Revenue")); targets.append(get_idx("COGS")); values.append(total_cogs)
        
        sources.append(get_idx("Total Revenue")); targets.append(get_idx("Gross Profit")); values.append(gross_profit)

        # 3. Gross Profit -> Opex
        for cat in opex_cats:
            amt = grp.get(cat, 0)
            if amt > 0:
                sources.append(get_idx("Gross Profit")); targets.append(get_idx(cat)); values.append(amt)
        
        sources.append(get_idx("Gross Profit")); targets.append(get_idx("Operating Profit")); values.append(operating_profit)

        # 4. Operating Profit -> Net Income
        if tax > 0:
            sources.append(get_idx("Operating Profit")); targets.append(get_idx("Tax")); values.append(tax)
        
        sources.append(get_idx("Operating Profit")); targets.append(get_idx("Net Income")); values.append(net_income)

        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=20, thickness=20, line=dict(color="black", width=0.5), label=labels, color=colors),
            link=dict(source=sources, target=targets, value=values, color="rgba(200,200,200,0.3)")
        )])
        
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
