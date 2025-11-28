# app.py
# "How X Makes Money" - Streamlit + Plotly + GPT-4o-mini
# VERSION: The "AI Auditor" (Cross-Verification Logic)

import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# PDF text extraction
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# OpenAI client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


st.set_page_config(page_title="How X Makes Money", layout="wide")

# -------------------------------------------------------------------
# 0. Global Configuration
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
    "Net Income": "#0F9D58"     # Green
}

# -------------------------------------------------------------------
# 1. Universal Page Scanner (The "Metal Detector")
# -------------------------------------------------------------------
def extract_text_from_uploaded_file(uploaded_file) -> str:
    """
    Scans the ENTIRE PDF and selects pages based on a 'Financial Density Score'.
    This ensures we find the P&L table whether it's on page 5 or page 100.
    """
    if uploaded_file is None: return ""
    
    if uploaded_file.name.lower().endswith(".pdf"):
        if PdfReader is None:
            st.error("Pypdf is not installed.")
            return ""
        try:
            reader = PdfReader(uploaded_file)
            num_pages = len(reader.pages)
            page_scores = [] 

            # SCORING RUBRIC
            tier_1_keywords = [
                "consolidated statement of profit", "consolidated statement of income", 
                "consolidated statement of operations", "consolidated statement of earnings"
            ]
            tier_2_keywords = [
                "segment information", "revenue by segment", "disaggregated revenue",
                "revenue from operations", "net sales", "cost of materials", "cost of revenues"
            ]
            # Penalize Standalone to avoid the Reliance error
            negative_keywords = ["standalone", "separate financial statements"]

            for i in range(num_pages):
                try:
                    text = reader.pages[i].extract_text()
                    if not text: continue
                    low_text = text.lower()
                    score = 0
                    
                    # 1. Big boost for the Main P&L Table
                    for kw in tier_1_keywords:
                        if kw in low_text: score += 20
                    
                    # 2. Boost for Segment Data (Revenue breakdown)
                    for kw in tier_2_keywords:
                        if kw in low_text: score += 5

                    # 3. Penalize Standalone
                    for kw in negative_keywords:
                        if kw in low_text: score -= 15
                        
                    # 4. Context Boost (Look for actual data columns)
                    if "year ended" in low_text and ("in millions" in low_text or "in crores" in low_text):
                        score += 2

                    page_scores.append((i, score))
                except: continue
            
            # Select Top 8 Scoring Pages to ensure we catch Segment info which might be in Notes
            top_pages = sorted(page_scores, key=lambda x: x[1], reverse=True)[:8]
            top_indices = [p[0] for p in top_pages]
            
            # Add neighbors (tables often span 2 pages)
            final_indices = set(top_indices)
            for idx in top_indices:
                if idx + 1 < num_pages: final_indices.add(idx + 1)
            
            sorted_indices = sorted(list(final_indices))
            
            extracted_text = ""
            for i in sorted_indices:
                extracted_text += f"--- PAGE {i+1} ---\n{reader.pages[i].extract_text()}\n\n"
                
            return extracted_text

        except Exception as e:
            st.error(f"PDF Error: {e}")
            return ""
            
    # Fallback for TXT files
    try:
        return uploaded_file.read().decode("utf-8", errors="ignore")[:100000]
    except:
        return ""

# -------------------------------------------------------------------
# 2. "AI Auditor" Logic
# -------------------------------------------------------------------

def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

def extract_pnl_with_llm(raw_text: str):
    """
    Uses Chain-of-Thought reasoning to Cross-Verify Revenue and correctly identify COGS.
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client is not configured.")

    system_prompt = """
    You are an expert Financial Auditor. You do not just extract data; you VERIFY it.
    
    YOUR MISSION: Build a 'Sankey Diagram' dataset for the company's most recent CONSOLIDATED year.

    --- STEP 1: THE REVENUE AUDIT ---
    1. **Find the Net Revenue:** Look for "Revenue from Operations" or "Net Sales". 
       - DANGER: In Indian/Asian reports, do NOT use "Value of Sales" or "Gross Revenue" if they include GST/Excise duties. Use the NET figure.
    2. **Find the Segments:** Look for "Segment Revenue" (e.g. Retail, Digital, O2C, Cloud).
    3. **The Check:** Does Sum(Segments) ≈ Net Revenue?
       - If YES: Use the Segments as your Revenue sources.
       - If NO: Prioritize the "Net Revenue" total from the P&L as the source of truth. Use "Other Revenue" to balance it if needed.

    --- STEP 2: THE COST IDENTIFICATION (The Logic Filter) ---
    Determine the Business Model to find the correct "Cost of Revenue" (COGS):
    - **Type A: Manufacturing/Retail (e.g., Reliance, Walmart)**
      - COGS = "Cost of Materials Consumed" + "Purchases of Stock-in-Trade" + "Changes in Inventories" + "Excise Duty".
      - YOU MUST SUM THESE UP (or list them individually) as 'COGS'. Do NOT treat them as Operating Expenses.
    - **Type B: Tech/Services (e.g., Google, Meta)**
      - COGS = "Cost of Revenues" (often includes TAC, Data Center costs).
    - **Type C: Banking**
      - COGS = "Interest Expended".

    --- STEP 3: SANITY CHECK ---
    - Calculate: (Net Revenue - COGS) / Net Revenue.
    - **Rule:** If a Retailer/Manufacturer has > 50% Gross Margin, YOU MISSED COSTS. Look harder for "Materials" or "Purchases".
    
    --- STEP 4: OPERATING EXPENSES & TAX ---
    - Extract R&D, Sales & Marketing, G&A, Depreciation, Finance Costs.
    - Extract Tax Expense.

    OUTPUT JSON FORMAT:
    {
        "company": "Company Name",
        "currency": "Currency",
        "lines": [
            {"item": "Retail Segment", "amount": 200000, "category": "Revenue"},
            {"item": "Cost of Materials", "amount": 150000, "category": "COGS"},
            {"item": "R&D", "amount": 5000, "category": "R&D"},
            {"item": "Income Tax", "amount": 2000, "category": "Tax"}
        ]
    }
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    data = json.loads(response.choices[0].message.content)
    
    df = pd.DataFrame(data.get("lines", []))
    if "category" not in df.columns: df["category"] = "Other Opex"
    df = df.rename(columns={"item": "Item", "amount": "Amount", "category": "Category"})
    
    return df, data.get("company"), data.get("currency")

# -------------------------------------------------------------------
# 3. Main App Logic
# -------------------------------------------------------------------

st.sidebar.header("Settings")
brand_color = st.sidebar.color_picker("Brand Color", "#4285F4")
min_share = st.sidebar.slider("Min Revenue Share", 0.0, 0.2, 0.05, 0.01)

st.title("How X Makes Money (The Auditor)")

col1, col2 = st.columns([1, 2])
with col1:
    input_mode = st.radio("Input Mode", ["Upload File", "Paste Text", "Use Sample Data"])

raw_text = ""
if input_mode == "Upload File":
    f = st.file_uploader("Upload 10-K / Annual Report (PDF)", type=["pdf", "txt"])
    if f: raw_text = extract_text_from_uploaded_file(f)
elif input_mode == "Paste Text":
    raw_text = st.text_area("Paste Income Statement text")
elif input_mode == "Use Sample Data":
    st.session_state.raw_df = pd.DataFrame({
        "Item": ["Search", "YouTube", "Cloud", "TAC", "R&D", "S&M", "Tax"],
        "Amount": [50000, 8000, 6000, 12000, 9000, 4000, 3000],
        "Category": ["Revenue", "Revenue", "Revenue", "COGS", "R&D", "Sales & Marketing", "Tax"]
    })

if (input_mode != "Use Sample Data") and raw_text:
    if st.button("Analyze with AI"):
        with st.spinner("Auditing financial statements..."):
            try:
                df_result, company, currency = extract_pnl_with_llm(raw_text)
                st.session_state.raw_df = df_result
                if company: st.success(f"Identified: {company} ({currency})")
            except Exception as e:
                st.error(f"AI Error: {e}")

# -------------------------------------------------------------------
# 4. Visualization
# -------------------------------------------------------------------

if "raw_df" in st.session_state and st.session_state.raw_df is not None:
    df = st.session_state.raw_df.copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
    
    # Sort by Amount descending for cleaner table
    df = df.sort_values(by="Amount", ascending=False)

    st.subheader("1. Data Audit")
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
    
    # Logic to build the Sankey
    grp = clean_df.groupby("Category")["Amount"].sum()
    
    total_revenue = grp.get("Revenue", 0)
    # COGS Flow: Sum of all items tagged COGS
    total_cogs = grp.get("COGS", 0)
    gross_profit = total_revenue - total_cogs
    
    opex_cats = ["R&D", "Sales & Marketing", "G&A", "Other Opex"]
    total_opex = sum(grp.get(c, 0) for c in opex_cats)
    operating_profit = gross_profit - total_opex
    tax = grp.get("Tax", 0)
    net_income = operating_profit - tax

    # Metrics
    k1, k2, k3, k4 = st.columns(4)
    if total_revenue > 0:
        k1.metric("Revenue", f"{total_revenue:,.0f}")
        k2.metric("Gross Margin", f"{(gross_profit/total_revenue)*100:.1f}%")
        k3.metric("Op Margin", f"{(operating_profit/total_revenue)*100:.1f}%")
        k4.metric("Net Margin", f"{(net_income/total_revenue)*100:.1f}%")

    if st.button("Generate Sankey Diagram"):
        labels = []
        sources = []
        targets = []
        values = []
        colors = []
        
        label_idx = {}
        def get_idx(name):
            if name not in label_idx:
                label_idx[name] = len(labels)
                labels.append(name)
                if name in CATEGORY_COLORS: colors.append(CATEGORY_COLORS[name])
                elif name == "Total Revenue": colors.append(brand_color)
                else: colors.append("rgba(180,180,180,0.5)")
            return label_idx[name]

        # Flow 1: Segments -> Revenue
        for _, row in clean_df[clean_df["Category"] == "Revenue"].iterrows():
            if total_revenue > 0 and (row["Amount"] < (total_revenue * min_share)):
                s_idx = get_idx("Other Revenue")
            else:
                s_idx = get_idx(row["Item"])
            sources.append(s_idx); targets.append(get_idx("Total Revenue")); values.append(row["Amount"])

        # Flow 2: Revenue -> COGS (Split or Consolidated)
        if total_cogs > 0:
            # We can flow everything to a generic "Cost of Revenue" node to keep it clean
            sources.append(get_idx("Total Revenue")); targets.append(get_idx("Cost of Revenue")); values.append(total_cogs)
        
        sources.append(get_idx("Total Revenue")); targets.append(get_idx("Gross Profit")); values.append(gross_profit)

        # Flow 3: Gross Profit -> Opex
        for cat in opex_cats:
            amt = grp.get(cat, 0)
            if amt > 0:
                sources.append(get_idx("Gross Profit")); targets.append(get_idx(cat)); values.append(amt)
        
        sources.append(get_idx("Gross Profit")); targets.append(get_idx("Operating Profit")); values.append(operating_profit)

        # Flow 4: Operating Profit -> Net Income
        if tax > 0:
            sources.append(get_idx("Operating Profit")); targets.append(get_idx("Tax")); values.append(tax)
        
        sources.append(get_idx("Operating Profit")); targets.append(get_idx("Net Income")); values.append(net_income)

        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=20, thickness=20, line=dict(color="black", width=0.5), label=labels, color=colors),
            link=dict(source=sources, target=targets, value=values, color="rgba(200,200,200,0.3)")
        )])
        
        fig.update_layout(title_text=f"Financial Flow: {st.session_state.get('detected_company', '')}", font_size=14, height=600)
        st.plotly_chart(fig, use_container_width=True)
