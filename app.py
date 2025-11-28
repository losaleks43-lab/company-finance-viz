# app.py
# "How X Makes Money" - Streamlit + Plotly + GPT Extraction
# VERSION: AI-Driven Reasoning (Adaptive to Business Model)

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

# We map the AI's output tags to specific colors in the Sankey diagram
CATEGORY_COLORS = {
    "Revenue": "#4285F4",       # Blue
    "COGS": "#DB4437",          # Red
    "Gross Profit": "#BDBDBD",  # Grey (Calculated node)
    "R&D": "#AB47BC",           # Purple
    "Sales & Marketing": "#F4B400", # Yellow
    "G&A": "#00ACC1",           # Teal
    "Other Opex": "#8D6E63",    # Brown
    "Tax": "#E91E63",           # Pink
    "Net Income": "#0F9D58"     # Green
}

# -------------------------------------------------------------------
# 1. AI Extraction Logic (The "Brain")
# -------------------------------------------------------------------

def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

def extract_pnl_with_llm(raw_text: str):
    """
    Uses Chain-of-Thought reasoning to extract and categorize P&L data.
    The AI decides what is COGS vs Opex based on the company type.
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client is not configured.")

    system_prompt = """
    You are an expert Financial Analyst building a dataset for a Sankey diagram.

    YOUR GOAL: 
    Extract a structured Income Statement from the provided text. You must determine the correct category for each line item based on the company's business model (e.g., Retail vs. Tech vs. Bank).

    --- STEP 1: REVENUE (Left Side) ---
    - Find the **Revenue Segments** (e.g., "iPhone", "Services" or "Oil to Chemicals", "Retail").
    - IGNORE aggregate totals like "Total Revenue" or "Revenue from Operations" to avoid double counting.
    - Tag these as category: "Revenue".

    --- STEP 2: DIRECT COSTS (Middle - The "Cost of Revenue" Flow) ---
    - Identify costs that scale directly with sales. Tag these as category: "COGS".
    - **For Retail/Manufacturing (e.g., Reliance, Amazon):** This INCLUDES "Cost of materials consumed", "Purchases of stock-in-trade", "Changes in inventories", and "Excise Duty".
    - **For Tech (e.g., Google):** This INCLUDES "Traffic Acquisition Costs (TAC)" and "Data Center Costs".
    - **For Banks:** This INCLUDES "Interest Expense".

    --- STEP 3: OPERATING EXPENSES (Right Side) ---
    - Identify the major operating expense buckets.
    - Standardize tags where possible: "R&D", "Sales & Marketing", "G&A".
    - If a line is generic (e.g. "Other expenses"), tag it "Other Opex".

    --- STEP 4: TAX ---
    - Extract "Income Tax Expense" or "Current Tax". Tag as category: "Tax".

    OUTPUT FORMAT (JSON):
    {
        "company": "Company Name",
        "currency": "Currency Code",
        "lines": [
            {"item": "Name of Line Item", "amount": 12345.6, "category": "Revenue"},
            {"item": "Cost of materials", "amount": 5000.0, "category": "COGS"},
            {"item": "Selling expenses",  "amount": 1000.0, "category": "Sales & Marketing"}
        ]
    }
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Cost-effective model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    data = json.loads(response.choices[0].message.content)
    
    # Convert to DataFrame
    df = pd.DataFrame(data.get("lines", []))
    # Ensure columns exist
    if "category" not in df.columns: df["category"] = "Other Opex"
    
    # Title Case columns for the UI
    df = df.rename(columns={"item": "Item", "amount": "Amount", "category": "Category"})
    
    return df, data.get("company"), data.get("currency")


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """Extract text from PDF or TXT."""
    if uploaded_file is None: return ""
    
    # PDF Handling
    if uploaded_file.name.lower().endswith(".pdf"):
        if PdfReader is None:
            st.error("Pypdf is not installed. Please install it to read PDFs.")
            return ""
        try:
            reader = PdfReader(uploaded_file)
            # Smart filter: Look for P&L pages only to save tokens
            keywords = ["consolidated statement of profit", "income statement", "statement of operations", "segment revenue", "revenue by"]
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text() or ""
                # Only keep pages that look like financial tables
                if any(k in text.lower() for k in keywords):
                    text_parts.append(text)
            
            # If smart filter found nothing, use first 50 pages
            if text_parts:
                full_text = "\n".join(text_parts)
            else:
                full_text = "\n".join([p.extract_text() for p in reader.pages[:50]])
                
            return full_text[:100000] # Cap at 100k chars
        except Exception as e:
            st.error(f"PDF Error: {e}")
            return ""
            
    # Text Handling
    return uploaded_file.read().decode("utf-8", errors="ignore")[:100000]


# -------------------------------------------------------------------
# 2. UI & Main Logic
# -------------------------------------------------------------------

st.sidebar.header("Settings")
brand_color = st.sidebar.color_picker("Brand Color", "#4285F4")
min_share = st.sidebar.slider("Min Revenue Share", 0.0, 0.2, 0.05, 0.01)

st.title("How X Makes Money (AI-Powered)")

# Input Section
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
    # Dummy data for demo
    st.session_state.raw_df = pd.DataFrame({
        "Item": ["Search", "YouTube", "Cloud", "TAC", "R&D", "S&M", "Tax"],
        "Amount": [50000, 8000, 6000, 12000, 9000, 4000, 3000],
        "Category": ["Revenue", "Revenue", "Revenue", "COGS", "R&D", "Sales & Marketing", "Tax"]
    })

# Extraction Trigger
if (input_mode != "Use Sample Data") and raw_text:
    if st.button("Analyze with AI"):
        with st.spinner("AI is reading the financial report..."):
            try:
                df_result, company, currency = extract_pnl_with_llm(raw_text)
                st.session_state.raw_df = df_result
                if company: st.success(f"Identified: {company} ({currency})")
            except Exception as e:
                st.error(f"AI Error: {e}")

# -------------------------------------------------------------------
# 3. Data Review & Visualization
# -------------------------------------------------------------------

if "raw_df" in st.session_state and st.session_state.raw_df is not None:
    df = st.session_state.raw_df.copy()

    # Data Cleaning
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
    
    st.subheader("1. Data Review")
    
    # Allow user to override AI's categorization if needed
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

    # Filter out ignored rows
    clean_df = edited_df[edited_df["Category"] != "Ignore"].copy()

    # -------------------------------------------------------------------
    # 4. Sankey Logic (The "Engine")
    # -------------------------------------------------------------------
    
    # Calculate Aggregates
    grp = clean_df.groupby("Category")["Amount"].sum()
    
    total_revenue = grp.get("Revenue", 0)
    total_cogs = grp.get("COGS", 0)
    gross_profit = total_revenue - total_cogs
    
    opex_cats = ["R&D", "Sales & Marketing", "G&A", "Other Opex"]
    total_opex = sum(grp.get(c, 0) for c in opex_cats)
    
    operating_profit = gross_profit - total_opex
    tax = grp.get("Tax", 0)
    net_income = operating_profit - tax

    # Key Metrics Display
    k1, k2, k3, k4 = st.columns(4)
    if total_revenue > 0:
        k1.metric("Revenue", f"{total_revenue:,.0f}")
        k2.metric("Gross Margin", f"{(gross_profit/total_revenue)*100:.1f}%")
        k3.metric("Op Margin", f"{(operating_profit/total_revenue)*100:.1f}%")
        k4.metric("Net Margin", f"{(net_income/total_revenue)*100:.1f}%")
    else:
        st.warning("Total Revenue is 0. Please check the data table above.")

    # -------------------------------------------------------------------
    # 5. Build the Diagram
    # -------------------------------------------------------------------
    
    if st.button("Generate Sankey Diagram"):
        labels = []
        sources = []
        targets = []
        values = []
        colors = []
        
        # Helper to manage node indices
        label_idx = {}
        def get_idx(name):
            if name not in label_idx:
                label_idx[name] = len(labels)
                labels.append(name)
                # Assign color based on known categories or default to grey
                if name in CATEGORY_COLORS:
                    colors.append(CATEGORY_COLORS[name])
                elif name == "Total Revenue":
                    colors.append(brand_color)
                else:
                    colors.append("rgba(180,180,180,0.5)")
            return label_idx[name]

        # --- FLOW 1: Segment Revenue -> Total Revenue ---
        rev_df = clean_df[clean_df["Category"] == "Revenue"]
        # Group small segments
        for _, row in rev_df.iterrows():
            if total_revenue > 0 and (row["Amount"] < (total_revenue * min_share)):
                s_idx = get_idx("Other Revenue")
            else:
                s_idx = get_idx(row["Item"])
            
            t_idx = get_idx("Total Revenue")
            sources.append(s_idx)
            targets.append(t_idx)
            values.append(row["Amount"])

        # --- FLOW 2: Total Revenue -> COGS & Gross Profit ---
        if total_cogs > 0:
            sources.append(get_idx("Total Revenue"))
            targets.append(get_idx("COGS")) # Display as "COGS" or "Cost of Revenue"
            values.append(total_cogs)
            
        sources.append(get_idx("Total Revenue"))
        targets.append(get_idx("Gross Profit"))
        values.append(gross_profit)

        # --- FLOW 3: Gross Profit -> Opex & Operating Profit ---
        for cat in opex_cats:
            amt = grp.get(cat, 0)
            if amt > 0:
                sources.append(get_idx("Gross Profit"))
                targets.append(get_idx(cat))
                values.append(amt)
        
        sources.append(get_idx("Gross Profit"))
        targets.append(get_idx("Operating Profit")) # Intermediate node
        values.append(operating_profit)

        # --- FLOW 4: Operating Profit -> Tax & Net Income ---
        if tax > 0:
            sources.append(get_idx("Operating Profit"))
            targets.append(get_idx("Tax"))
            values.append(tax)
        
        sources.append(get_idx("Operating Profit"))
        targets.append(get_idx("Net Income"))
        values.append(net_income)

        # Render Plotly Chart
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(200,200,200,0.3)"
            )
        )])
        
        fig.update_layout(title_text="Financial Flow", font_size=14, height=600)
        st.plotly_chart(fig, use_container_width=True)
