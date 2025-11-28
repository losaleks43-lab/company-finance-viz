# app.py
# "How X Makes Money" - Streamlit + Plotly + GPT extraction
# VERSION: Intelligent Context-Aware Extraction + Robust Table Handling

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
# 0. Global helpers and configuration
# -------------------------------------------------------------------

CATEGORIES = [
    "Revenue",
    "COGS",
    "R&D",
    "Sales & Marketing",
    "G&A",
    "Other Opex",
    "Tax",
    "Ignore",
]


def guess_category(name: str) -> str:
    """
    Determines if a line item is Revenue, COGS, or Opex.
    Used as a fallback if context from AI is missing.
    """
    if not isinstance(name, str):
        return "Ignore"
    n = name.lower()

    # 1. COGS (Critical split)
    if any(w in n for w in ["cost of", "cogs", "benefit and claims", "interest expense", "cost of merchandise", "traffic acquisition", "tac"]):
        return "COGS"

    # 2. REVENUE
    if any(w in n for w in ["revenue", "sales", "net sales", "premium", "turnover", "receipts", "membership", "subscriptions"]):
        return "Revenue"

    # 3. TAX
    if "tax" in n:
        return "Tax"

    # 4. OPEX BUCKETS
    if "selling" in n and "general" in n: 
        return "Other Opex" # Safer for mixed bags

    if any(w in n for w in ["research", "r&d", "development", "technology"]):
        return "R&D"
    if any(w in n for w in ["marketing", "selling", "advertising", "promotion", "commercial"]):
        return "Sales & Marketing"
    if any(w in n for w in ["general", "admin", "depreciation", "amortization", "overhead"]):
        return "G&A"

    return "Other Opex"


def load_sample_df() -> pd.DataFrame:
    """Sample income statement (Google Style)."""
    data = {
        "Item": [
            "Google Search & other", "YouTube ads", "Google Network", 
            "Google Cloud", "Other Bets",
            "Cost of revenues", "R&D", "Sales & marketing", 
            "General & administrative", "Income taxes"
        ],
        "Amount": [
            44000, 7000, 8000, 
            6000, 200,
            35000, 10000, 5000, 
            3000, 4000
        ],
        "Category": [
            "Revenue", "Revenue", "Revenue",
            "Revenue", "Revenue",
            "COGS", "R&D", "Sales & Marketing",
            "G&A", "Tax"
        ]
    }
    return pd.DataFrame(data)


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])
    return df


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha) string."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(150,150,150,{alpha:.2f})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Session defaults
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "detected_company" not in st.session_state:
    st.session_state.detected_company = "Example Corp"


# -------------------------------------------------------------------
# 1. OpenAI client and AI extraction helper
# -------------------------------------------------------------------

def get_openai_client():
    """Create OpenAI client using Streamlit secrets or env var."""
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or OpenAI is None:
        return None

    return OpenAI(api_key=api_key)


def extract_pnl_with_llm(raw_text: str):
    """
    Intelligent extraction with CONTEXT PRESERVATION.
    - Items found by the Revenue Prompt are forced to 'Revenue' category.
    - Items found by the Cost Prompt are passed to a specific expense categorizer 
      (that never guesses 'Revenue').
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError(
            "OpenAI client is not configured. "
            "Install openai and set OPENAI_API_KEY in secrets or env."
        )

    def call_llm(system_prompt: str, text: str) -> dict:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    # 1. REVENUE PROMPT: STRICTER on Double Counting
    revenue_system_prompt = """
    You are an expert Financial Analyst. Extract data for the Revenue flows of a Sankey diagram.
    INPUT DATA: Text from a financial report (10-K/Annual Report).
    YOUR TASK: Extract the revenue breakdown for the most recent year available.

    CRITICAL ANTI-DOUBLE-COUNTING RULES:
    1. **Lowest Level Only:** If you have a hierarchy (e.g. "Services Total" -> "Advertising" -> "Search"), ONLY extract the lowest level ("Search", "YouTube", etc.).
    2. **EXCLUDE AGGREGATES:** Do NOT include "Google Services Total", "Total Advertising", or "Total Net Sales".
    3. **Segments:** Look for specific product lines: "Search", "YouTube", "Cloud", "Hardware", "Play".
    
    Output ONLY valid JSON: {"lines": [{"item": "Search", "amount": 12345}]}
    """.strip()

    data_rev = call_llm(revenue_system_prompt, raw_text)

    # 2. COST PROMPT
    cost_system_prompt = """
    You are an expert Financial Analyst. Extract data for the Cost & Expense flows.
    YOUR TASK: Identify the **Major Cost Drivers** for the most recent year.
    
    ADAPTIVE LOGIC RULES:
    1. **Cost of Sales:** Find the line for direct costs (COGS, Cost of Revenue, TAC).
    2. **Major Opex:** Extract top 3-5 operating expense lines exactly as written.
    3. **Tax:** Extract "Income tax expense".
    4. **NO Subtotals:** ABSOLUTELY NO "Gross Profit", "Operating Income", "Total Expenses".
    
    Output ONLY valid JSON: {"lines": [{"item": "Cost of revenues", "amount": 40000}]}
    """.strip()

    data_cost = call_llm(cost_system_prompt, raw_text)

    # 3. HELPER: Expense-Only Categorizer
    def categorize_expense_internal(name: str) -> str:
        n = name.lower()
        if any(w in n for w in ["cost of", "cogs", "tac", "traffic acquisition", "benefit and claims"]):
            return "COGS"
        if "tax" in n:
            return "Tax"
        if any(w in n for w in ["research", "r&d", "development", "technology"]):
            return "R&D"
        if any(w in n for w in ["marketing", "selling", "advertising", "promotion", "commercial"]):
            return "Sales & Marketing"
        if any(w in n for w in ["general", "admin", "depreciation", "amortization", "overhead"]):
            return "G&A"
        return "Other Opex"

    # 4. MERGE (Applying Strict Context)
    lines = []
    detected_company = data_rev.get("company")
    detected_currency = data_rev.get("currency")

    # REVENUE: Hard-code category to 'Revenue'.
    for line in data_rev.get("lines", []):
        try:
            lines.append({
                "Item": line["item"], 
                "Amount": float(line["amount"]), 
                "Category": "Revenue" 
            })
        except:
            continue

    # COSTS: Use the Expense-Only categorizer.
    for line in data_cost.get("lines", []):
        try:
            item_name = line["item"]
            amount = float(line["amount"])
            cat = categorize_expense_internal(item_name) 
            lines.append({
                "Item": item_name, 
                "Amount": amount, 
                "Category": cat
            })
        except:
            continue

    df = pd.DataFrame(lines)
    return df, detected_company, detected_currency


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """
    Turn uploaded PDF/TXT into plain text for the LLM.
    Includes smart filtering to find Revenue Segment tables.
    """
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()

    # TXT
    if name.endswith(".txt") or uploaded_file.type == "text/plain":
        try:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            return text[:100000]
        except Exception:
            return ""

    # PDF
    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError(
                "pypdf is not installed. Add 'pypdf' to requirements.txt."
            )

        try:
            reader = PdfReader(uploaded_file)
            all_pages_text = []
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                    all_pages_text.append(t)
                except Exception:
                    all_pages_text.append("")

            # 1. Search for Strong Keywords (Table Titles & Segments)
            strong_keywords = [
                "consolidated statements of income",
                "consolidated statement of income",
                "consolidated statements of operations",
                "consolidated statement of operations",
                "consolidated statements of earnings",
                "consolidated statement of earnings",
                "segment information",          
                "disaggregated revenue",        
                "revenue by category"           
            ]
            
            # 2. Search for Weak Keywords (Line items) if titles fail
            weak_keywords = [
                "net sales", "cost of sales", "operating income", "income tax expense"
            ]

            candidate_indices = []
            
            # Phase 1: Look for strong table titles
            for i, t in enumerate(all_pages_text):
                low = t.lower()
                if any(kw in low for kw in strong_keywords):
                    candidate_indices.append(i)
            
            # Phase 2: If no titles found, look for density of line items
            if not candidate_indices:
                for i, t in enumerate(all_pages_text):
                    low = t.lower()
                    matches = sum(1 for kw in weak_keywords if kw in low)
                    if matches >= 2: 
                        candidate_indices.append(i)

            expanded_indices = set()
            for i in candidate_indices:
                # Grab page + next page
                expanded_indices.add(i)
                if i + 1 < len(all_pages_text):
                    expanded_indices.add(i + 1)

            if expanded_indices:
                selected_pages = [all_pages_text[i] for i in sorted(expanded_indices)]
                text = "\n\n".join(selected_pages)
            else:
                text = "\n\n".join(all_pages_text[:50])

            return text[:100000]
        except Exception:
            return ""

    # Fallback
    try:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        return text[:100000]
    except Exception:
        return ""


# -------------------------------------------------------------------
# 2. Layout: branding and input controls
# -------------------------------------------------------------------

st.sidebar.header("Branding")

brand_color = st.sidebar.color_picker("Primary brand color", "#4285F4")

logo_file = st.sidebar.file_uploader(
    "Company logo (PNG/JPG, optional)",
    type=["png", "jpg", "jpeg"],
)

st.sidebar.markdown("---")
st.sidebar.header("Input data")

input_mode = st.sidebar.radio(
    "How do you want to provide data?",
    ["AI (upload/paste statement)", "Upload CSV/Excel", "Use sample data"],
)

company_name_override = st.sidebar.text_input(
    "Company name (optional override)",
    value=st.session_state.detected_company,
)

st.sidebar.markdown("---")
min_share = st.sidebar.slider(
    "Min revenue share for separate node",
    0.0, 0.20, 0.05, 0.01,
    help="Revenue items smaller than this share of total revenue "
         "are grouped into 'Other revenue'.",
)

# Top of main page
if logo_file is not None:
    st.image(logo_file, width=140)

st.title("How X Makes Money")
st.write(
    "Upload a company's financial statement and let AI extract the income statement, "
    "or upload a ready CSV/Excel. Review the line items and generate a Sankey diagram "
    "showing how the company makes and spends money."
)

# -------------------------------------------------------------------
# 3. Input modes -> st.session_state.raw_df
# -------------------------------------------------------------------

raw_df = None

if input_mode == "Use sample data":
    raw_df = load_sample_df()
    st.session_state.raw_df = raw_df
    st.session_state.detected_company = company_name_override or "Example Corp"

elif input_mode == "Upload CSV/Excel":
    uploaded_csv = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="File must have columns 'Item' and 'Amount'.",
        key="csv_uploader",
    )
    if uploaded_csv is not None:
        try:
            raw_df = pd.read_csv(uploaded_csv)
        except Exception:
            uploaded_csv.seek(0)
            raw_df = pd.read_excel(uploaded_csv)
        st.session_state.raw_df = raw_df
        st.session_state.detected_company = company_name_override or "Example Corp"

elif input_mode == "AI (upload/paste statement)":
    st.subheader("Step 0 – Provide income statement")

    uploaded_stmt = st.file_uploader(
        "Upload financial statement (PDF or TXT)",
        type=["pdf", "txt"],
        key="stmt_uploader",
    )

    raw_text_manual = st.text_area(
        "Or paste the income statement text here (income statement section is enough).",
        height=260,
        key="raw_text_area",
    )

    if st.button("Extract with AI"):
        try:
            raw_text_for_ai = extract_text_from_uploaded_file(uploaded_stmt)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            st.stop()

        if not raw_text_for_ai.strip():
            raw_text_for_ai = raw_text_manual

        if not raw_text_for_ai.strip():
            st.warning("Please upload a PDF/TXT file or paste some text first.")
            st.stop()

        with st.spinner("Calling GPT-4o-mini to extract the income statement..."):
            try:
                df_ai, detected_company, detected_currency = extract_pnl_with_llm(
                    raw_text_for_ai
                )
            except Exception as e:
                st.error(
                    "AI extraction failed. Check your API key or try a simpler snippet. "
                    f"Technical detail: {e}"
                )
                st.stop()

        if df_ai.empty:
            st.error("AI did not return any line items. Try a clearer statement.")
            st.stop()

        st.session_state.raw_df = df_ai
        if detected_company:
            st.session_state.detected_company = detected_company

        if detected_company:
            st.success(f"Detected company: {detected_company}")
        if detected_currency:
            st.caption(f"Detected currency: {detected_currency}")

# After all branches, use whatever we have in session_state
raw_df = st.session_state.raw_df

if raw_df is None:
    st.info("Provide data via AI extraction, CSV upload, or the sample option in the sidebar.")
    st.stop()

df = raw_df.copy()

# -------------------------------------------------------------------
# 4. Column validation and auto categorization
# -------------------------------------------------------------------

# Normalize columns first
df.columns = [c.lower() for c in df.columns]

if "item" not in df.columns or "amount" not in df.columns:
    st.error(
        "Your data must contain columns named 'Item' and 'Amount' "
        "(case insensitive). Found columns: " + ", ".join(df.columns)
    )
    st.stop()

# Standardize column names
df = df.rename(columns={"item": "Item", "amount": "Amount"})
if "category" in df.columns:
    df = df.rename(columns={"category": "Category"})

df = ensure_numeric(df)

if df.empty:
    st.error("No valid numeric 'Amount' values found.")
    st.stop()

# If Category column is missing (e.g. CSV upload), add it via guessing
if "Category" not in df.columns:
    df["Category"] = df["Item"].apply(guess_category)
else:
    # Fill any NaNs or empty strings in Category
    df["Category"] = df["Category"].fillna("Other Opex")
    mask = df["Category"] == ""
    if mask.any():
        df.loc[mask, "Category"] = df.loc[mask, "Item"].apply(guess_category)

st.subheader("Step 1 – Review and adjust categories")
st.write(
    "We guessed a category for each line based on its name. "
    "You can change the Category column below. Lines marked as Ignore "
    "will not appear in the visualization."
)

edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    column_config={
        "Category": st.column_config.SelectboxColumn(
            "Category",
            options=CATEGORIES,
            help="How this line item should be treated in the Sankey diagram.",
        ),
        "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
    },
    use_container_width=True,
    key="data_editor",
)

df = edited_df.copy()
df = ensure_numeric(df)
df = df[df["Category"] != "Ignore"]

if df.empty:
    st.error("All rows are marked as Ignore. Please assign some categories.")
    st.stop()

# -------------------------------------------------------------------
# 5. Aggregation and derived metrics
# -------------------------------------------------------------------

cat_sums = df.groupby("Category")["Amount"].sum().to_dict()

total_revenue = cat_sums.get("Revenue", 0.0)
cogs = cat_sums.get("COGS", 0.0)
rnd = cat_sums.get("R&D", 0.0)
sm = cat_sums.get("Sales & Marketing", 0.0)
ga = cat_sums.get("G&A", 0.0)
other_opex = cat_sums.get("Other Opex", 0.0)
tax = cat_sums.get("Tax", 0.0)

# If there are revenues but absolutely no costs/tax, something is off.
if total_revenue > 0 and (cogs + rnd + sm + ga + other_opex + tax) == 0:
    st.error(
        "The AI extraction found revenue but no cost or tax lines. "
        "This would make all margins 100 %, so the visualization would be misleading.\n\n"
        "Please either:\n"
        "• paste or upload a more complete income statement (including 'Cost of sales', expenses, tax), or\n"
        "• manually add the main cost lines in the table above and set their Category (e.g. COGS, G&A, Tax)."
    )
    st.stop()

gross_profit = max(total_revenue - cogs, 0)
total_opex = rnd + sm + ga + other_opex
operating_profit = max(gross_profit - total_opex, 0)
net_income = max(operating_profit - tax, 0)

company_name = company_name_override or st.session_state.detected_company or "This company"


# -------------------------------------------------------------------
# 6. Sankey builder with brand color and revenue grouping
# -------------------------------------------------------------------

def build_sankey(df: pd.DataFrame, primary_color: str, min_share: float):
    labels = []
    color_map = {}

    primary = primary_color
    profit_c = "#0F9D58"
    cost_c = "#DB4437"
    rnd_c = "#AB47BC"
    sm_c = "#F4B400"
    ga_c = "#00ACC1"
    other_c = "#8D6E63"

    labels.extend(
        [
            "Total revenue",
            "Cost of revenues",
            "Gross profit",
            "R&D",
            "Sales & marketing",
            "G&A",
            "Other opex",
            "Operating profit",
            "Tax",
            "Net income",
        ]
    )

    base_colors = {
        "Total revenue": primary,
        "Cost of revenues": cost_c,
        "Gross profit": hex_to_rgba(primary, 1.0),
        "R&D": rnd_c,
        "Sales & marketing": sm_c,
        "G&A": ga_c,
        "Other opex": other_c,
        "Operating profit": profit_c,
        "Tax": cost_c,
        "Net income": profit_c,
    }
    for lab, col in base_colors.items():
        color_map[lab] = col

    # Revenue rows and grouping of small ones – these are your "products"
    revenue_rows = df[df["Category"] == "Revenue"].copy()
    revenue_nodes = []

    if not revenue_rows.empty:
        total_rev_seg = revenue_rows["Amount"].sum()
        threshold = total_rev_seg * min_share

        major_rows = revenue_rows[revenue_rows["Amount"] >= threshold]
        minor_rows = revenue_rows[revenue_rows["Amount"] < threshold]

        for _, row in major_rows.iterrows():
            revenue_nodes.append((row["Item"], float(row["Amount"])))

        if not minor_rows.empty:
            other_sum = float(minor_rows["Amount"].sum())
            revenue_nodes.append(("Other revenue", other_sum))

    # Create labels for revenue nodes
    for name, _amount in revenue_nodes:
        if name not in labels:
            labels.append(name)
            color_map[name] = hex_to_rgba(primary, 0.8)

    idx = {lab: i for i, lab in enumerate(labels)}

    sources, targets, values, link_colors = [], [], [], []

    def add_link(src_label, tgt_label, value):
        if value <= 0:
            return
        s = idx[src_label]
        t = idx[tgt_label]
        sources.append(s)
        targets.append(t)
        values.append(value)
        src_color = color_map.get(src_label, "#AAAAAA")
        link_colors.append(hex_to_rgba(src_color, 0.35))

    # 1) Revenue segments ("products") -> Total revenue
    for name, amount in revenue_nodes:
        add_link(name, "Total revenue", amount)

    # 2) Total revenue -> COGS and Gross profit
    add_link("Total revenue", "Cost of revenues", cogs)
    add_link("Total revenue", "Gross profit", gross_profit)

    # 3) Gross profit -> Opex categories and Operating profit
    add_link("Gross profit", "R&D", rnd)
    add_link("Gross profit", "Sales & marketing", sm)
    add_link("Gross profit", "G&A", ga)
    add_link("Gross profit", "Other opex", other_opex)
    add_link("Gross profit", "Operating profit", operating_profit)

    # 4) Operating profit -> Tax and Net income
    add_link("Operating profit", "Tax", tax)
    add_link("Operating profit", "Net income", net_income)

    node_colors = [color_map.get(lab, "#CCCCCC") for lab in labels]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20,
                    thickness=22,
                    line=dict(color="rgba(0,0,0,0.15)", width=0.5),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"How {company_name} Makes Money",
        font=dict(size=14, family="Arial"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


# -------------------------------------------------------------------
# 7. Show metrics and visualization
# -------------------------------------------------------------------

st.subheader("Step 2 – Key figures")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total revenue", f"{total_revenue:,.0f}")
with col2:
    gross_margin = (gross_profit / total_revenue * 100) if total_revenue else 0
    st.metric("Gross margin", f"{gross_margin:.1f} %")
with col3:
    op_margin = (operating_profit / total_revenue * 100) if total_revenue else 0
    st.metric("Operating margin", f"{op_margin:.1f} %")
with col4:
    net_margin = (net_income / total_revenue * 100) if total_revenue else 0
    st.metric("Net margin", f"{net_margin:.1f} %")

st.subheader("Step 3 – Visualization")
st.write(
    "Click Generate chart after you are happy with the categories above. "
    "Hover over the flows to see exact amounts."
)

if st.button("Generate chart"):
    fig = build_sankey(df, brand_color, min_share)
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Tip: use the camera icon in the top right of the chart to download it as a PNG."
    )
else:
    st.caption("Press the button above to build the Sankey diagram.")
