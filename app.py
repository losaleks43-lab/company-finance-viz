# app.py
# "How X Makes Money" - Streamlit + Plotly + GPT extraction
# FIXED VERSION: Correct model name, larger text limit, improved prompts.

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
    """Keyword-based guess for a line item category."""
    if not isinstance(name, str):
        return "Ignore"
    n = name.lower()

    # Revenue
    if any(w in n for w in [
        "revenue", "revenues", "net sales", "sales",
        "subscriptions", "subscription",
        "licensing", "license", "ads", "advertising",
        "cloud", "services", "membership"
    ]):
        return "Revenue"

    # COGS / cost of revenues
    if any(w in n for w in [
        "cost of revenues", "cost of revenue",
        "cost of goods", "cost of sales", "cogs"
    ]):
        return "COGS"

    # R&D
    if any(w in n for w in ["research", "r&d", "development"]):
        return "R&D"

    # Sales & Marketing
    if any(w in n for w in [
        "selling", "sales and marketing", "sales & marketing",
        "marketing"
    ]):
        return "Sales & Marketing"

    # G&A
    if any(w in n for w in [
        "general and administrative", "g&a",
        "administrative", "admin"
    ]):
        return "G&A"

    # Tax
    if "tax" in n:
        return "Tax"

    # Fallback
    return "Other Opex"


def load_sample_df() -> pd.DataFrame:
    """Sample income statement."""
    data = {
        "Item": [
            "Search advertising revenue",
            "YouTube advertising revenue",
            "Cloud services revenue",
            "Other revenue",
            "Cost of revenues",
            "R&D",
            "Sales and marketing",
            "General and administrative",
            "Other operating expenses",
            "Income tax expense",
        ],
        "Amount": [
            72000,
            33000,
            35000,
            6000,
            80000,
            35000,
            25000,
            15000,
            4000,
            8000,
        ],
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
    Use GPT to extract an income statement from raw text.
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError(
            "OpenAI client is not configured. "
            "Install openai and set OPENAI_API_KEY in secrets or env."
        )

    # Helper for making a single JSON-returning call
    def call_llm(system_prompt: str, text: str) -> dict:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # <--- FIXED: Correct model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            response_format={"type": "json_object"}  # <--- ADDED: Enforce JSON mode
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)

    # -------- 1) Revenue-only extraction --------
    revenue_system_prompt = """
You are a meticulous financial analyst.

Task: From the provided text, extract ONLY revenue / net sales lines that
belong to the income statement.

CRITICAL RULES:
1. Use ONLY numbers that clearly appear in the text.
2. If multiple years are shown, ALWAYS use the MOST RECENT year.
3. SEGMENTS VS TOTALS:
   - If the text breaks down revenue by segment (e.g. "Walmart U.S.", "Sam's Club"), extract the SEGMENTS only.
   - DO NOT include the "Total Net Sales" or "Consolidated Revenues" line if segments are present, as this causes double counting.
   - Only include the Total line if no segment breakdown is found.

Output ONLY valid JSON:
{
  "company": "Company name or null",
  "currency": "3 letter currency code or null",
  "lines": [
    {"item": "Net sales - Segment A", "amount": 1234.56},
    {"item": "Net sales - Segment B", "amount": 2345.67}
  ]
}
""".strip()

    data_rev = call_llm(revenue_system_prompt, raw_text)

    # -------- 2) Cost / expense / tax extraction --------
    cost_system_prompt = """
You are a meticulous financial analyst.

Task: From the provided text, extract ONLY COST / EXPENSE / TAX line items
that belong to the income statement (profit and loss).

You MUST include, when present:
- Cost of sales / Cost of revenues / Cost of goods sold.
- Operating expenses (SG&A, Marketing, R&D, etc).
- Income tax expense.

CRITICAL RULES:
1. Use ONLY numbers that appear in the text.
2. If multiple years are shown, ALWAYS use the MOST RECENT year.
3. DO NOT include subtotals like "Total operating expenses", "Gross profit", "Operating income", "Net income".
4. DO NOT include revenue or net sales here.

Output ONLY valid JSON:
{
  "lines": [
    {"item": "Cost of sales", "amount": 999.99},
    {"item": "Selling, general and administrative", "amount": 888.88},
    {"item": "Income tax expense", "amount": 777.77}
  ]
}
""".strip()

    data_cost = call_llm(cost_system_prompt, raw_text)

    # -------- 3) Merge the two results --------
    lines = []

    # Company / currency from revenue call (if any)
    detected_company = data_rev.get("company")
    detected_currency = data_rev.get("currency")

    for src in (data_rev, data_cost):
        for line in src.get("lines", []):
            try:
                item = line["item"]
                amount = float(line["amount"])
                lines.append({"Item": item, "Amount": amount})
            except Exception:
                continue

    df = pd.DataFrame(lines)

    return df, detected_company, detected_currency

def extract_text_from_uploaded_file(uploaded_file) -> str:
    """
    Turn uploaded PDF/TXT into plain text for the LLM.
    """
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()

    # TXT
    if name.endswith(".txt") or uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        return text[:100000]  # <--- INCREASED LIMIT

    # PDF
    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError(
                "pypdf is not installed. Add 'pypdf' to requirements.txt."
            )

        reader = PdfReader(uploaded_file)
        all_pages_text = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
                all_pages_text.append(t)
            except Exception:
                all_pages_text.append("")

        # 1. Search for Strong Keywords (Table Titles)
        strong_keywords = [
            "consolidated statements of income",
            "consolidated statement of income",
            "consolidated statements of operations",
            "consolidated statement of operations",
            "consolidated statements of earnings",
            "consolidated statement of earnings",
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
                if matches >= 2: # At least 2 p&l terms on the page
                    candidate_indices.append(i)

        expanded_indices = set()
        for i in candidate_indices:
            # Grab page + next page (often tables span 2 pages)
            expanded_indices.add(i)
            if i + 1 < len(all_pages_text):
                expanded_indices.add(i + 1)

        # Use detected pages, or default to first 50 pages if detection fails
        if expanded_indices:
            selected_pages = [all_pages_text[i] for i in sorted(expanded_indices)]
            text = "\n\n".join(selected_pages)
        else:
            # Fallback: Read first 50 pages (usually covers 10-K financial section)
            text = "\n\n".join(all_pages_text[:50])

        # Soft truncate to keep within context window (GPT-4o-mini has 128k context)
        # 100k chars is approx 25k tokens, very safe.
        return text[:100000]  # <--- INCREASED LIMIT

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

cols_lower = {c.lower(): c for c in df.columns}
if "item" not in cols_lower or "amount" not in cols_lower:
    st.error(
        "Your data must contain columns named 'Item' and 'Amount' "
        "(case insensitive). Found columns: " + ", ".join(df.columns)
    )
    st.stop()

item_col = cols_lower["item"]
amount_col = cols_lower["amount"]

df = df[[item_col, amount_col]].rename(columns={item_col: "Item", amount_col: "Amount"})
df = ensure_numeric(df)

if df.empty:
    st.error("No valid numeric 'Amount' values found.")
    st.stop()

# We always infer Category locally
df["Category"] = df["Item"].apply(guess_category)

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



