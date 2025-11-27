# app.py
# "How X Makes Money" - Streamlit + Plotly + GPT extraction

import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # handled later

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
    """Very simple keyword-based guess for a line item category."""
    if not isinstance(name, str):
        return "Ignore"
    n = name.lower()

    if any(w in n for w in [
        "revenue", "sales", "subscriptions", "subscription",
        "licensing", "license", "ads", "advertising",
        "cloud", "services", "membership"
    ]):
        return "Revenue"

    if any(w in n for w in [
        "cost of revenues", "cost of revenue",
        "cost of goods", "cogs", "cost of sales"
    ]):
        return "COGS"

    if any(w in n for w in ["research", "r&d", "development"]):
        return "R&D"

    if any(w in n for w in [
        "selling", "sales and marketing", "sales & marketing",
        "marketing"
    ]):
        return "Sales & Marketing"

    if any(w in n for w in [
        "general and administrative", "g&a",
        "administrative", "admin"
    ]):
        return "G&A"

    if "tax" in n:
        return "Tax"

    return "Other Opex"


def load_sample_df() -> pd.DataFrame:
    """Sample Google like income statement."""
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
        return "rgba(150,150,150,{:.2f})".format(alpha)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Streamlit session defaults
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
    # st.secrets for Streamlit Cloud
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    # local env var fallback
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    if OpenAI is None:
        return None

    return OpenAI(api_key=api_key)


def extract_pnl_with_llm(raw_text: str):
    """
    Use GPT to extract an income statement from raw text.

    Returns:
        df (DataFrame with Item, Amount, Category),
        detected_company (str or None),
        detected_currency (str or None)
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError(
            "OpenAI client is not configured. "
            "Install openai and set OPENAI_API_KEY in secrets or env."
        )

    system_prompt = """
You are a financial analyst. Extract the INCOME STATEMENT (profit and loss)
from the given text and map it to a standardized schema.

Output only valid JSON in this exact format:

{
  "company": "Company name or null",
  "currency": "3 letter currency code or null",
  "lines": [
    {"item": "Sales", "amount": 1234.56, "category": "Revenue"},
    {"item": "Cost of goods sold", "amount": 999.99, "category": "COGS"},
    {"item": "Research and development", "amount": 111.11, "category": "R&D"},
    {"item": "Selling and marketing", "amount": 222.22, "category": "Sales & Marketing"},
    {"item": "General and administrative", "amount": 333.33, "category": "G&A"},
    {"item": "Other operating expenses", "amount": 444.44, "category": "Other Opex"},
    {"item": "Income tax expense", "amount": 555.55, "category": "Tax"}
  ]
}

Rules:
- Only include items that clearly belong to the income statement.
- Category must be one of:
  "Revenue","COGS","R&D","Sales & Marketing","G&A","Other Opex","Tax".
- Use positive numbers for all amounts.
- If multiple lines belong to the same conceptual bucket, keep them as separate items.
""".strip()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    # remove optional markdown fences
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
            if content.lower().startswith("json"):
                content = content[4:]
        content = content.strip()

    data = json.loads(content)

    lines = data.get("lines", [])
    rows = []
    for line in lines:
        try:
            rows.append(
                {
                    "Item": line["item"],
                    "Amount": float(line["amount"]),
                    "Category": line.get("category", "Other Opex"),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)

    detected_company = data.get("company")
    detected_currency = data.get("currency")

    return df, detected_company, detected_currency


# -------------------------------------------------------------------
# 2. Layout: title, branding, and input controls
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
    ["Paste statement (AI extract)", "Upload CSV/Excel", "Use sample data"],
)

company_name_override = st.sidebar.text_input(
    "Company name (optional override)",
    value=st.session_state.detected_company,
)

# Top of main page
if logo_file is not None:
    st.image(logo_file, width=140)

st.title("How X Makes Money")
st.write(
    "Paste a company income statement for AI extraction or upload a ready CSV, "
    "review the line items, and generate a Sankey diagram that shows how the "
    "company makes and spends money."
)

# -------------------------------------------------------------------
# 3. Input modes to populate st.session_state.raw_df
# -------------------------------------------------------------------

raw_df = None

if input_mode == "Use sample data":
    raw_df = load_sample_df()
    st.session_state.raw_df = raw_df
    st.session_state.detected_company = company_name_override or "Example Corp"

elif input_mode == "Upload CSV/Excel":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="File must have columns 'Item' and 'Amount'.",
        key="csv_uploader",
    )
    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            raw_df = pd.read_excel(uploaded)
        st.session_state.raw_df = raw_df
        st.session_state.detected_company = company_name_override or "Example Corp"

elif input_mode == "Paste statement (AI extract)":
    st.subheader("Step 0 – Paste income statement text")
    raw_text = st.text_area(
        "Paste the income statement or the relevant section of the annual report.",
        height=260,
        key="raw_text_area",
    )

    if st.button("Extract with AI"):
        if not raw_text.strip():
            st.warning("Please paste some text first.")
        else:
            with st.spinner("Calling GPT to extract the income statement..."):
                try:
                    df_ai, detected_company, detected_currency = extract_pnl_with_llm(raw_text)
                except Exception as e:
                    st.error(
                        "AI extraction failed. Check your API key or try a simpler snippet. "
                        f"Technical detail: {e}"
                    )
                    st.stop()

            if df_ai.empty:
                st.error("AI did not return any line items. Try pasting a clearer statement.")
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
    st.info("Provide data by pasting a statement and clicking Extract, uploading a CSV, or using the sample.")
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

if "Category" not in df.columns:
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

gross_profit = max(total_revenue - cogs, 0)
total_opex = rnd + sm + ga + other_opex
operating_profit = max(gross_profit - total_opex, 0)
net_income = max(operating_profit - tax, 0)

# pick final company name
company_name = company_name_override or st.session_state.detected_company or "This company"

# -------------------------------------------------------------------
# 6. Sankey builder with brand color
# -------------------------------------------------------------------

def build_sankey(df: pd.DataFrame, primary_color: str):
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

    revenue_rows = df[df["Category"] == "Revenue"]
    for _, row in revenue_rows.iterrows():
        name = row["Item"]
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

    # 1) Revenue segments -> Total revenue
    for _, row in revenue_rows.iterrows():
        seg_name = row["Item"]
        amount = float(row["Amount"])
        add_link(seg_name, "Total revenue", amount)

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
    fig = build_sankey(df, brand_color)
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Tip: use the camera icon in the top right of the chart to download it as a PNG."
    )
else:
    st.caption("Press the button above to build the Sankey diagram.")

