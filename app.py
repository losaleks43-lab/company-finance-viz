# app.py
# Streamlit + Plotly prototype: "How X Makes Money" visualization
#
# Usage:
#   1. Install deps (in a virtual env is best):
#        pip install streamlit pandas plotly openpyxl
#   2. Run:
#        streamlit run app.py
#   3. In the browser: upload a CSV/Excel with columns: Item, Amount
#      or use the built-in sample, adjust categories, click "Generate chart".

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="How X Makes Money", layout="wide")

st.title("💸 How X Makes Money")
st.write(
    "Upload a company's income statement (or use the sample data), "
    "map line items to categories, and generate a Sankey diagram showing "
    "how the company makes and spends money."
)

# -------------------------------------------------------------------
# 1. Helpers
# -------------------------------------------------------------------

CATEGORIES = [
    "Revenue",          # top-line revenue items
    "COGS",             # cost of goods / cost of revenues
    "R&D",              # research & development
    "Sales & Marketing",
    "G&A",              # general & administrative
    "Other Opex",       # other operating expenses
    "Tax",
    "Ignore",           # not used in visualization
]


def guess_category(name: str) -> str:
    """Very simple keyword-based guess for a line item category."""
    if not isinstance(name, str):
        return "Ignore"
    n = name.lower()

    # Revenue
    if any(w in n for w in ["revenue", "sales", "subscriptions", "subscription",
                            "licensing", "license", "ads", "advertising",
                            "cloud", "services", "membership"]):
        return "Revenue"

    # COGS / cost of revenues
    if any(w in n for w in ["cost of revenues", "cost of revenue",
                            "cost of goods", "cogs", "cost of sales"]):
        return "COGS"

    # R&D
    if any(w in n for w in ["research", "r&d", "development"]):
        return "R&D"

    # Sales & Marketing
    if any(w in n for w in ["selling", "sales and marketing", "sales & marketing",
                            "marketing"]):
        return "Sales & Marketing"

    # G&A
    if any(w in n for w in ["general and administrative", "g&a",
                            "administrative", "admin"]):
        return "G&A"

    # Tax
    if "tax" in n:
        return "Tax"

    # Fallback
    return "Other Opex"


def load_sample_df() -> pd.DataFrame:
    """Sample 'Google-ish' income statement to play with."""
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
            72000,  # numbers in millions
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
    df = pd.DataFrame(data)
    return df


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])
    return df


# -------------------------------------------------------------------
# 2. Input: upload or sample
# -------------------------------------------------------------------

st.sidebar.header("Input data")

input_mode = st.sidebar.radio(
    "Choose data source",
    ["Use sample data", "Upload CSV/Excel"],
)

company_name = st.sidebar.text_input("Company name", value="Example Corp")

if input_mode == "Use sample data":
    df = load_sample_df()
else:
    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="File must have at least two columns: 'Item' and 'Amount'.",
    )
    if uploaded is None:
        st.warning("Upload a file on the left sidebar or switch to sample data.")
        st.stop()
    # Try CSV then Excel
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_excel(uploaded)

# Basic column check / rename
cols_lower = {c.lower(): c for c in df.columns}
if "item" not in cols_lower or "amount" not in cols_lower:
    st.error(
        "Your file must contain columns named 'Item' and 'Amount' "
        "(case-insensitive). Found columns: " + ", ".join(df.columns)
    )
    st.stop()

item_col = cols_lower["item"]
amount_col = cols_lower["amount"]

df = df[[item_col, amount_col]].rename(columns={item_col: "Item", amount_col: "Amount"})
df = ensure_numeric(df)

if df.empty:
    st.error("No valid numeric 'Amount' values found.")
    st.stop()

# -------------------------------------------------------------------
# 3. Auto-categorize and let the user edit in a data editor
# -------------------------------------------------------------------

if "Category" not in df.columns:
    df["Category"] = df["Item"].apply(guess_category)

st.subheader("Step 1 – Review and adjust categories")

st.write(
    "We guessed a category for each line based on its name. "
    "You can change the *Category* column below. Lines marked as **Ignore** "
    "won't appear in the visualization."
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

# -------------------------------------------------------------------
# 4. Aggregation and basic sanity checks
# -------------------------------------------------------------------

# Use edited data moving forward
df = edited_df.copy()
df = ensure_numeric(df)

# Drop ignored rows
df = df[df["Category"] != "Ignore"]

if df.empty:
    st.error("All rows are marked as 'Ignore'. Please assign some categories.")
    st.stop()

# Aggregate by category
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

# -------------------------------------------------------------------
# 5. Build Sankey nodes and links
# -------------------------------------------------------------------

def build_sankey(df: pd.DataFrame):
    labels = []
    color_map = {}

    # Core nodes
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
        "Total revenue": "#4285F4",      # blue
        "Cost of revenues": "#DB4437",   # red
        "Gross profit": "#0F9D58",       # green
        "R&D": "#AB47BC",                # purple
        "Sales & marketing": "#F4B400",  # yellow
        "G&A": "#00ACC1",                # teal
        "Other opex": "#8D6E63",         # brown
        "Operating profit": "#0F9D58",
        "Tax": "#DB4437",
        "Net income": "#0F9D58",
    }
    for lab, col in base_colors.items():
        color_map[lab] = col

    # Revenue segment nodes (one per line with Category == "Revenue")
    revenue_rows = df[df["Category"] == "Revenue"]
    for _, row in revenue_rows.iterrows():
        name = row["Item"]
        if name not in labels:
            labels.append(name)
            color_map[name] = "#8AB4F8"  # lighter blue

    # NOW build index after all labels are known
    idx = {lab: i for i, lab in enumerate(labels)}

    sources = []
    targets = []
    values = []
    link_colors = []

    def add_link(src_label, tgt_label, value):
        if value <= 0:
            return
        s = idx[src_label]
        t = idx[tgt_label]
        sources.append(s)
        targets.append(t)
        values.append(value)
        link_colors.append("rgba(150,150,150,0.4)")

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
                    pad=18,
                    thickness=20,
                    line=dict(color="black", width=0.3),
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
        font=dict(size=12),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig

# -------------------------------------------------------------------
# 6. Show metrics + Sankey
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
    "Click **Generate chart** after you are happy with the categories above. "
    "Hover over the flows to see exact amounts."
)

if st.button("Generate chart"):
    fig = build_sankey(df)
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Tip: use the camera icon in the top-right of the chart to download it as a PNG for your report/slides."
    )
else:
    st.caption("Press the button above to build the Sankey diagram.")
