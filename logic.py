# logic.py
# Backend Logic: PDF Scanning & AI Analysis
import os
import json
import pandas as pd
import streamlit as st

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# -------------------------------------------------------------------
# 1. Client Setup
# -------------------------------------------------------------------
def get_openai_client():
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        api_key = os.getenv("OPENAI_API_KEY")
        
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

# -------------------------------------------------------------------
# 2. Universal Page Scanner
# -------------------------------------------------------------------
def extract_text_from_uploaded_file(uploaded_file) -> str:
    """
    Scans PDF for pages with high financial density.
    Prioritizes 'Consolidated' tables, penalizes 'Standalone'.
    """
    if uploaded_file is None: return ""
    
    if uploaded_file.name.lower().endswith(".pdf"):
        if PdfReader is None:
            return "ERROR: Pypdf is not installed."
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
                "segment information", "revenue by segment", "disaggregated revenue", "revenue by geography",
                "revenue from operations", "net sales", "cost of sales"
            ]
            negative_keywords = ["standalone", "separate financial statements"]

            for i in range(num_pages):
                try:
                    text = reader.pages[i].extract_text()
                    if not text: continue
                    low_text = text.lower()
                    score = 0
                    
                    # 1. Big boost for Main P&L
                    for kw in tier_1_keywords:
                        if kw in low_text: score += 20
                    
                    # 2. Boost for Segment/Geo Data
                    for kw in tier_2_keywords:
                        if kw in low_text: score += 5

                    # 3. Penalize Standalone
                    for kw in negative_keywords:
                        if kw in low_text: score -= 15
                        
                    # 4. Context Boost
                    if "year ended" in low_text and ("in millions" in low_text or "in crores" in low_text):
                        score += 2

                    page_scores.append((i, score))
                except: continue
            
            # Select Top 8 Scoring Pages
            top_pages = sorted(page_scores, key=lambda x: x[1], reverse=True)[:8]
            top_indices = [p[0] for p in top_pages]
            
            # Add neighbors
            final_indices = set(top_indices)
            for idx in top_indices:
                if idx + 1 < num_pages: final_indices.add(idx + 1)
            
            sorted_indices = sorted(list(final_indices))
            
            extracted_text = ""
            for i in sorted_indices:
                extracted_text += f"--- PAGE {i+1} ---\n{reader.pages[i].extract_text()}\n\n"
                
            return extracted_text

        except Exception as e:
            return f"PDF Error: {e}"
            
    # Fallback for TXT files
    try:
        return uploaded_file.read().decode("utf-8", errors="ignore")[:100000]
    except:
        return ""

# -------------------------------------------------------------------
# 3. Multi-Mode Analysis Logic
# -------------------------------------------------------------------
def analyze_financials(raw_text: str, mode: str = "product"):
    """
    Universal Analysis Function.
    modes:
      - 'product': Segment breakdown (Left side)
      - 'geo': Geographic breakdown (Left side)
      - 'pnl': Standard Consolidated P&L (No breakdown)
    """
    client = get_openai_client()
    if client is None:
        return None, None, None

    # --- SHARED COST INSTRUCTIONS ---
    cost_instructions = """
    --- STEP 2: COSTS & EXPENSES (Right Side) ---
    1. **DIRECT COSTS (COGS):**
       - Manufacturing/Retail: "Cost of Materials", "Purchases", "Inventory Changes", "Excise Duty".
       - Tech: "Cost of Revenue", "TAC", "Data Center Costs".
       - Bank: "Interest Expended".
       - **Crucial:** Extract these as individual lines. Tag category as "COGS".
       
    2. **OPERATING EXPENSES:**
       - Extract major buckets: "R&D", "Sales & Marketing", "G&A".
       - If "Employee Benefits" is a major separate line, extract it.
       
    3. **TAX:**
       - Extract "Income Tax Expense". Tag category as "Tax".
    """

    # --- MODE SPECIFIC INSTRUCTIONS ---
    if mode == "geo":
        rev_instructions = """
        --- STEP 1: REVENUE BY GEOGRAPHY (Left Side) ---
        - Find the **Geographic/Regional Revenue** table (e.g. "North America", "Europe", "APAC", "India", "US").
        - Extract the regions as your Revenue items.
        - Tag them as category: "Revenue".
        - **Check:** Sum of regions must approx match Total Net Revenue.
        """
    elif mode == "pnl":
        rev_instructions = """
        --- STEP 1: STANDARD P&L (Left Side) ---
        - Find the **"Total Revenue"** or **"Net Sales"** line from the main Income Statement.
        - Do NOT split by segment or geography. Just use the single top-line number.
        - Label it "Net Sales" or "Total Revenue".
        - Tag it as category: "Revenue".
        """
    else: # default to 'product'
        rev_instructions = """
        --- STEP 1: REVENUE BY PRODUCT/SEGMENT (Left Side) ---
        - Find the **Business Segment Revenue** (e.g. "iPhone", "Services", "Retail", "Digital", "Cloud").
        - Extract these segments.
        - Tag them as category: "Revenue".
        - **Anti-Double Counting:** Do NOT include "Total Revenue" if you list segments.
        """

    system_prompt = f"""
    You are an expert Financial Analyst.
    
    YOUR GOAL: Extract the financial flows for {mode.upper()} view.
    
    {rev_instructions}
    
    {cost_instructions}

    OUTPUT JSON FORMAT:
    {{
        "company": "Company Name",
        "currency": "Currency",
        "lines": [
            {{"item": "North America", "amount": 50000, "category": "Revenue"}},
            {{"item": "Cost of Materials", "amount": 30000, "category": "COGS"}}
        ]
    }}
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
