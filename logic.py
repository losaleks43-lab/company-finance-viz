# logic.py
import os
import json
import pandas as pd
import streamlit as st

# Try imports to handle missing libraries gracefully
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
    # Tries to get key from Streamlit secrets first, then Environment variables
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
    
    # PDF Handling
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
                "segment information", "revenue by segment", "disaggregated revenue",
                "revenue from operations", "net sales", "cost of sales"
            ]
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
                    
                    # 2. Boost for Segment Data
                    for kw in tier_2_keywords:
                        if kw in low_text: score += 5

                    # 3. Penalize Standalone (Crucial for Reliance/Asian reports)
                    for kw in negative_keywords:
                        if kw in low_text: score -= 15
                        
                    # 4. Context Boost
                    if "year ended" in low_text and ("in millions" in low_text or "in crores" in low_text):
                        score += 2

                    page_scores.append((i, score))
                except: continue
            
            # Select Top 6 Scoring Pages
            top_pages = sorted(page_scores, key=lambda x: x[1], reverse=True)[:6]
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
# 3. AI Analysis Logic
# -------------------------------------------------------------------
def extract_pnl_with_llm(raw_text: str):
    """
    Uses Chain-of-Thought reasoning to extract and categorize P&L data.
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client is not configured.")

    system_prompt = """
    You are an expert Financial Analyst building a dataset for a Sankey diagram.
    
    YOUR GOAL: Extract the **CONSOLIDATED** Income Statement for the most recent year.
    
    --- PHASE 1: IDENTIFY BUSINESS MODEL ---
    Determine if the company is:
    A) Tech/Services (e.g., Google, Meta) -> Look for "Cost of Revenue" or "TAC".
    B) Manufacturing/Retail (e.g., Reliance, Walmart) -> Look for "Cost of Materials", "Purchases", "Inventory Changes".
    C) Bank/Financial -> Look for "Interest Expense", "Provisions".

    --- PHASE 2: EXTRACT LINE ITEMS ---

    1. **REVENUE (Left Side)**
       - Find **Net Revenue** (ignore "Gross Revenue" if it includes collected taxes like GST/Excise).
       - Find the **Segment Breakdown** (e.g. "Cloud", "Search" or "Retail", "O2C").
       - Output the SEGMENTS. If segments are unavailable, output the Net Revenue total.
       - **Double Counting Check:** Do NOT output both the Total AND the Segments. Segments only.

    2. **DIRECT COSTS / COGS (Middle)**
       - Based on the Business Model identified in Phase 1, extract the direct costs.
       - **Crucial:** If the report lists "Cost of Materials", "Purchases of Stock", and "Inventory Changes" separately, extract them ALL individually. Tag them as "COGS".
       - Do NOT extract "Total Expenses".

    3. **OPERATING EXPENSES (Right Side)**
       - Extract major buckets: "R&D", "Sales & Marketing", "G&A", "Depreciation".
       - If "Employee Benefits" is a major separate line (common in India/EU), extract it.

    4. **TAX**
       - Extract "Tax Expense" or "Current + Deferred Tax". Tag as "Tax".

    OUTPUT JSON FORMAT:
    {
        "company": "Company Name",
        "currency": "Currency Symbol",
        "lines": [
            {"item": "Segment A", "amount": 100, "category": "Revenue"},
            {"item": "Cost of Materials", "amount": 60, "category": "COGS"},
            {"item": "R&D", "amount": 10, "category": "R&D"}
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
