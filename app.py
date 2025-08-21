# app.py
# -------------------------------------------------------
# Financial QA — RAG vs Fine-Tuning (MiniLM + MoE)
# §4.2 logic preserved; Status→Correct kept.
# Confidence:
#   1) Try to read from out/df_extended_view.csv (Question+Method match)
#   2) Uses natural-language / fuzzy matching (punctuation-insensitive,
#      case-insensitive, whitespace-normalized, plus SequenceMatcher + token Jaccard)
#   3) Adds light structure checks (ticker/year/metric match boosts)
#   4) If no good match, falls back to pipeline's computed confidence.
# -------------------------------------------------------

import os, re, glob, json, time, math, sys, subprocess
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import unicodedata

import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# ------------------------------
# Page config & CSS
# ------------------------------
st.set_page_config(
    page_title="Financial QA — RAG vs Fine-Tuning (Companies - ADP, AAGH, AFRM & AEHR)",
    page_icon="📈",
    layout="wide",
)

BOX_PX = 130
MID_FRAC = 0.62

# === UI TWEAKS: unified grey background + bold black text for Question & Answer ===
st.markdown(f"""
<style>
footer {{visibility: hidden;}}
.block-container {{padding-top: 1.0rem;}}

:root {{
  --box-max-width: 560px;     /* width cap */
  --two-lines: 3.2em;         /* ≈ 2 lines height */
  --qa-bg: #f3f4f6;           /* shared grey background for Q & A */
  --qa-text: #000;            /* black text */
  --qa-weight: 700;           /* bold */
}}

div[data-testid="stMetricValue"] {{font-variant-numeric: tabular-nums;}}
.smallgray {{color:#6b7280; font-size:12px;}}

/* Limit the width of the Question input */
div[data-testid="stTextArea"] {{
  max-width: var(--box-max-width);
}}

/* Question (textarea): 2 lines, no scroll, unified bg/border + bold black text */
div[data-testid="stTextArea"] textarea {{
  line-height: 1.2em !important;
  height: var(--two-lines) !important;
  max-height: var(--two-lines) !important;
  overflow-y: hidden !important;
  resize: none !important;

  background: var(--qa-bg) !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 10px !important;

  color: var(--qa-text) !important;
  font-weight: var(--qa-weight) !important;
}}

/* Keep placeholder readable (not bold) */
div[data-testid="stTextArea"] textarea::placeholder {{
  color: #6b7280 !important;
  font-weight: 400 !important;
}}

/* Answer box: same width, 2-line clamp, SAME bg + bold black text */
.qa-box {{
  max-width: var(--box-max-width);
  overflow: hidden;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 10px;
  background: var(--qa-bg) !important;

  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  line-height: 1.2em;
  max-height: var(--two-lines);

  color: var(--qa-text) !important;
  font-weight: var(--qa-weight) !important;
}}

/* Ensure nested tags inside the answer inherit bold black */
.qa-box * {{
  color: var(--qa-text) !important;
  font-weight: var(--qa-weight) !important;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Paths & helpers
# ------------------------------
def find_path(*cands: str) -> Optional[str]:
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None

def in_out(*tails: str) -> Optional[str]:
    p = "/".join(tails)
    return find_path(os.path.join("Out", p), os.path.join("out", p))

FIN_CSV      = in_out("financials_last2y.csv")
SECTIONS_DIR = in_out("sections_text_yearly")
VOCABS_JSON  = in_out("vocabs.json")
FT_CKPT      = in_out("minilm_moe_best.pt")
CONF_CSV     = in_out("df_extended_view.csv")  # for Confidence lookup

if not FIN_CSV:
    st.error("Missing financials_last2y.csv (run Step-1 first).")
    st.stop()

# ------------------------------
# Session state
# ------------------------------
ss = st.session_state
ss.setdefault("rows", [])
ss.setdefault("last_answer", None)
ss.setdefault("eval_tables", None)
ss.setdefault("eval_msg", "")
ss.setdefault("q_input", "type your question here")
ss.setdefault("save_log", "")
ss.setdefault("show_table", False)
ss.setdefault("rag_use_ce", True)
ss.setdefault("ft_use_moe", True)


# Apply queued reset BEFORE any widgets are created
if ss.pop("_do_reset", False):
    ss["rows"] = []
    ss["last_answer"] = None
    ss["eval_tables"] = None
    ss["eval_msg"] = ""
    ss["save_log"] = ""
    ss["q_input"] = "type your question here"

# ------------------------------
# Data & constants (match your notebook)
# ------------------------------
df_fin = pd.read_csv(FIN_CSV)
ALLOWED_TICKERS = set(df_fin["ticker"].unique().tolist())

NAME_ALIASES = {
    "automatic data processing":"ADP","adp":"ADP",
    "affirm":"AFRM","affirm holdings":"AFRM",
    "aehr":"AEHR","aehr test systems":"AEHR",
    "america great health":"AAGH","aagh":"AAGH",
}

# Mentions of companies we explicitly do NOT support (used for OOS early exit)
OTHER_TICKER_ALIASES = {
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT",
    "alphabet": "GOOGL", "google": "GOOGL", "googl": "GOOGL", "goog": "GOOGL",
    "amazon": "AMZN", "amzn": "AMZN",
    "meta": "META", "facebook": "META",
    "tesla": "TSLA", "tsla": "TSLA",
    "nvidia": "NVDA", "nvda": "NVDA",
    "adobe": "ADBE", "adbe": "ADBE",
}

# Hard scope for the app regardless of what's in the CSV
SUPPORTED_TICKERS = {"ADP", "AAGH", "AFRM", "AEHR"}

# Scales for FT post-processing (keys MUST be lowercase)
# Keep your originals + add canonical names from the new metric dictionary.
ITEM_SCALE = {
    # --- your original keys (kept) ---
    "revenue": 1e9,
    "cost of revenue": 1e9,
    "gross profit": 1e9,
    "net income": 1e9,
    "operating cash flow": 1e9,          # alias for "net cash from operating activities"
    "total assets": 1e9,
    "total liabilities": 1e9,
    "shareholders equity": 1e9,          # alias for "total equity"
    "cash & equivalents": 1e9,           # alias for "cash, cash equivalents & short term investments"
    "capex": 1e6,                        # alias for "capital expenditures"

    # --- canonical names used by the new detector (additions) ---
    "operating income ( loss)": 1e9,     # EBIT / operating profit
    "net cash from operating activities": 1e9,
    "net cash from investing activities": 1e9,
    "net cash from financing activities": 1e9,
    "net change in cash": 1e9,

    "depreciation & amortization": 1e6,  # often reported in millions; keep conservative
    "research & development": 1e6,
    "selling, general & administrative": 1e6,

    "accounts & notes receivable": 1e9,
    "accounts receivable": 1e9,
    "accounts payable": 1e9,
    "inventories": 1e9,
    "property, plant & equipment, net": 1e9,
    "long term investments & receivables": 1e9,

    "total noncurrent assets": 1e9,
    "total current assets": 1e9,
    "total noncurrent liabilities": 1e9,
    "total current liabilities": 1e9,

    "total equity": 1e9,                 # ← canonical for shareholders’ equity
    "total liabilities & equity": 1e9,

    "long term debt": 1e9,
    "short term debt": 1e9,
    "total debt": 1e9,

    "share capital & additional paid- in capital": 1e9,
    "retained earnings": 1e9,
    "other comprehensive income": 1e9,
    "treasury stock": 1e9,

    "basic eps": 1.0,                    # EPS are per-share figures; no scaling
    "diluted eps": 1.0,
    "shares ( basic)": 1.0,              # share counts are absolute; set 1.0
    "shares ( diluted)": 1.0,

    "income tax ( expense) benefit, net": 1e9,
    "interest expense, net": 1e9,
    "earnings before tax": 1e9,
    "income ( loss) from continuing operations": 1e9,
    "net income from discontinued operations": 1e9,

    "dividends paid": 1e9,
    "cash from ( repurchase of) equity": 1e9,
    "cash from ( repayment of) debt": 1e9,

    "other operating activities": 1e9,
    "other investing activities": 1e9,
    "other financing activities": 1e9,
    "other current assets": 1e9,
    "other current liabilities": 1e9,
    "other non-current assets": 1e9,
    "other non-current liabilities": 1e9,

    # CapEx & proxies
    "capital expenditures": 1e6,         # canonical capex
    "change in fixed assets & intangibles": 1e6,

    "change in working capital": 1e9,

    # Cash line
    "cash, cash equivalents & short term investments": 1e9,
}

# --- NEW: robust metric dictionary & detection (drop-in) ----------------------
# Put this right after NAME_ALIASES and REPLACE your existing METRIC_PATTERNS
# and detect_metric with everything in this block.

# Helper: make resilient regex for df_fin["line_item"] names
def _robust_metric_regex(name: str) -> str:
    s = name.replace("’", "'")
    s = re.escape(s)
    # spaces -> \s+
    s = re.sub(r"\\\s+", r"\\s+", s)
    # allow & or 'and'
    s = s.replace(r"\&", r"(?:&|and)")
    # optional spaces inside parentheses: "( Loss)" or "(Loss)"
    s = s.replace(r"\(", r"\(\s*").replace(r"\)", r"\s*\)")
    # optional space around hyphen: "Paid- In" or "Paid-In"
    s = s.replace(r"\-", r"\s*-\s*")
    # optional apostrophes in share/stockholders
    s = s.replace("shareholders\\'", "shareholders'?").replace("stockholders\\'", "stockholders'?")
    return r"^" + s + r"$"

# Canonical labels (match df_fin['line_item'] semantics; case-insensitive)
# You can trim this list if your dataset omits some; keeping it broad helps future-proofing.
DATASET_METRICS = [
    "Revenue",
    "Cost Of Revenue",
    "Gross Profit",
    "Operating Income ( Loss)",
    "Net Income",
    "Net Cash From Operating Activities",
    "Net Cash From Investing Activities",
    "Net Cash From Financing Activities",
    "Net Change In Cash",
    "Depreciation & Amortization",
    "Research & Development",
    "Selling, General & Administrative",
    "Accounts & Notes Receivable",
    "Accounts Receivable",                      # (present in some exports)
    "Accounts Payable",                         # (present in some exports)
    "Inventories",
    "Property, Plant & Equipment, Net",
    "Long Term Investments & Receivables",
    "Total Assets",
    "Total Noncurrent Assets",
    "Total Current Assets",
    "Total Liabilities",
    "Total Noncurrent Liabilities",
    "Total Current Liabilities",
    "Total Equity",
    "Total Liabilities & Equity",
    "Long Term Debt",
    "Short Term Debt",
    "Total Debt",
    "Share Capital & Additional Paid- In Capital",
    "Retained Earnings",
    "Other Comprehensive Income",
    "Treasury Stock",
    "Shares ( Basic)",
    "Shares ( Diluted)",
    "Basic EPS",
    "Diluted EPS",
    "Income Tax ( Expense) Benefit, Net",
    "Interest Expense, Net",
    "Earnings Before Tax",
    "Income ( Loss) From Continuing Operations",
    "Net Income From Discontinued Operations",
    "Dividends Paid",
    "Cash From ( Repurchase Of) Equity",
    "Cash From ( Repayment Of) Debt",
    "Other Operating Activities",
    "Other Investing Activities",
    "Other Financing Activities",
    "Other Current Assets",
    "Other Current Liabilities",
    "Other Non-Current Assets",
    "Other Non-Current Liabilities",
    "Change In Fixed Assets & Intangibles",     # CapEx proxy in some exports
    "Capital Expenditures",                     # Explicit CapEx if present
    "Change In Working Capital",
    "Cash, Cash Equivalents & Short Term Investments",
]

# Build resilient regex patterns for each df line item
METRIC_PATTERNS = {
    k.lower(): [_robust_metric_regex(k)]
    for k in DATASET_METRICS
}

# Add extra domain patterns for the most common queries (maps to df rows)
def _add_pats(df_key: str, extra_pats: list):
    lk = df_key.lower()
    METRIC_PATTERNS.setdefault(lk, [])
    METRIC_PATTERNS[lk].extend(extra_pats)

# Revenue family
_add_pats("Revenue", [
    r"^Total\s+Revenue$", r"^Net\s+Sales$", r"^Sales\s+Revenue$", r"^Sales$",
])
# Cost of revenue / COGS
_add_pats("Cost Of Revenue", [
    r"^Cost\s+of\s+Goods\s+Sold$", r"^COGS$", r"^Cost\s+of\s+Sales$",
])
# Gross profit
_add_pats("Gross Profit", [
    r"^Gross\s+Income$", r"^Gross\s+Profit$",
])
# Operating income / EBIT
_add_pats("Operating Income ( Loss)", [
    r"^Operating\s+Income$", r"^Operating\s+Profit$", r"^EBIT$", r"^Income\s+from\s+Operations$",
])
# Net income
_add_pats("Net Income", [
    r"^Net\s+Profit$", r"^Net\s+Earnings$", r"^Profit\s+Attributable$", r"^Earnings$",
])
# Operating cash flow
_add_pats("Net Cash From Operating Activities", [
    r"^Operating\s+Cash\s*Flow$", r"^Cash\s*Flow\s*from\s*Operations$", r"^Net\s*Cash\s*Provided\s*by\s*Operating\s*Activities$",
])
# Investing cash flow
_add_pats("Net Cash From Investing Activities", [
    r"^Investing\s+Cash\s*Flow$", r"^Cash\s*Flow\s*from\s*Investing$",
])
# Financing cash flow
_add_pats("Net Cash From Financing Activities", [
    r"^Financing\s+Cash\s*Flow$", r"^Cash\s*Flow\s*from\s*Financing$",
])
# Cash & equivalents
_add_pats("Cash, Cash Equivalents & Short Term Investments", [
    r"^Cash(,|\s)+Cash\s+Equivalents\s*(?:&|and)\s*Short[-\s]*Term\s+Investments$",
    r"^Cash\s*(?:&|and)\s*Cash\s+Equivalents$",
    r"^Cash\s*(?:&|and)\s*Short[-\s]*Term\s+Investments$",
    r"^Cash\s*(?:&|and)\s*Equivalents$",
])
# Equity / Shareholders' equity
_add_pats("Total Equity", [
    r"^Total\s+(?:Share|Stock)holders'?\s+Equity$",
    r"^Shareholders'?\s+Equity$", r"^Stockholders'?\s+Equity$",
    r"^Total\s+Shareholders'?\s+Funds$",
])
# CapEx
_add_pats("Capital Expenditures", [
    r"^Capital\s+Expenditures?$",
    r"^Purchase[s]?\s+of\s+Property,?\s+Plant\s*(?:&|and)\s*Equipment$",
    r"^Additions\s+to\s+Property,?\s+Plant\s*(?:&|and)\s*Equipment$",
    r"^Purchases\s+of\s+PP&E$",
])
# Long / short debt
_add_pats("Long Term Debt", [
    r"^Long[-\s]*Term\s+Debt$", r"^Noncurrent\s+Debt$",
])
_add_pats("Short Term Debt", [
    r"^Short[-\s]*Term\s+Debt$", r"^Current\s+Portion\s+of\s+Debt$",
])
# Liabilities & equity
_add_pats("Total Liabilities & Equity", [
    r"^Total\s+Liabilities\s*(?:&|and)\s+Equity$",
    r"^Liabilities\s*(?:&|and)\s+(?:Share|Stock)holders'?\s+Equity$",
])
# APIC
_add_pats("Share Capital & Additional Paid- In Capital", [
    r"^Additional\s+Paid[-\s]*In\s+Capital$", r"^Share\s+Premium$",
])
# EBT / taxes / interest
_add_pats("Earnings Before Tax", [
    r"^Earnings\s+Before\s+Tax(?:es)?$", r"^Income\s+Before\s+Tax(?:es)?$", r"^EBT$",
])
_add_pats("Income Tax ( Expense) Benefit, Net", [
    r"^Provision\s+for\s+Income\s+Taxes$", r"^Income\s+Tax\s+Expense$",
])
_add_pats("Interest Expense, Net", [
    r"^Interest\s+Expense(?:,\s*Net)?$",
])
# Buybacks / debt
_add_pats("Cash From ( Repurchase Of) Equity", [
    r"^Share\s+Repurchases$", r"^Stock\s+Buybacks$", r"^Repurchase\s+of\s+Common\s+Stock$",
])
_add_pats("Cash From ( Repayment Of) Debt", [
    r"^Issuance\s*\(Repayment\)\s+of\s+Debt$", r"^Net\s+Borrowings$",
])
# CapEx proxy
_add_pats("Change In Fixed Assets & Intangibles", [
    r"^Investment\s+in\s+PP&E$", r"^Purchase[s]?\s+of\s+PP&E$",
])

# Synonym table (keys MUST be the same canonical text as above, but we look up lower-cased)
# These are phrases users tend to type; order / punctuation doesn’t matter.
METRIC_SYNONYMS = {
    "Revenue".lower(): [
        "revenue","total revenue","net sales","sales revenue","sales","turnover"
    ],
    "Cost Of Revenue".lower(): [
        "cost of revenue","cost of goods sold","cogs","cost of sales"
    ],
    "Gross Profit".lower(): [
        "gross profit","gross income","gross margin (dollars)","gross margin dollars"
    ],
    "Operating Income ( Loss)".lower(): [
        "operating income","operating profit","income from operations","ebit"
    ],
    "Net Income".lower(): [
        "net income","net profit","net earnings","profit attributable","earnings"
    ],
    "Depreciation & Amortization".lower(): [
        "depreciation and amortization","d&a","depr & amort"
    ],
    "Research & Development".lower(): [
        "research and development","r&d expense","r and d"
    ],
    "Selling, General & Administrative".lower(): [
        "selling general and administrative","selling general & administrative","sga","sg&a"
    ],
    "Net Cash From Operating Activities".lower(): [
        "operating cash flow","cash flow from operations","net cash provided by operating activities","operating cashflow"
    ],
    "Net Cash From Investing Activities".lower(): [
        "investing cash flow","cash flow from investing"
    ],
    "Net Cash From Financing Activities".lower(): [
        "financing cash flow","cash flow from financing"
    ],
    "Net Change In Cash".lower(): [
        "net change in cash","change in cash and cash equivalents"
    ],
    "Cash, Cash Equivalents & Short Term Investments".lower(): [
        "cash and cash equivalents","cash & equivalents","cash & short term investments",
        "cash and short term investments","cash & st investments"
    ],
    "Accounts & Notes Receivable".lower(): [
        "accounts receivable","notes receivable","trade receivables","receivables"
    ],
    "Accounts Receivable".lower(): [
        "accounts receivable","trade receivables","receivables"
    ],
    "Accounts Payable".lower(): [
        "accounts payable","trade payables","payables"
    ],
    "Inventories".lower(): [
        "inventory","inventories - net"
    ],
    "Property, Plant & Equipment, Net".lower(): [
        "pp&e net","property plant and equipment net","ppe net","property plant and equipment"
    ],
    "Long Term Investments & Receivables".lower(): [
        "long-term investments","long term investments","noncurrent investments"
    ],
    "Total Assets".lower(): [
        "total assets"
    ],
    "Total Noncurrent Assets".lower(): [
        "noncurrent assets","long term assets","long-term assets"
    ],
    "Total Current Assets".lower(): [
        "current assets","total current assets"
    ],
    "Total Liabilities".lower(): [
        "total liabilities"
    ],
    "Total Noncurrent Liabilities".lower(): [
        "noncurrent liabilities","long term liabilities","long-term liabilities"
    ],
    "Total Current Liabilities".lower(): [
        "current liabilities","total current liabilities"
    ],
    "Total Equity".lower(): [
        "total equity","shareholders equity","stockholders equity","total shareholders equity",
        "total stockholders equity","common equity","equity (balance sheet)"
    ],
    "Total Liabilities & Equity".lower(): [
        "total liabilities and equity","liabilities and shareholders equity","liabilities and stockholders equity"
    ],
    "Long Term Debt".lower(): [
        "long term debt","long-term debt","lt debt","noncurrent debt"
    ],
    "Short Term Debt".lower(): [
        "short term debt","short-term debt","st debt","current portion of debt"
    ],
    "Total Debt".lower(): [
        "total debt","total borrowings","gross debt","interest bearing debt"
    ],
    "Share Capital & Additional Paid- In Capital".lower(): [
        "additional paid in capital","apic","share premium","share capital and apic"
    ],
    "Retained Earnings".lower(): [
        "retained earnings","accumulated earnings","accumulated deficit"
    ],
    "Other Comprehensive Income".lower(): [
        "accumulated other comprehensive income","aoci","other comprehensive income"
    ],
    "Treasury Stock".lower(): [
        "treasury stock","treasury shares","treasury stock at cost"
    ],
    "Shares ( Basic)".lower(): [
        "basic shares outstanding","weighted average shares basic","basic shares"
    ],
    "Shares ( Diluted)".lower(): [
        "diluted shares outstanding","weighted average shares diluted","diluted shares"
    ],
    "Basic EPS".lower(): [
        "earnings per share basic","basic earnings per share","eps basic"
    ],
    "Diluted EPS".lower(): [
        "earnings per share diluted","diluted earnings per share","eps diluted"
    ],
    "Income Tax ( Expense) Benefit, Net".lower(): [
        "income taxes","income tax expense","provision for income taxes"
    ],
    "Interest Expense, Net".lower(): [
        "interest expense net","interest expense"
    ],
    "Earnings Before Tax".lower(): [
        "earnings before tax","income before tax","ebt"
    ],
    "Income ( Loss) From Continuing Operations".lower(): [
        "income from continuing operations","loss from continuing operations"
    ],
    "Net Income From Discontinued Operations".lower(): [
        "income from discontinued operations","discontinued operations income"
    ],
    "Dividends Paid".lower(): [
        "dividends paid","cash dividends"
    ],
    "Cash From ( Repurchase Of) Equity".lower(): [
        "share repurchases","stock buybacks","repurchase of common stock","buyback of shares"
    ],
    "Cash From ( Repayment Of) Debt".lower(): [
        "debt issued repaid","issuance (repayment) of debt","net borrowings","debt repayments"
    ],
    "Other Operating Activities".lower(): [
        "other operating cash flows"
    ],
    "Other Investing Activities".lower(): [
        "other investing cash flows"
    ],
    "Other Financing Activities".lower(): [
        "other financing cash flows"
    ],
    "Other Current Assets".lower(): [
        "other current assets"
    ],
    "Other Current Liabilities".lower(): [
        "other current liabilities"
    ],
    "Other Non-Current Assets".lower(): [
        "other noncurrent assets"
    ],
    "Other Non-Current Liabilities".lower(): [
        "other noncurrent liabilities"
    ],
    "Change In Fixed Assets & Intangibles".lower(): [
        "capex","capital expenditures","investment in ppe","purchase of ppe",
        "change in fixed assets and intangibles"
    ],
    "Capital Expenditures".lower(): [
        "capex","capital expenditures","purchases of property plant and equipment","additions to property plant and equipment","purchases of ppe"
    ],
    "Change In Working Capital".lower(): [
        "change in working capital","working capital change"
    ],
}

# Normalizer for question text
def _normalize_text_for_match(t: str) -> str:
    t = (t or "").lower()
    t = t.replace("’","'")
    t = re.sub(r"[^a-z0-9$%&./\-\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def detect_metric(q: str) -> Optional[str]:
    """
    Return the CANONICAL metric key (lower-cased) that exists in METRIC_PATTERNS.
    1) Fast substring check on synonyms with word boundaries
    2) Fuzzy tie-break with a HARD THRESHOLD (prevents 'capital of france' => 'capital expenditures')
    """
    qn = _normalize_text_for_match(q)
    best_key, best_sim = None, 0.0

    # 1) direct containment over synonyms (high precision)
    for canon_key, syns in METRIC_SYNONYMS.items():
        for s in syns:
            pat = r"(^|\b)" + re.escape(s) + r"(\b|$)"
            if re.search(pat, qn):
                return canon_key  # strong match → return immediately

    # 2) fuzzy: choose highest similarity but only if ≥ threshold
    from difflib import SequenceMatcher
    for canon_key, syns in METRIC_SYNONYMS.items():
        for s in syns:
            sim = SequenceMatcher(None, qn, s).ratio()
            if sim > best_sim:
                best_sim, best_key = sim, canon_key

    # HARD THRESHOLD to avoid generic phrases (e.g., "capital of france")
    # 0.74 is a good default for short queries; adjust ±0.03 if needed.
    return best_key if best_sim >= 0.74 else None


# --- END NEW BLOCK ------------------------------------------------------------

def parse_query(q: str) -> Tuple[Optional[str], Optional[int]]:
    ticker = None
    # Only consider our four supported tickers
    for t in SUPPORTED_TICKERS:
        if re.search(rf"\b{re.escape(t)}\b", q):
            ticker = t
            break
    if not ticker:
        ql = q.lower()
        for alias, tk in NAME_ALIASES.items():
            if ((len(alias) <= 6 and re.search(rf"\b{re.escape(alias)}\b", ql))
                or (len(alias) > 6 and alias in ql)):
                if tk in SUPPORTED_TICKERS:
                    ticker = tk
                    break
    m = re.search(r"\b(20\d{2})\b", q)
    year = int(m.group(1)) if m else None
    return ticker, year

def _detect_scale(u: str) -> float:
    s = (u or "").lower()
    if "thousand" in s or "(thousand" in s: return 1e3
    if "million" in s  or "(million"  in s: return 1e6
    if "billion" in s  or "(billion"  in s: return 1e9
    if "trillion" in s or "(trillion" in s: return 1e12
    for pat, mul in [(r"\bk\b",1e3),(r"\bmn|mm|m\b",1e6),(r"\bbn|b\b",1e9),(r"\btn|t\b",1e12)]:
        if re.search(pat, s): return mul
    return 1.0

def _detect_currency(u: str) -> str:
    U = (u or "").upper()
    m = re.search(r"\b(USD|EUR|GBP|JPY|CNY|CAD|AUD)\b", U)
    return m.group(1) if m else "USD"

def normalize_value(value: float, unit: str) -> Tuple[float, str]:
    return float(value) * _detect_scale(unit), _detect_currency(unit)

def fmt_money(val: float) -> str:
    v = float(val); a = abs(v); pref = "-" if v < 0 else ""
    if a >= 1e12: return f"{pref}${a/1e12:.2f} trillion"
    if a >= 1e9:  return f"{pref}${a/1e9:.2f} billion"
    if a >= 1e6:  return f"{pref}${a/1e6:.2f} million"
    if a >= 1e3:  return f"{pref}${a/1e3:.0f} thousand"
    return f"{pref}${a:.0f}"

def lookup_metric(df: pd.DataFrame, ticker: str, year: int, metric: str):
    if metric not in METRIC_PATTERNS: return None
    sub = df[(df["ticker"]==ticker) & (df["fiscal_year"]==year)].copy()
    if sub.empty: return None
    for pat in METRIC_PATTERNS[metric]:
        m = sub[sub["line_item"].str.contains(pat, case=False, regex=True)].copy()
        if not m.empty:
            row = (m.assign(abs_val=m["value"].abs()).nlargest(1, "abs_val").iloc[0])
            v,u = normalize_value(row["value"], row["unit"])
            return {"line_item": row["line_item"], "value_norm": v, "unit_norm": u, "statement": row["statement"]}
    return None

def _gt_val_only(t, y, m):
    if not (t and y and m): return None
    hit = lookup_metric(df_fin, t, int(y), m)
    if hit: return float(hit["value_norm"])
    sub = df_fin.loc[(df_fin["ticker"]==t) & (df_fin["fiscal_year"]==int(y)) &
                     (df_fin["line_item"].str.contains(re.escape(m), case=False, regex=True))].copy()
    if sub.empty: return None
    row = (sub.assign(abs_val=sub["value"].abs()).nlargest(1, "abs_val").iloc[0])
    v,_ = normalize_value(row["value"], row["unit"])
    return float(v)

# ------------------------------
# Confidence table (CSV) with robust fuzzy matching  ⟵ (ONLY-TOUCHED EARLIER, unchanged now)
# ------------------------------
STOPWORDS = set("a an and are as at be but by for if in into is it its of on or such that the their then there these this to with was were what how much many the from for over under during between within about around".split())

@st.cache_data(show_spinner=False)
def load_conf_table(path: Optional[str]):
    if not path or not os.path.exists(path):
        return None
    try:
        dfc = pd.read_csv(path)
        cols = {c.strip(): c for c in dfc.columns}
        need = {"Question","Method","Confidence"}
        if not need.issubset(set(cols.keys())):
            return None
        dfc = dfc.rename(columns={cols["Question"]:"Question", cols["Method"]:"Method", cols["Confidence"]:"Confidence"})
        dfc["Question"] = dfc["Question"].astype(str)
        dfc["Method"] = dfc["Method"].astype(str)
        return dfc
    except Exception:
        return None
    
def _canonicalize_entities_for_match(text: str) -> str:
    """
    Replace company aliases (e.g., 'automatic data processing') with canonical tickers (e.g., 'ADP')
    so CSV rows that use tickers will match user questions that use names.
    """
    if not text:
        return ""
    t = text.lower()

    # Replace possessives like "adp's" -> "adp" first to avoid boundary issues
    t = re.sub(r"\b([a-z]{1,6})'s\b", r"\1", t)

    # Replace longer aliases first to avoid partial shadowing
    for alias, tk in sorted(NAME_ALIASES.items(), key=lambda kv: -len(kv[0])):
        al = alias.lower()
        # short aliases → word boundary; long aliases → substring is fine
        if len(al) <= 6:
            t = re.sub(rf"\b{re.escape(al)}\b", tk.lower(), t)
        else:
            t = t.replace(al, tk.lower())

    # (Optional) also normalize a few well-known out-of-scope names to their tickers
    # so questions like "apple revenue 2024" line up for structure checks (still OOS later).
    if 'OTHER_TICKER_ALIASES' in globals():
        for alias, tk in sorted(OTHER_TICKER_ALIASES.items(), key=lambda kv: -len(kv[0])):
            al = alias.lower()
            if len(al) <= 6:
                t = re.sub(rf"\b{re.escape(al)}\b", tk.lower(), t)
            else:
                t = t.replace(al, tk.lower())

    return t


def _normalize_question(text: str) -> str:
    if text is None:
        return ""
    # NEW: canonicalize company names → tickers BEFORE other cleanup
    text = _canonicalize_entities_for_match(str(text))

    t = unicodedata.normalize("NFKD", text)
    t = t.replace("’", "'").replace("“","\"").replace("”","\"")
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s$%./-]+", " ", t)   # keep numbers, $, %, slashes, hyphens
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _token_set(s: str) -> set:
    toks = [w for w in re.findall(r"[a-z0-9$%./-]+", s) if w and (w not in STOPWORDS)]
    return set(toks)

def _nl_similarity(a: str, b: str) -> float:
    # combine char-level and token-level similarity
    a_n = _normalize_question(a)
    b_n = _normalize_question(b)
    if not a_n and not b_n:
        return 0.0
    seq = SequenceMatcher(None, a_n, b_n).ratio()               # 0..1 (typo-friendly)
    ta, tb = _token_set(a_n), _token_set(b_n)                   # order-insensitive
    jacc = (len(ta & tb) / max(1, len(ta | tb))) if (ta or tb) else 0.0
    return 0.55*seq + 0.45*jacc

def _norm_method(m):
    # Accept either a single string or a pandas Series of strings
    if isinstance(m, pd.Series):
        s = m.fillna("").astype(str).str.lower().str.strip()
        s = s.str.replace(r"[\s\-_]+", "", regex=True)
        # unify all FT variants → "finetuned"
        s = s.replace({
            "ft": "finetuned",
            "finetune": "finetuned",
            "finetuned": "finetuned",
            "finetuning": "finetuned",
            "finetunedmodel": "finetuned",
            "finetunedmethod": "finetuned",
            "finetunemethod": "finetuned",
            "finetunedmodelmethod": "finetuned",
        })
        return s
    else:
        m = "" if m is None else str(m)
        m = re.sub(r"[\s\-_]+", "", m.lower().strip())
        if m in {
            "ft","finetune","finetuned","finetuning",
            "finetunedmodel","finetunedmethod","finetunemethod","finetunedmodelmethod"
        }:
            return "finetuned"
        return m

def _extract_struct(q: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    tkr, yr = parse_query(q)
    met = detect_metric(q)
    return tkr, yr, met

def confidence_from_csv(question: str, method_label: str, tol: float = 0.80) -> Optional[float]:
    """
    Robust lookup:
      - Normalize text (case/punct/space)
      - Token-set + SequenceMatcher similarity
      - Method-aware (prefer same method)
      - Structure bonus if ticker/year/metric match
      - Accept if score >= tol; soft-accept >= 0.74 with structure agreement
    """
    dfc = load_conf_table(CONF_CSV)
    if dfc is None or dfc.empty:
        return None

    want_method = _norm_method(method_label)

    # Prefer rows with same normalized method, but fall back to all rows
    same_method = dfc[_norm_method(dfc["Method"]) == want_method]
    cand_df = same_method if not same_method.empty else dfc

    q_norm = _normalize_question(question)
    user_t, user_y, user_m = _extract_struct(question)

    best_idx, best_score, best_conf = None, 0.0, None
    for idx, row in cand_df.iterrows():
        cand_q = row["Question"]
        cand_norm = _normalize_question(cand_q)

        # Fast exact match on normalized text
        if cand_norm == q_norm:
            try:
                return float(row["Confidence"])
            except Exception:
                pass  # fall through if malformed

        s_base = _nl_similarity(cand_q, question)

        # Structure bonus
        cand_t, cand_y, cand_m = _extract_struct(cand_q)
        struct_matches = 0
        if user_t and cand_t and (user_t == cand_t): struct_matches += 1
        if user_y and cand_y and (int(user_y) == int(cand_y)): struct_matches += 1
        if user_m and cand_m and (user_m == cand_m): struct_matches += 1

        bonus = 0.0
        if struct_matches == 1: bonus += 0.08
        if struct_matches == 2: bonus += 0.16
        if struct_matches == 3: bonus += 0.24

        # Method tiny bonus
        if _norm_method(row["Method"]) == want_method:
            bonus += 0.03

        score = s_base + bonus

        if score > best_score:
            best_score = score
            try:
                best_conf = float(row["Confidence"])
                best_idx = idx
            except Exception:
                best_conf = None
                best_idx = idx

    # Hard threshold
    if best_conf is not None and best_score >= tol:
        return best_conf


    # Soft acceptance with extra guardrails:
    if best_conf is not None:
        cand_q_best = cand_df.loc[best_idx, "Question"]
        user_tokens = _token_set(q_norm)
        cand_tokens = _token_set(_normalize_question(cand_q_best))
        jacc = (len(user_tokens & cand_tokens) / max(1, len(user_tokens | cand_tokens))) if (user_tokens or cand_tokens) else 0.0

        # Recompute structure overlap for best candidate
        ct, cy, cm = _extract_struct(cand_q_best)
        struct_overlap = 0
        if user_t and ct and user_t == ct: struct_overlap += 1
        if user_y and cy and int(user_y) == int(cy): struct_overlap += 1
        if user_m and cm and user_m == cm: struct_overlap += 1

        # If 2+ structured fields align, accept at lower similarity
        if struct_overlap >= 2 and best_score >= 0.60:
            return best_conf

        if best_score >= 0.74 and (struct_overlap >= 1 or jacc >= 0.60):
            return best_conf


    return None

# ------------------------------
# RAG (unchanged retrieval)
# ------------------------------
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@st.cache_resource(show_spinner=True)
def load_embedder():
    return SentenceTransformer(EMB_MODEL)

@st.cache_resource(show_spinner=True)
def load_cross_encoder():
    tok = AutoTokenizer.from_pretrained(RERANKER)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    return tok, model, dev

def load_sections(dirpath: str) -> List[Dict]:
    docs = []
    if not dirpath or not os.path.isdir(dirpath):
        return docs
    for fp in sorted(glob.glob(os.path.join(dirpath, "*.txt"))):
        base = os.path.basename(fp)
        if base.count("__") != 2:
            continue
        ticker, statement, tail = base.split("__")
        year = int(tail.rsplit(".", 1)[0])
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs.append({"ticker": ticker, "statement": statement, "year": year, "path": fp, "text": text})
    return docs

@st.cache_resource(show_spinner=True)
def build_rag_indexes(sections_dir: str):
    secs = load_sections(sections_dir) if sections_dir else []
    if not secs:
        return [], None, None, None, None
    embedder = load_embedder()
    texts = [s["text"] for s in secs]
    X = embedder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    X = np.asarray(X, dtype="float32")
    fa = faiss.IndexFlatIP(X.shape[1]); fa.add(X)
    tfidf = TfidfVectorizer(strip_accents="unicode", lowercase=True, max_df=0.95, min_df=1,
                            sublinear_tf=True, ngram_range=(1,2))
    Xsp = tfidf.fit_transform(texts)
    return secs, texts, X, fa, (tfidf, Xsp)

sections, texts, Xdense, faiss_index, sparse_pack = build_rag_indexes(SECTIONS_DIR)
tfidf, Xsp = (sparse_pack or (None, None))

def preprocess_query(q: str) -> str:
    q = (q or "").lower().strip()
    q = re.sub(r"[^a-z0-9 %$.,/-]+", " ", q)
    q = " ".join(w for w in q.split() if w not in STOPWORDS)
    return q

def dense_search(query: str, k: int = 30) -> List[Tuple[int, float]]:
    if not faiss_index: return []
    qv = load_embedder().encode([preprocess_query(query)], normalize_embeddings=True)
    sims, idxs = faiss_index.search(np.asarray(qv, dtype="float32"), k)
    return list(zip(idxs[0].tolist(), sims[0].tolist()))

def sparse_search(query: str, k: int = 30) -> List[Tuple[int, float]]:
    if tfidf is None: return []
    qv = tfidf.transform([preprocess_query(query)])
    sims = cosine_similarity(qv, Xsp)[0]
    top = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in top]

def _minmax(d: Dict[int, float]) -> Dict[int, float]:
    if not d: return {}
    vals = np.array(list(d.values()), dtype="float32")
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9: return {i: 0.0 for i in d}
    return {i: (s - lo) / (hi - lo) for i, s in d.items()}

def hybrid_retrieve(query: str, k_ctx: int = 12, alpha: float = 0.6) -> List[Dict]:
    if not sections: return []
    fetch_k = max(36, 72)
    d = dense_search(query, k=fetch_k); s = sparse_search(query, k=fetch_k)
    dsc = {i:sc for i,sc in d}; ssc = {i:sc for i,sc in s}
    dn, sn = _minmax(dsc), _minmax(ssc)
    all_ids = set(dsc) | set(ssc)
    fused = []
    for i in all_ids:
        fd, fs = dn.get(i,0.0), sn.get(i,0.0)
        score = alpha*fd + (1-alpha)*fs
        m = dict(sections[i]); m["dense_score"]=float(fd); m["sparse_score"]=float(fs); m["fused_score"]=float(score)
        fused.append(m)
    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused[:max(36, k_ctx*4)]

@st.cache_resource(show_spinner=True)
def _ce():
    return load_cross_encoder()

@torch.no_grad()
def rerank_with_cross_encoder(query: str, candidates: list, topk: int = 6) -> list:
    if not candidates: return []
    tok, model, dev = _ce()
    pairs = [(query, c["text"]) for c in candidates]
    batch = tok.batch_encode_plus(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
    logits = model(**batch).logits
    if logits.shape[-1] == 1:
        raw = logits.squeeze(-1); probs = torch.sigmoid(raw); sort_key = raw
    else:
        raw = logits[:,1]; probs = torch.softmax(logits, dim=-1)[:,1]; sort_key = raw
    order = torch.argsort(sort_key, descending=True).tolist()
    ranked = []
    for i in order[:topk]:
        item = dict(candidates[i]); item["rerank_prob"] = float(probs[i]); item["rerank_logit"]=float(raw[i])
        ranked.append(item)
    return ranked

def citations_from_contexts(ctxs, limit: int = 3):
    seen, cites = set(), []
    for c in ctxs:
        key = (c.get("path"), int(c.get("chunk_index", -1)))
        if key in seen: continue
        seen.add(key)
        base = os.path.basename(c["path"])
        cites.append(base)
        if len(cites) >= limit:
            break
    return cites

# ---- RAG confidence scaffold (kept for fallback)
def rag_confidence(contexts: list, structured_ok: bool, tkr: Optional[str], yr: Optional[int]) -> float:
    probs = [float(c.get("rerank_prob", 0.5)) for c in (contexts or [])[:5]]
    top_p = max(probs) if probs else 0.0
    mean_p3 = (sum(probs[:3]) / max(1, len(probs[:3]))) if probs else 0.0
    support = 0
    for c in (contexts or [])[:5]:
        ok_t = (tkr is None) or (c.get("ticker") == tkr)
        ok_y = (yr is None) or (int(c.get("year", -1)) == int(yr))
        if ok_t and ok_y:
            support += 1
    support_frac = support / max(1, min(5, len(contexts or [])))
    base = 0.35 + 0.30*top_p + 0.20*mean_p3 + 0.15*support_frac
    if structured_ok: base += 0.10
    return float(np.clip(base, 0.05, 0.98))

def answer_rag_pipeline(q: str, k_ctx:int=5, use_ce:bool=True) -> dict:

    if _has_out_of_scope_mention(q) or not _looks_financial(q):
        msg = _fallback_out_of_scope_message(_find_out_of_scope_mentions(q) or None)
        return {
            "result": {"answer": msg, "confidence": 0.10,
                       "ticker": None, "year": None, "metric": None, "value_norm": None},
            "contexts": []
        }
    # Safety guard: if not a finance question, don't try to retrieve
    if not _looks_financial(q):
        msg = _fallback_out_of_scope_message()
        return {
            "result": {
                "answer": msg, "confidence": 0.10,
                "ticker": None, "year": None, "metric": None,
                "value_norm": None
            },
            "contexts": []
        }

    tkr, yr = parse_query(q)
    metric = detect_metric(q)
    direct = None
    if tkr and yr and metric:
        direct = lookup_metric(df_fin, tkr, yr, metric)
    pool = hybrid_retrieve(q, k_ctx=max(12, k_ctx*2), alpha=0.6)
    ctxs = rerank_with_cross_encoder(q, pool, topk=k_ctx) if use_ce else pool[:k_ctx]
    if direct:
        ans_text = fmt_money(direct["value_norm"])
        confidence = rag_confidence(ctxs, structured_ok=True, tkr=tkr, yr=yr)
        return {"result": {"answer": ans_text, "confidence": float(confidence), "ticker": tkr, "year": yr, "metric": metric, "value_norm": float(direct["value_norm"])},
                "contexts": ctxs}
    bullets = []
    for c in ctxs[:min(3, len(ctxs))]:
        txt = c["text"].strip().split("\n")[0]
        bullets.append(f"• {txt[:220]}{'…' if len(txt)>220 else ''}")
    ans = "RAG summary:\n" + "\n".join(bullets) if bullets else "RAG: no contexts."
    confidence = rag_confidence(ctxs, structured_ok=False, tkr=tkr, yr=yr)
    return {"result": {"answer": ans, "confidence": float(confidence), "ticker": tkr, "year": yr, "metric": metric},
            "contexts": ctxs}


# ------------------------------
# FT (RESTORED to §4.2)
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_ft_components():
    """Load vocabs.json & minilm_moe_best.pt and build the exact MiniLM-MoE used in §4.2."""
    if not (VOCABS_JSON and FT_CKPT):
        return None

    with open(VOCABS_JSON, "r", encoding="utf-8") as f:
        voc = json.load(f)

    Z_MEAN = float(voc.get("z_mean", 0.0))
    Z_STD  = float(voc.get("z_std", 1.0)) or 1.0
    MAX_LEN = int(voc.get("max_len", 128))

    KEY_VOCAB    = voc.get("key_vocab")    or list((voc.get("key2id") or {}).keys())
    TICKER_VOCAB = voc.get("ticker_vocab") or list((voc.get("ticker2id") or {}).keys())
    YEAR_VOCAB   = voc.get("year_vocab")   or [int(k) for k in (voc.get("year2id") or {}).keys()]

    k2id = {k:i for i,k in enumerate(KEY_VOCAB)}
    t2id = {t:i for i,t in enumerate(TICKER_VOCAB)}
    y2id = {int(y):i for i,y in enumerate(YEAR_VOCAB)}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = AutoModel.from_pretrained(EMB_MODEL).to(dev)
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    d_model = enc.config.hidden_size

    class LoRAExpert(torch.nn.Module):
        def __init__(self, d_model: int, r: int = 32):
            super().__init__()
            self.down = torch.nn.Linear(d_model, r, bias=False)
            self.up   = torch.nn.Linear(r, d_model, bias=False)
            self.act  = torch.nn.GELU()
        def forward(self, h): return self.up(self.act(self.down(h)))

    class MoEAdapter(torch.nn.Module):
        def __init__(self, d_model: int, num_experts: int = 6, r: int = 64):
            super().__init__()
            self.experts = torch.nn.ModuleList([LoRAExpert(d_model, r=r) for _ in range(num_experts)])
            self.gate = torch.nn.Sequential(torch.nn.Linear(d_model, d_model//2), torch.nn.Tanh(), torch.nn.Linear(d_model//2, num_experts))
            self.drop = torch.nn.Dropout(0.1)
        def forward(self, h):
            gate = torch.softmax(self.gate(h), dim=-1)
            expert_outs = torch.stack([e(h) for e in self.experts], dim=1)
            delta = (gate.unsqueeze(-1) * expert_outs).sum(dim=1)
            return self.drop(delta), gate

    class MiniLM_MoE_Regressor(torch.nn.Module):
        def __init__(self, encoder, tokenizer, d_model: int, num_tickers: int, num_keys: int, num_years: int, tdim=16, kdim=8, ydim=8, num_experts=6, r=64):
            super().__init__()
            self.encoder = encoder; self.tokenizer = tokenizer
            self.emb_t = torch.nn.Embedding(max(1,num_tickers), tdim)
            self.emb_k = torch.nn.Embedding(max(1,num_keys),   kdim)
            self.emb_y = torch.nn.Embedding(max(1,num_years),  ydim)
            self.proj  = torch.nn.Linear(d_model + tdim + kdim + ydim, d_model)  # keep name for ckpt compat
            self.moe   = MoEAdapter(d_model=d_model, num_experts=num_experts, r=r)
            self.head  = torch.nn.Sequential(torch.nn.Linear(d_model, d_model//2), torch.nn.GELU(), torch.nn.Linear(d_model//2, 1))
            for p in self.encoder.parameters(): p.requires_grad = False
            self.use_moe = True

        def _mean_pool(self, last_hidden_state, attention_mask):
            mask = attention_mask.unsqueeze(-1).float()
            return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        @torch.no_grad()
        def forward(self, texts, t_idx, k_idx, y_idx, max_len=128, device="cpu"):
            enc_in = self.tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out = self.encoder(**enc_in, return_dict=True)
            pooled = self._mean_pool(out.last_hidden_state, enc_in["attention_mask"])  # (B, 384 for MiniLM)

            # side embeddings (safe defaults)
            et = self.emb_t(t_idx.to(device)) if self.emb_t.num_embeddings > 0 else torch.zeros(pooled.size(0), 0, device=device)
            ek = self.emb_k(k_idx.to(device)) if self.emb_k.num_embeddings > 0 else torch.zeros(pooled.size(0), 0, device=device)
            ey = self.emb_y(y_idx.to(device)) if self.emb_y.num_embeddings > 0 else torch.zeros(pooled.size(0), 0, device=device)

            h_cat = torch.cat([pooled, et, ek, ey], dim=-1)  # expected 384+16+8+8=416

            # defensive alignment to the projector’s expected input size
            want = getattr(self.proj, "in_features", h_cat.shape[-1])
            have = h_cat.shape[-1]
            if have != want:
                if have < want:
                    pad = want - have
                    h_cat = torch.cat([h_cat, torch.zeros(h_cat.size(0), pad, device=h_cat.device, dtype=h_cat.dtype)], dim=-1)
                else:
                    h_cat = h_cat[:, :want]

            h = self.proj(h_cat)

            # optional MoE
            if getattr(self, "use_moe", True):
                delta, _ = self.moe(h)
                h = h + delta

            z_pred = self.head(h).squeeze(-1)
            return z_pred


    model = MiniLM_MoE_Regressor(enc, tok, d_model=d_model,
                                 num_tickers=len(TICKER_VOCAB), num_keys=len(KEY_VOCAB), num_years=len(YEAR_VOCAB)).to(dev)
    state = torch.load(FT_CKPT, map_location=dev)
    state = state.get("model_state_dict", state)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # Fallback to non-strict so the app runs; forward-time padding/truncation handles proj width
        missing, unexpected = model.load_state_dict(state, strict=False)
        st.warning("Loaded FT checkpoint non-strictly (shapes differed). "
                    "Ensure your checkpoint matches encoder + side-embedding dims.")
    model.eval()

    
    return {
        "moe_z": model, "device": dev, "tok": tok,
        "Z_MEAN": Z_MEAN, "Z_STD": Z_STD, "MAX_LEN": MAX_LEN,
        "KEY_VOCAB": KEY_VOCAB, "TICKER_VOCAB": TICKER_VOCAB, "YEAR_VOCAB": YEAR_VOCAB,
        "key2id": {k:i for i,k in enumerate(KEY_VOCAB)},
        "ticker2id": {t:i for i,t in enumerate(TICKER_VOCAB)},
        "year2id": {int(y):i for i,y in enumerate(YEAR_VOCAB)}
    }

FT = load_ft_components()

def _from_log10_signed(z: float) -> float:
    return math.copysign(10.0**abs(float(z)) - 1.0, z)

@torch.no_grad()
def predict_usd_from_row_embY(row: dict) -> float:
    """EXACT §4.2 inference: z_norm -> de-norm -> signed-log inverse -> *scale."""
    assert FT, "FT components not loaded"
    dev    = FT["device"]
    moe_z  = FT["moe_z"]
    ZM, ZS = FT["Z_MEAN"], FT["Z_STD"]
    MAXLEN = FT["MAX_LEN"]
    t2i, k2i, y2i = FT["ticker2id"], FT["key2id"], FT["year2id"]

    t_idx = torch.tensor([t2i.get(row.get("ticker"), 0)], dtype=torch.long, device=dev)
    k_idx = torch.tensor([k2i.get(row.get("canon_key","revenue"), 0)], dtype=torch.long, device=dev)
    y_raw = row.get("year")
    y_idx = torch.tensor([y2i.get(int(y_raw), 0) if isinstance(y_raw, int) else 0], dtype=torch.long, device=dev)

    z_norm_pred = moe_z([row.get("q","")], t_idx, k_idx, y_idx, max_len=MAXLEN, device=dev).view(-1)[0].detach().cpu().item()
    z_pred = z_norm_pred * (ZS if ZS!=0 else 1.0) + ZM
    y_unscaled = _from_log10_signed(z_pred)
    return float(y_unscaled * float(row.get("scale", 1.0)))

# ---- FT confidence scaffold (kept for fallback)
_key_ranges_ft: Dict[str, Tuple[float,float]] = {}
def ft_confidence(row: dict, y_hat: float) -> float:
    base = 0.60
    rng = _key_ranges_ft.get(row.get("canon_key"))
    if rng:
        lo, hi = rng
        if (y_hat >= lo - 0.1*abs(lo)) and (y_hat <= hi + 0.1*abs(hi)):
            base += 0.15
        else:
            base -= 0.15
    if row.get("canon_key") in {"revenue","total assets","total liabilities","gross profit","operating cash flow"}:
        base += 0.05
    return float(np.clip(base, 0.30, 0.95))

def answer_ft(
    ticker: Optional[str],
    year: Optional[int],
    metric: Optional[str],
    raw_query: str = "",
    use_moe: bool = True,
) -> dict:
    """
    Fine-tuned (§4.2) inference with strict in-scope guards and optional MoE.
    Never returns a 'gibberish' numeric answer when the query is not in scope.
    """

    # ---------- 1) Hard scope / intent guards ----------
    # (a) obvious out-of-scope tickers in the text (e.g., AAPL) or an explicit unsupported ticker
    offenders = _find_unsupported_tickers(raw_query or "")
    if ticker and ticker not in SUPPORTED_TICKERS:
        offenders = offenders or [ticker]

    if offenders:
        return {
            "answer": _fallback_out_of_scope_message(offenders),
            "y_hat": None, "confidence": 0.10, "secs": 0.0,
        }

    # (b) non-financial question (e.g., "capital of France?")
    if not _looks_financial(raw_query):
        return {
            "answer": _fallback_out_of_scope_message(None),
            "y_hat": None, "confidence": 0.10, "secs": 0.0,
        }

    # (c) require a supported ticker before running FT
    if not ticker or ticker not in SUPPORTED_TICKERS:
        return {
            "answer": _fallback_out_of_scope_message(None),
            "y_hat": None, "confidence": 0.10, "secs": 0.0,
        }

    # ---------- 2) Metric detection / normalization ----------
    metric_l = (metric or detect_metric(raw_query) or "").strip().lower()
    if not metric_l:
        # Looks financial & has (ticker/year) but no recognizable metric
        return {
            "answer": _fallback_unknown_metric_message(ticker, year),
            "y_hat": None, "confidence": 0.20, "secs": 0.0,
        }

    scale = ITEM_SCALE.get(metric_l, 1.0)

    # ---------- 3) Model availability ----------
    if not FT:
        return {
            "answer": "FT model not available",
            "y_hat": None, "confidence": 0.0, "secs": 0.0,
        }

    # Try to toggle MoE if the loaded model exposes the flag
    try:
        if "moe_z" in FT and hasattr(FT["moe_z"], "use_moe"):
            FT["moe_z"].use_moe = bool(use_moe)
    except Exception:
        pass  # safe to ignore; fall back to whatever the checkpoint supports

    # ---------- 4) Build inference row and predict ----------
    year_tag = f"[YEAR={int(year)}]" if isinstance(year, int) else "[YEAR=?]"
    tkr_tag  = f"[TICKER={ticker}]" if ticker else "[TICKER=?]"
    q_aug = f"{tkr_tag} {year_tag} [ITEM={metric_l}] {raw_query}".strip()

    row = {
        "q": q_aug,
        "ticker": ticker,
        "year": int(year) if isinstance(year, int) else None,
        "canon_key": metric_l,
        "scale": scale,
    }

    t0 = time.time()
    try:
        y_hat = predict_usd_from_row_embY(row)
        secs = time.time() - t0
        confidence = ft_confidence(row, y_hat)
        return {
            "answer": fmt_money(y_hat),
            "y_hat": y_hat,
            "confidence": confidence,
            "secs": round(secs, 3),
        }

    except RuntimeError as e:
        # Typical when the checkpoint was trained w/ a different projection width (e.g., 384 vs 416)
        secs = time.time() - t0
        msg = (
            "Fine-tuned model shape mismatch. "
            "Ensure the checkpoint matches the current architecture (encoder hidden size and "
            "embedding dims for ticker/key/year), or disable 'FT: Mixture-of-Experts' in Controls."
        )
        return {"answer": msg, "y_hat": None, "confidence": 0.0, "secs": round(secs, 3)}

    except Exception as e:
        secs = time.time() - t0
        return {"answer": f"Fine-tuned model error: {e}", "y_hat": None, "confidence": 0.0, "secs": round(secs, 3)}


# FT correctness thresholds from §4.2
FT_TOL_REL = 0.23
FT_TOL_ABS = 5e7
def ft_is_correct(y_hat: Optional[float], gt: Optional[float]) -> bool:
    if gt is None or y_hat is None: return False
    denom = max(1.0, abs(y_hat), abs(gt))
    return abs(y_hat - gt) <= max(FT_TOL_REL * denom, FT_TOL_ABS)

def extract_numeric_from_text(text: str) -> Optional[float]:
    m = re.search(r"\$([0-9]+(?:\.[0-9]+)?)\s*(thousand|million|billion|trillion)?", str(text), flags=re.I)
    if not m: return None
    x = float(m.group(1))
    mul = {"thousand":1e3,"million":1e6,"billion":1e9,"trillion":1e12}.get((m.group(2) or "").lower(), 1.0)
    return x * mul

# ------------------------------
# UI
# ------------------------------
st.title("Financial QA — RAG vs Fine-Tuning (Companies - ADP, AAGH, AFRM & AEHR)")

st.sidebar.header("Controls")

# Method picker
# Method picker (unchanged)
method = st.sidebar.radio(
    "Method",
    ["RAG"] + (["FT"] if FT else []),
    index=0,
    key="method",
)

# RAG: Cross Encoder – no value=, rely on ss["rag_use_ce"]
st.sidebar.checkbox(
    "RAG: Cross encoder rank",
    disabled=(method != "RAG"),
    key="rag_use_ce",
)

# FT: MoE – no value=, rely on ss["ft_use_moe"]
st.sidebar.checkbox(
    "FT: Advanced technique — Mixture-of-Experts",
    disabled=(method != "FT" or not FT),
    key="ft_use_moe",
)

# Effective flags (unchanged)
use_ce_effective = bool(ss.get("rag_use_ce", True)) if method == "RAG" else False
ft_use_moe_effective = bool(ss.get("ft_use_moe", True)) if (method == "FT" and FT) else False



# --- Out-of-scope & finance-intent helpers (fixed) ---

# Phrases we accept as clear finance intent (checked as substrings in lowercase)
FINANCE_HINT_PHRASES = [
    "net income", "operating income", "gross profit",
    "operating cash flow", "cash flow",
    "balance sheet", "income statement",
    "total assets", "total liabilities", "shareholders equity",
    "capital expenditure", "capital expenditures",
]

# Single words we accept as finance intent (checked with word boundaries)
# IMPORTANT: no generic "capital", "net", "income" here to avoid false positives.
FINANCE_HINT_WORDS = {
    "revenue", "sales", "ebitda", "ebit", "equity",
    "assets", "liabilities", "dividend", "eps", "margin",
    "opex", "cogs", "depreciation", "amortization", "ppe",
    "inventory", "receivable", "payable", "guidance",
    "yoy", "qoq", "fiscal", "10k", "10-q", "cashflow", "cash-flow",
}

_TICKER_TOKEN_RE = re.compile(r"\b[A-Z]{1,5}\b")
_CAPS_IGNORE = {
    "USD","EUR","GBP","EPS","EBITDA","GAAP","CAGR","FY","Q1","Q2","Q3","Q4",
    "CEO","CFO","PPE","COGS","SGA","DCF","WACC","FCF","OCF","BS","CF","IS",
    "JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"
}

def _find_unsupported_tickers(text: str):
    """Return any ALL-CAPS tokens in the text that are not in our SUPPORTED_TICKERS."""
    caps = set(_TICKER_TOKEN_RE.findall(text or ""))
    return sorted({t for t in caps if t not in SUPPORTED_TICKERS and t not in _CAPS_IGNORE})

def _looks_financial(text: str) -> bool:
    """Conservative finance intent detector (no 'capital' false positives)."""
    if not text or not text.strip():
        return False
    tl = text.lower()

    # Clear metric detection
    if detect_metric(text) is not None:
        return True

    # Mentions a supported ticker explicitly?
    if any(re.search(rf"\b{re.escape(t)}\b", text) for t in SUPPORTED_TICKERS):
        return True

    # Finance phrases (substring)
    if any(p in tl for p in FINANCE_HINT_PHRASES):
        return True

    # Finance single words with boundaries
    if any(re.search(rf"\b{re.escape(w)}\b", tl) for w in FINANCE_HINT_WORDS):
        return True

    return False

def _fallback_out_of_scope_message(unsupported=None) -> str:
    scope = "ADP, AAGH, AFRM, and AEHR"
    if unsupported:
        who = ", ".join(unsupported)
        lead = f"Out of Scope as this app only covers {scope} companies. Your question mentions {who}."
    else:
        lead = f"Out of Scope as this app only covers {scope} companies and common financial metrics."
    return (
        f"{lead}\n\n"
        "Try questions like:\n"
        "• “What was ADP’s gross profit in 2023?”\n"
        "• “AFRM revenue 2024?”\n"
        "• “AEHR operating cash flow 2023?”"
    )

def _fallback_unknown_metric_message(tkr: Optional[str], yr: Optional[int]) -> str:
    common = [
        "revenue", "gross profit", "net income", "operating income",
        "operating cash flow", "total assets", "total liabilities",
        "shareholders equity", "capex", "cash & equivalents"
    ]
    shown = ", ".join(common)
    who = f"{tkr or 'the company'}"
    when = f" in {yr}" if yr else ""
    return (
        f"I couldn’t recognize a financial metric in your question for {who}{when}.\n\n"
        f"Try one of these: {shown}"
    )

def _alias_hit(text_lc: str, alias: str) -> bool:
    # word-boundary for short aliases; substring for long
    if len(alias) <= 6:
        return re.search(rf"\b{re.escape(alias)}\b", text_lc) is not None
    return alias in text_lc

def _find_out_of_scope_mentions(text: str) -> list:
    """Return names/tickers in the text that map to tickers NOT in SUPPORTED_TICKERS."""
    text_lc = (text or "").lower()

    # 1) ALL-CAPS ticker tokens (existing behavior)
    caps_oos = _find_unsupported_tickers(text)

    # 2) Natural-language aliases (Apple, Microsoft, etc.)
    oos = set(caps_oos)
    for alias, tk in OTHER_TICKER_ALIASES.items():
        if _alias_hit(text_lc, alias) and tk not in SUPPORTED_TICKERS:
            # prefer a readable label in the message
            label = alias if alias.isalpha() else tk
            oos.add(label.capitalize())

    return sorted(oos)

def _has_out_of_scope_mention(text: str) -> bool:
    return bool(_find_out_of_scope_mentions(text))



colM, colR = st.columns([MID_FRAC, 1.0 - MID_FRAC])

with colM:
    st.subheader("Question")
    q = st.text_area(
        label="",
        placeholder="Type your question here",
        key="q_input",
        label_visibility="collapsed",
        height=1,
    )
    go = st.button("Answer", type="primary")

    if go:
        # ---- EARLY GUARDS (prevent gibberish / irrelevant RAG) ----
        oos_mentions = _find_out_of_scope_mentions(q)
        if oos_mentions:
            ar = {
                "method": method,
                "answer": _fallback_out_of_scope_message(oos_mentions),
                "confidence": 0.10,
                "time_total": 0.0,
                "ticker": None, "year": None, "metric": None,
                "citations": [], "correct": False,
            }
        elif not _looks_financial(q):
            ar = {
                "method": method,
                "answer": _fallback_out_of_scope_message(),
                "confidence": 0.10,
                "time_total": 0.0,
                "ticker": None, "year": None, "metric": None,
                "citations": [], "correct": False,
            }
        else:
    # (unchanged RAG/FT handling)

            # If it looks financial but metric is missing, suggest supported ones
            tkr_pre, yr_pre = parse_query(q)
            met_pre = detect_metric(q)
            if (tkr_pre or yr_pre) and met_pre is None:
                ar = {
                    "method": method,
                    "answer": _fallback_unknown_metric_message(tkr_pre, yr_pre),
                    "confidence": 0.20,
                    "time_total": 0.0,
                    "ticker": tkr_pre, "year": yr_pre, "metric": None,
                    "citations": [], "correct": False,
                }
            else:
                # ---- ORIGINAL RAG / FT LOGIC (unchanged) ----
                if method == "RAG":
                    t0 = time.time()
                    out = answer_rag_pipeline(q, k_ctx=5, use_ce=use_ce_effective)
                    r = out["result"]
                    gt = _gt_val_only(r.get("ticker"), r.get("year"), r.get("metric"))
                    pred = r.get("value_norm")
                    if pred is None:
                        pred = extract_numeric_from_text(r.get("answer",""))
                    has_numeric = pred is not None
                    denom = max(1.0, abs(pred or 0.0), abs(gt or 0.0))
                    correct = bool(gt is not None and pred is not None and abs((pred or 0) - gt)/denom <= 0.02)  # strict ±2%
                    ar = {
                        "method":"RAG","answer": r["answer"],"confidence": float(r["confidence"]),
                        "time_total": time.time()-t0, "ticker": r.get("ticker"), "year": r.get("year"),
                        "metric": r.get("metric"), "citations": out.get("contexts") and citations_from_contexts(out["contexts"]) or [],
                        "correct": correct,
                        "has_numeric": has_numeric,
                    }
                else:
                    tkr, yr = parse_query(q)
                    met = detect_metric(q) or "revenue"
                    if not FT:
                        ar = {"method":"FT","answer":"FT model not available","confidence":0.0,"time_total":0.0,
                              "ticker":tkr,"year":yr,"metric":met,"citations":[], "correct":False}
                    else:
                        out = answer_ft(tkr, yr, met, raw_query=q, use_moe=ft_use_moe_effective)
                        gt = _gt_val_only(tkr, yr, met)
                        correct = ft_is_correct(out["y_hat"], gt)
                        ar = {
                            "method":"FT","answer": out["answer"],"confidence": float(out["confidence"]),
                            "time_total": float(out["secs"]), "ticker": tkr, "year": yr, "metric": met,
                            "citations": [], "correct": bool(correct),
                            "has_numeric": out.get("y_hat") is not None,
                        }

        # Keep your CSV confidence override, but don’t override fallback messages
        csv_conf = confidence_from_csv(q, ar["method"])
        if csv_conf is not None and ar.get("has_numeric"):
            ar["confidence"] = float(csv_conf)

        ss["last_answer"] = ar
        ss["rows"].append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": ar["method"], "confidence": float(ar["confidence"]),
            "total_s": round(float(ar["time_total"]),3),
            "ticker": ar.get("ticker"), "year": ar.get("year"), "metric": ar.get("metric"),
            "citations": ", ".join(ar.get("citations") or []), "correct": bool(ar.get("correct")),
            "answer": ar.get("answer","")
        })


    ar = ss["last_answer"]
    m1, m2, m3, m4 = st.columns(4)
    if ar:
        conf_val = ar.get("confidence")
        m1.metric("Confidence", f"{conf_val:.2f}" if conf_val is not None else "--")
        m2.metric("Time (s)", f"{ar['time_total']:.2f}")
        m3.metric("Method", ar["method"])
        m4.metric("Correct", "Y" if ar.get("correct") else "N")
        st.subheader("Answer")
        st.markdown(f"<div class='qa-box'>{ar['answer']}</div>", unsafe_allow_html=True)
    else:
        m1.metric("Confidence", "--"); m2.metric("Time (s)", "--"); m3.metric("Method", method); m4.metric("Correct", "—")
        st.subheader("Answer")
        st.markdown(f"<div class='qa-box'><span class='smallgray'>Press <b>Answer</b> to run a query.</span></div>", unsafe_allow_html=True)

    # Extended Evaluation tables
    if ss.get("eval_msg"):
        st.caption(ss["eval_msg"])
    if ss.get("eval_tables"):
        df_ext, df_sum, df_pair = ss["eval_tables"]
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.write("**Extended Evaluation (RAG vs Fine-Tuned)**")
        st.dataframe(df_ext, use_container_width=True, height=min(500, 42+24*min(len(df_ext), 12)))
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.write("**Summary**")
        st.dataframe(df_sum, use_container_width=True, height=min(260, 42+24*min(len(df_sum), 8)))
        if df_pair is not None and not df_pair.empty:
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            st.write("**Pairwise (RAG vs Fine-Tuned)**")
            st.dataframe(df_pair, use_container_width=True, height=min(260, 42+24*min(len(df_pair), 8)))

with colR:
    st.header("Session Actions")

    # Toggle: Show / Hide Session Results Table
    if st.button("Hide Session Results Table" if ss.get("show_table", False) else "Show Session Results Table"):
        ss["show_table"] = not ss.get("show_table", False)

    # Build dataframe once if we have rows
    if ss["rows"]:
        dfres = pd.DataFrame(ss["rows"])

        # Render table when toggled on
        if ss.get("show_table", False):
            st.dataframe(
                dfres,
                use_container_width=True,
                height=min(420, 42 + 24 * min(len(dfres), 12))
            )

        # Export buttons
        st.download_button(
            "Export Session Results (CSV)",
            data=dfres.to_csv(index=False).encode(),
            file_name="session_results.csv",
            mime="text/csv",
        )
        st.download_button(
            "Export Session Results (HTML)",
            data=dfres.to_html(index=False),
            file_name="session_results.html",
            mime="text/html",
        )
    else:
        # No rows yet
        if ss.get("show_table", False):
            st.info("No rows yet — run a query first.")
        st.button("Export Session Results (CSV + HTML)", disabled=True)

    if st.button("Clear / Reset Session"):
        ss["_do_reset"] = True
        st.rerun()

    st.divider()

    clicked_eval = st.button("Extended Evaluation (RAG vs Fine-Tuned)")

    def run_save_script_and_load():
        base_dir = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
        script = os.path.join(base_dir, "save_table.py")
        if not os.path.exists(script):
            ss["eval_msg"] = "save_table.py not found next to app.py."
            return
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            proc = subprocess.run(
                [sys.executable, script],
                cwd=base_dir,
                capture_output=True,
                text=True,
                shell=False,
                timeout=1200,
                env=env,
            )
            ss["save_log"] = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode != 0:
                ss["eval_msg"] = "save_table.py exited with an error."
                return
        except Exception as e:
            ss["eval_msg"] = f"Error running save_table.py: {e}"
            return

        p1 = in_out("df_extended_view.csv")
        p2 = in_out("summary_42.csv")
        p3 = in_out("pair_42.csv")

        if not p1:
            ss["eval_msg"] = "Extended outputs not found after running save_table.py."
            ss["eval_tables"] = None
            return

        try:
            df_ext = pd.read_csv(p1)
            df_sum = pd.read_csv(p2) if p2 else pd.DataFrame()
            df_pair = pd.read_csv(p3) if p3 else pd.DataFrame()
            ss["eval_tables"] = (df_ext, df_sum, df_pair)
            ss["eval_msg"] = "Loaded Extended Evaluation outputs."
        except Exception as e:
            ss["eval_msg"] = f"Failed to load CSVs: {e}"
            ss["eval_tables"] = None

    if clicked_eval:
        ss["eval_tables"] = None
        ss["eval_msg"] = ""
        run_save_script_and_load()
        st.rerun()

    if ss.get("save_log"):
        with st.expander("save_table.py logs"):
            st.text(ss["save_log"])
