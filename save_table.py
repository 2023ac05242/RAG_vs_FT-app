#!/usr/bin/env python
# coding: utf-8
"""
save_table.py — produce out/df_extended_view.csv (RAG vs Fine-Tuned evaluation table)

What this script does:
  1) Loads per-year section text files from out/sections_text_yearly/*.txt
  2) Chunks them, builds dense (FAISS) and sparse (TF-IDF) indices
  3) Sets up a hybrid retrieval pipeline + cross-encoder reranker
  4) Loads structured financials (out/financials_last2y.csv) for exact numeric lookups
  5) Loads Q/A pairs (out/qa_pairs.jsonl) to build a small FT evaluation set
  6) Tries to load a fine-tuned MiniLM-MoE checkpoint (out/minilm_moe_best.pt).
     If not found, falls back to a kNN baseline.
  7) Runs an extended evaluation and writes:
        - out/df_extended_view.csv
        - out/summary_42.csv
        - out/pair_42.csv

You can safely run this standalone (python save_table.py).
"""

import os
import re
import glob
import math
import json
import time
import uuid
import random
import warnings
from hashlib import blake2b
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Optional helpers
try:
    import pdfplumber  # not used in this script flow, but harmless if installed
except Exception:
    pdfplumber = None

# Third-party ML libs
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Quiet noisy logs/warnings
os.environ["TQDM_NOTEBOOK"] = "0"
warnings.filterwarnings(
    "ignore",
    message=r".*`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning,
    module=r"transformers\.tokenization_utils_base",
)

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# -----------------------------
# 0) Device & paths
# -----------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("[info] CUDA:", torch.cuda.get_device_name(0))
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[info] Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("[info] CPU")

OUT_DIR = "out"
SECTIONS_DIR = os.path.join(OUT_DIR, "sections_text_yearly")
FIN_PATH = os.path.join(OUT_DIR, "financials_last2y.csv")
QA_PATH = os.path.join(OUT_DIR, "qa_pairs.jsonl")
CKPT_BEST = os.path.join(OUT_DIR, "minilm_moe_best.pt")
VOCABS_PATH = os.path.join(OUT_DIR, "vocabs.json")

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # or "intfloat/e5-small-v2"
RERANKER  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# knobs
FUSE_RETURN = 36
FUSE_FETCH_MULT = 2

# -----------------------------
# 1) Utilities
# -----------------------------
def ensure_paths():
    os.makedirs(OUT_DIR, exist_ok=True)
    assert os.path.exists(FIN_PATH), f"Missing {FIN_PATH}"
    assert os.path.isdir(SECTIONS_DIR) and glob.glob(os.path.join(SECTIONS_DIR, "*.txt")), \
        f"No section files found in {SECTIONS_DIR}"
    assert os.path.exists(QA_PATH), f"Missing {QA_PATH}"

def load_sections(dirpath: str) -> List[Dict]:
    docs = []
    for fp in sorted(glob.glob(os.path.join(dirpath, "*.txt"))):
        base = os.path.basename(fp)  # e.g., AAPL__income_statement__2024.txt
        if base.count("__") != 2:
            continue
        ticker, statement, tail = base.split("__")
        year = int(tail.rsplit(".", 1)[0])
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs.append({"ticker": ticker, "statement": statement, "year": year, "path": fp, "text": text})
    return docs

def stable_id(*parts) -> str:
    h = blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore")); h.update(b"|")
    return h.hexdigest()

# money formatting
def fmt_money(val: float) -> str:
    v = float(val); a = abs(v)
    sgn = "-" if v < 0 else ""
    if a >= 1e12: return f"{sgn}${a/1e12:.2f} trillion"
    if a >= 1e9:  return f"{sgn}${a/1e9:.2f} billion"
    if a >= 1e6:  return f"{sgn}${a/1e6:.2f} million"
    if a >= 1e3:  return f"{sgn}${a/1e3:.0f} thousand"
    return f"{sgn}${a:.0f}"

# -----------------------------
# 2) Token-aware chunking
# -----------------------------
tok = AutoTokenizer.from_pretrained(EMB_MODEL)

def _decode(ids):
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

def num_tokens(text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))

def chunk_by_tokens(text: str, max_tokens: int, overlap_tokens: int = 20):
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text.strip()] if text.strip() else []
    # light sentence split
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z(])', text.strip()) or [text.strip()]
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        s_ids = tok.encode(s, add_special_tokens=False)
        s_len = len(s_ids)
        # wrap very long sentences
        if s_len > max_tokens:
            start = 0
            while start < s_len:
                end = min(s_len, start + max_tokens)
                piece = _decode(s_ids[start:end]).strip()
                if piece:
                    chunks.append(piece)
                start = max(end - overlap_tokens, end)
            continue
        # pack window
        if cur_len + s_len <= max_tokens:
            cur.append(s); cur_len += s_len
        else:
            if cur:
                chunks.append(" ".join(cur).strip())
            if overlap_tokens > 0 and chunks:
                prev_ids = tok.encode(chunks[-1], add_special_tokens=False)
                tail     = _decode(prev_ids[-overlap_tokens:])
                cur, cur_len = [tail, s], len(prev_ids[-overlap_tokens:]) + s_len
            else:
                cur, cur_len = [s], s_len
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if c and num_tokens(c) <= max_tokens + 5]

def build_chunks(sections: List[Dict], sizes=(100, 400), overlap=20) -> List[Dict]:
    out = []
    for doc in sections:
        for max_toks in sizes:
            pieces = chunk_by_tokens(doc["text"], max_tokens=max_toks, overlap_tokens=overlap)
            for j, txt in enumerate(pieces):
                cid = stable_id(doc["ticker"], doc["statement"], doc["year"], max_toks, j)
                out.append({
                    "id": cid,
                    "text": txt,
                    "ticker": doc["ticker"],
                    "statement": doc["statement"],
                    "year": int(doc["year"]),
                    "source": doc["path"],
                    "chunk_size": max_toks,
                    "chunk_index": j,
                    "n_tokens": num_tokens(txt),
                })
    return out

# -----------------------------
# 3) Embeddings + Indices
# -----------------------------
def build_indices(chunks: List[Dict]):
    texts = [c["text"] for c in chunks]
    embedder = SentenceTransformer(EMB_MODEL)
    print("[info] embedding model:", EMB_MODEL)
    X = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    X = np.asarray(X, dtype="float32")
    print("[info] embeddings:", X.shape)

    faiss_index = faiss.IndexFlatIP(X.shape[1])
    faiss_index.add(X)
    print("[info] FAISS ntotal:", faiss_index.ntotal)

    tfidf = TfidfVectorizer(strip_accents="unicode", lowercase=True, max_df=0.95, min_df=1, sublinear_tf=True, ngram_range=(1,2))
    Xsp = tfidf.fit_transform(texts)
    print("[info] TF-IDF matrix:", Xsp.shape)
    return embedder, texts, faiss_index, tfidf, Xsp

# -----------------------------
# 4) Hybrid retrieval + CE rerank
# -----------------------------
STOPWORDS = set("""
a an and are as at be but by for if in into is it its of on or such that the their then there these this to with
""".split())

def preprocess_query(q: str) -> str:
    q = (q or "").lower().strip()
    q = re.sub(r"[^a-z0-9 %$.,/-]+", " ", q)
    q = " ".join(w for w in q.split() if w not in STOPWORDS)
    return q

def _maybe_prefix_query(q: str) -> str:
    return f"query: {q}" if "e5" in EMB_MODEL.lower() else q

def build_company_maps(df_fin: pd.DataFrame):
    t2c, c2t = {}, {}
    for t, grp in df_fin.groupby("ticker"):
        name = str(grp["company"].iloc[0]).strip()
        if name:
            t2c[t] = name
            c2t[name.lower()] = t
    return t2c, c2t

def parse_query_factory(name_aliases: Dict[str,str], company_name_to_ticker: Dict[str,str], known_tickers: set):
    def parse_query(query: str):
        q_raw = query or ""
        q_low = q_raw.lower()
        # explicit uppercase ticker
        m = re.findall(r"\b([A-Z]{2,6})\b", q_raw)
        ticker = None
        for t in m:
            if not known_tickers or t in known_tickers:
                ticker = t; break
        # alias/company matches
        if not ticker:
            for alias, tk in name_aliases.items():
                if len(alias) <= 6:
                    if re.search(rf"\b{re.escape(alias)}\b", q_low):
                        ticker = tk; break
                else:
                    if alias in q_low:
                        ticker = tk; break
        if not ticker:
            for nm, tk in company_name_to_ticker.items():
                if nm in q_low:
                    ticker = tk; break
        m_year = re.search(r"\b(20\d{2})\b", q_raw)
        year = int(m_year.group(1)) if m_year else None
        return ticker, year
    return parse_query

def canonicalize_query_entities_factory(NAME_ALIASES, TICKER_TO_COMPANY, parse_query):
    def canonicalize_query_entities(q: str) -> str:
        q_raw = q or ""
        q_low = q_raw.lower()
        ticker = None
        if parse_query:
            t, _ = parse_query(q_raw)
            ticker = t
        if not ticker:
            for alias, tk in NAME_ALIASES.items():
                if (len(alias) <= 6 and re.search(rf"\b{re.escape(alias)}\b", q_low)) or (len(alias) > 6 and alias in q_low):
                    ticker = tk; break
        if ticker:
            company = TICKER_TO_COMPANY.get(ticker, "")
            prefix = f"{ticker} {company}".strip()
            for alias, tk in NAME_ALIASES.items():
                if tk == ticker:
                    q_low = re.sub(rf"\b{re.escape(alias)}\b", "", q_low)
            q_low = re.sub(r"\s+", " ", q_low).strip()
            return f"{prefix} {q_low}".strip()
        return q_raw
    return canonicalize_query_entities

def augment_passage_for_rerank_factory(TICKER_TO_COMPANY):
    def augment_passage_for_rerank(p: dict) -> str:
        ticker = p.get("ticker", "")
        company = TICKER_TO_COMPANY.get(ticker, "")
        header = f"{ticker} {company}".strip()
        text = p.get("text", "")
        return (header + "\n" + text).strip()
    return augment_passage_for_rerank

def build_search_functions(embedder, faiss_index, tfidf, Xsp, meta, parse_query, canonicalize_query_entities):
    def dense_search(query: str, k: int = 30) -> List[Tuple[int, float]]:
        q_can = canonicalize_query_entities(query)
        qn = _maybe_prefix_query(preprocess_query(q_can))
        qv = embedder.encode([qn], normalize_embeddings=True)
        sims, idxs = faiss_index.search(np.asarray(qv, dtype="float32"), k)
        return list(zip(idxs[0].tolist(), sims[0].tolist()))

    def sparse_search(query: str, k: int = 30) -> List[Tuple[int, float]]:
        q_can = canonicalize_query_entities(query)
        qv = tfidf.transform([preprocess_query(q_can)])
        sims = cosine_similarity(qv, Xsp)[0]
        top = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i])) for i in top]

    def _minmax(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores: return {}
        vals = np.array(list(scores.values()), dtype="float32")
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-9:
            return {i: 0.0 for i in scores}
        return {i: (s - lo) / (hi - lo) for i, s in scores.items()}

    def hybrid_retrieve(query: str, k_ctx: int = 12, alpha: float = 0.6) -> List[Dict]:
        fetch_k = max(FUSE_RETURN, int(FUSE_FETCH_MULT * FUSE_RETURN))
        d = dense_search(query, k=fetch_k)
        s = sparse_search(query, k=fetch_k)
        d_scores = {i: sc for i, sc in d}
        s_scores = {i: sc for i, sc in s}
        d_n, s_n = _minmax(d_scores), _minmax(s_scores)
        all_ids = set(d_scores) | set(s_scores)
        fused = []
        for i in all_ids:
            fd, fs = d_n.get(i, 0.0), s_n.get(i, 0.0)
            score = alpha*fd + (1-alpha)*fs
            m = dict(meta[i])
            m["dense_score"] = float(fd)
            m["sparse_score"] = float(fs)
            m["fused_score"]  = float(score)
            fused.append(m)
        fused.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused[:FUSE_RETURN]
    return dense_search, sparse_search, hybrid_retrieve

def build_reranker(augment_passage_for_rerank, canonicalize_query_entities):
    ce_tok   = AutoTokenizer.from_pretrained(RERANKER)
    ce_model = AutoModelForSequenceClassification.from_pretrained(RERANKER).to(DEVICE)
    ce_model.eval()

    @torch.no_grad()
    def rerank_with_cross_encoder(query: str, candidates: List[Dict], topk: int = 6, return_probs: bool = True) -> List[Dict]:
        if not candidates:
            return []
        q_can = canonicalize_query_entities(query)
        pairs = [(q_can, augment_passage_for_rerank(c)) for c in candidates]
        batch = ce_tok.batch_encode_plus(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        logits = ce_model(**batch).logits
        if logits.shape[-1] == 1:
            raw = logits.squeeze(-1)
            probs = torch.sigmoid(raw) if return_probs else None
            sort_key = raw
        else:
            raw = logits[:, 1]
            probs = torch.softmax(logits, dim=-1)[:, 1] if return_probs else None
            sort_key = raw
        order = torch.argsort(sort_key, descending=True).tolist()
        ranked = []
        for i in order[:topk]:
            item = dict(candidates[i])
            item["rerank_logit"] = float(raw[i])
            if return_probs:
                item["rerank_prob"] = float(probs[i])
            ranked.append(item)
        return ranked

    return rerank_with_cross_encoder

# -----------------------------
# 5) Structured lookup + helpers
# -----------------------------
METRIC_TO_STMT = {
    "revenue": "income_statement",
    "gross profit": "income_statement",
    "net income": "income_statement",
    "operating income": "income_statement",
    "total assets": "balance_sheet",
    "total liabilities": "balance_sheet",
    "shareholders equity": "balance_sheet",
    "operating cash flow": "cash_flow",
}

def detect_metric(query: str) -> Optional[str]:
    q = (query or "").lower()
    for key in ["operating cash flow", "gross profit", "net income", "operating income",
                "total assets", "total liabilities", "shareholders equity"]:
        if key in q:
            return key
    if "revenue" in q or "sales" in q:
        return "revenue"
    return None

METRIC_PATTERNS = {
    "revenue":               [r"^revenue$", r"^total revenue$", r"net sales", r"sales revenue"],
    "net income":            [r"^net income$", r"net profit", r"profit attributable", r"earnings$"],
    "gross profit":          [r"^gross profit$"],
    "operating income":      [r"^operating income$", r"^operating profit$"],
    "operating cash flow":   [r"^net cash from operating activities$", r"operating cash flow", r"cash flow from operations"],
    "total assets":          [r"^total assets$"],
    "total liabilities":     [r"^total liabilities$"],
    "shareholders equity":   [r"^shareholders'? equity$", r"^stockholders'? equity$", r"^total equity$"],
}

def citations_from_contexts(ctxs, limit: int = 3):
    seen, cites = set(), []
    for c in ctxs:
        key = (c.get("source"), int(c.get("chunk_index", -1)))
        if key in seen:
            continue
        seen.add(key)
        base = os.path.basename(c["source"])
        cites.append(f"{base}#chunk{c['chunk_index']}")
        if len(cites) >= limit:
            break
    return cites

# Unit normalization
_UNIT_SCALE_PATTERNS = [
    (r"\b(thousand|k|000s)\b", 1e3),
    (r"\b(million|mn|mm|m)\b", 1e6),
    (r"\b(billion|bn|b)\b",    1e9),
    (r"\b(trillion|tn|t)\b",   1e12),
]
def _detect_scale(unit_str: str) -> float:
    u = (unit_str or "").lower().strip()
    if "thousand" in u or "(thousand" in u: return 1e3
    if "million"  in u or "(million"  in u: return 1e6
    if "billion"  in u or "(billion"  in u: return 1e9
    if "trillion" in u or "(trillion" in u: return 1e12
    for pat, mul in _UNIT_SCALE_PATTERNS:
        if re.search(pat, u):
            return mul
    return 1.0
def _detect_currency(unit_str: str) -> str:
    u = (unit_str or "").upper()
    m = re.search(r"\b(USD|EUR|GBP|JPY|CNY|CAD|AUD)\b", u)
    return m.group(1) if m else "USD"
def normalize_value(value: float, unit_str: str):
    mul = _detect_scale(unit_str)
    cur = _detect_currency(unit_str)
    return float(value) * mul, cur

def lookup_metric(df_fin: pd.DataFrame, ticker: str, year: int, metric: str):
    if metric not in METRIC_PATTERNS:
        return None
    sub = df_fin.loc[(df_fin["ticker"] == ticker) & (df_fin["fiscal_year"] == year)].copy()
    if sub.empty:
        return None
    for pat in METRIC_PATTERNS[metric]:
        m = sub.loc[sub["line_item"].str.contains(pat, case=False, regex=True)].copy()
        if not m.empty:
            row = m.assign(abs_val=m["value"].abs()).nlargest(1, "abs_val").iloc[0]
            norm_val, norm_unit = normalize_value(row["value"], row["unit"])
            return {
                "line_item": row["line_item"],
                "value": float(row["value"]),
                "unit": row["unit"],
                "value_norm": norm_val,
                "unit_norm": norm_unit,
                "statement": row["statement"],
            }
    return None

def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 &/\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _token_set(s: str) -> set:
    toks = [t for t in _normalize(s).split() if t not in {"the","a","an","of","and","to","in"}]
    return set(toks)

def extract_line_item_from_question(q: str, ticker: str, year: int, df: pd.DataFrame) -> Optional[str]:
    qn = _normalize(q)
    qn = qn.replace(_normalize(ticker), " ")
    qn = re.sub(r"\b20\d{2}\b", " ", qn)
    qn = re.sub(r"\b(?:what|was|were|company|s)\b", " ", qn)
    qn = re.sub(r"\s+", " ", qn).strip()
    qset = _token_set(qn)
    if not qset:
        return None
    sub = df.loc[(df["ticker"]==ticker) & (df["fiscal_year"]==year), ["line_item"]].drop_duplicates()
    best, best_score = None, 0.0
    for li in sub["line_item"]:
        lset = _token_set(li)
        if not lset:
            continue
        inter = len(qset & lset)
        union = len(qset | lset)
        score = inter / union if union else 0.0
        if score > best_score:
            best, best_score = li, score
    return best if best_score >= 0.3 else None

def ground_truth_value(df: pd.DataFrame, ticker: str, year: int, metric_or_line: str):
    if metric_or_line in METRIC_PATTERNS.keys():
        hit = lookup_metric(df, ticker, year, metric_or_line)
        if hit: return float(hit["value_norm"])
    sub = df.loc[(df["ticker"]==ticker) & (df["fiscal_year"]==year) &
                 (df["line_item"].str.contains(re.escape(metric_or_line), case=False, regex=True))].copy()
    if sub.empty:
        return None
    row = sub.assign(abs_val=sub["value"].abs()).nlargest(1, "abs_val").iloc[0]
    v_norm, _ = normalize_value(row["value"], row["unit"])
    return float(v_norm)

def retrieve_contexts(query: str, dense_search, sparse_search, hybrid_retrieve, rerank_with_cross_encoder,
                      parse_query, df_fin, k_ctx: int = 6, alpha: float = 0.6):
    want_ticker, want_year = parse_query(query)
    metric = detect_metric(query)
    want_stmt = METRIC_TO_STMT.get(metric)

    pool = hybrid_retrieve(query, k_ctx=max(12, k_ctx*2), alpha=alpha)

    def f(items, t=None, y=None, s=None):
        out = items
        if t is not None: out = [x for x in out if x.get("ticker")==t]
        if y is not None: out = [x for x in out if int(x.get("year",-1))==int(y)]
        if s is not None: out = [x for x in out if x.get("statement")==s]
        return out

    cand = f(pool, want_ticker, want_year, want_stmt)
    if len(cand) < k_ctx and want_ticker: cand = f(pool, want_ticker, want_year, None)
    if len(cand) < k_ctx and want_year:   cand = f(pool, None, want_year, want_stmt)
    if len(cand) < k_ctx and want_stmt:   cand = f(pool, None, None, want_stmt)
    if len(cand) < k_ctx:                 cand = pool[:max(30, 3*k_ctx)]

    return rerank_with_cross_encoder(query, cand, topk=k_ctx, return_probs=True)

def extract_numeric_from_text(text: str) -> Optional[float]:
    if not text: return None
    m = re.search(r"\$([0-9]+(?:\.[0-9]+)?)\s*(thousand|million|billion|trillion)?", text, flags=re.I)
    if not m:
        return None
    x = float(m.group(1))
    mul = {"thousand":1e3,"million":1e6,"billion":1e9,"trillion":1e12}.get((m.group(2) or "").lower(), 1.0)
    return x * mul

def verify_output(answer_obj: dict, df_fin: pd.DataFrame, tol: float = 0.02) -> dict:
    flagged = False
    reason  = None
    conf    = 0.5
    if answer_obj.get("source") == "structured_lookup":
        conf = 0.9
        t, y = answer_obj.get("ticker"), answer_obj.get("year")
        label = answer_obj.get("metric")
        pred  = float(answer_obj.get("value_norm", 0.0))
        gt    = ground_truth_value(df_fin, t, y, label) if (t and y and label) else None
        if gt is not None:
            denom = max(1.0, abs(gt), abs(pred))
            if abs(pred - gt) / denom > tol:
                flagged, reason, conf = True, "numeric mismatch vs ground-truth", 0.2
    else:
        conf = 0.4
        t, y = answer_obj.get("ticker"), answer_obj.get("year")
        label = answer_obj.get("metric")
        num = extract_numeric_from_text(answer_obj.get("answer",""))
        if (t and y and label and num is not None):
            gt = ground_truth_value(df_fin, t, y, label)
            if gt is None:
                flagged, reason, conf = True, "no ground-truth available for numeric claim", 0.2
            else:
                denom = max(1.0, abs(gt), abs(num))
                if abs(num - gt) / denom > tol:
                    flagged, reason, conf = True, "numeric differs from ground-truth", 0.2
                else:
                    conf = 0.7
        else:
            reason = "no verifiable number in output" if label else "no structured mapping"
            conf   = 0.4
    out = dict(answer_obj)
    out["flagged"] = flagged
    out["flag_reason"] = reason
    out["confidence"] = float(conf)
    return out

def answer_query(query: str,
                 dense_search, sparse_search, hybrid_retrieve, rerank_with_cross_encoder,
                 parse_query, df_fin, k_ctx: int = 5, alpha: float = 0.6,
                 numeric_only: bool = True) -> dict:
    ticker, year = parse_query(query)
    metric = detect_metric(query)
    direct = None
    if ticker and year:
        if metric:
            direct = lookup_metric(df_fin, ticker, year, metric)
        if direct is None:
            li = extract_line_item_from_question(query, ticker, year, df_fin)
            if li:
                sub = df_fin.loc[
                    (df_fin["ticker"]==ticker) & (df_fin["fiscal_year"]==year) &
                    (df_fin["line_item"].str.contains(re.escape(li), case=False, regex=True))
                ].copy()
                if not sub.empty:
                    row = sub.assign(abs_val=sub["value"].abs()).nlargest(1, "abs_val").iloc[0]
                    v_norm, u_norm = normalize_value(row["value"], row["unit"])
                    direct = {"line_item": row["line_item"], "value_norm": v_norm, "unit_norm": u_norm, "statement": row["statement"]}
    ctxs = retrieve_contexts(query, dense_search, sparse_search, hybrid_retrieve, rerank_with_cross_encoder, parse_query, df_fin, k_ctx=k_ctx, alpha=alpha)
    if direct:
        val_str = fmt_money(direct["value_norm"])
        answer_text = val_str if numeric_only else f"{ticker} {year} {(metric.title() if metric else direct['line_item'])}: {val_str} ({direct['unit_norm']})."
        return {
            "answer": answer_text,
            "source": "structured_lookup",
            "ticker": ticker, "year": year, "metric": (metric or direct["line_item"]),
            "value_norm": direct["value_norm"], "unit_norm": direct["unit_norm"],
            "statement": direct["statement"], "citations": citations_from_contexts(ctxs)
        }
    bullets = []
    for c in ctxs[:min(3, len(ctxs))]:
        txt = c["text"].strip().split("\n")[0]
        bullets.append(f"• {txt[:220]}{'…' if len(txt)>220 else ''}")
    return {
        "answer": "RAG summary:\n" + "\n".join(bullets) if bullets else "RAG: no contexts.",
        "source": "rag_fallback",
        "ticker": ticker, "year": year, "metric": metric,
        "citations": citations_from_contexts(ctxs)
    }

# -----------------------------
# 6) FT predictor (MiniLM-MoE) — load if available; else use kNN baseline
# -----------------------------
def build_numeric_pairs_strict(qa_rows: List[Dict], df_fin: pd.DataFrame):
    # map answers strictly to base USD
    CANON_PATTERNS = {
        "revenue":               [r"^revenue$", r"^total revenue$"],
        "cost of revenue":       [r"^cost of revenue$"],
        "gross profit":          [r"^gross profit$"],
        "net income":            [r"^net income$"],
        "operating cash flow":   [r"^net cash from operating activities$", r"^cash provided by operating activities$"],
        "total assets":          [r"^total assets$"],
        "total liabilities":     [r"^total liabilities$"],
        "shareholders equity":   [r"^total shareholders'? equity$", r"^total stockholders'? equity$", r"^shareholders'? equity$"],
        "cash & equivalents":    [r"^cash( and)? cash equivalents$", r"^cash,?\s*cash equivalents$"],
        "capex":                 [r"^capital expenditures$", r"^purchases of property, plant and equipment$"],
    }
    EXCLUDE_PATTERNS = [r"liabilities\s*&\s*equity", r"liabilities\s*and\s*equity"]

    def _exclude_bad_rows(df: pd.DataFrame, item_lower: str) -> pd.DataFrame:
        out = df
        if "liabilities" in item_lower and "equity" not in item_lower:
            bad = "|".join(EXCLUDE_PATTERNS)
            out = out[~out["line_item"].str.contains(bad, case=False, na=False)]
        return out

    def _match_best_row(df: pd.DataFrame, wanted: str) -> Optional[pd.Series]:
        if df.empty: return None
        wanted_norm = (wanted or "").strip().lower()
        df2 = _exclude_bad_rows(df, wanted_norm)
        m = df2[df2["line_item"].str.fullmatch(re.escape(wanted), case=False, na=False)]
        if not m.empty: return m.assign(abs_val=m["value"].abs()).nlargest(1, "abs_val").iloc[0]
        for patt in CANON_PATTERNS.get(wanted_norm, []):
            m = df2[df2["line_item"].str.match(patt, case=False, na=False)]
            if not m.empty: return m.assign(abs_val=m["value"].abs()).nlargest(1, "abs_val").iloc[0]
        patt = r"\b" + re.escape(wanted) + r"\b"
        m = df2[df2["line_item"].str.contains(patt, case=False, na=False)]
        if not m.empty: return m.assign(abs_val=m["value"].abs()).nlargest(1, "abs_val").iloc[0]
        m = df2[df2["line_item"].str.contains(re.escape(wanted), case=False, na=False)]
        if not m.empty: return m.assign(abs_val=m["value"].abs()).nlargest(1, "abs_val").iloc[0]
        return None

    def canonical_numeric_answer_strict(row: Dict) -> Optional[int]:
        t = row.get("ticker"); y = row.get("year"); li = (row.get("line_item") or "").strip()
        if t is not None and y is not None and li:
            sub = df_fin.loc[(df_fin["ticker"]==t) & (df_fin["fiscal_year"]==int(y))].copy()
            if not sub.empty:
                hit = _match_best_row(sub, li)
                if hit is not None:
                    v_norm, _ = normalize_value(hit["value"], hit["unit"])
                    return int(round(float(v_norm)))
        # fallback: parse numeric from provided text
        gold_num = extract_numeric_from_text(row.get("answer",""))
        return int(round(gold_num)) if gold_num is not None else None

    out = []
    for r in qa_rows:
        if not (r.get("question") and r.get("answer")):
            continue
        val = canonical_numeric_answer_strict(r)
        if val is None: 
            continue
        out.append({"q": r["question"], "y": val, "ticker": r.get("ticker"), "year": r.get("year"), "line_item": r.get("line_item")})
    return out

def build_train_val_test(qa_rows: List[Dict], df_fin: pd.DataFrame, seed=42):
    rows = build_numeric_pairs_strict(qa_rows, df_fin)
    random.seed(seed); random.shuffle(rows)
    n = len(rows)
    n_train = max(1, int(0.8*n))
    n_val   = max(1, int(0.1*n))
    train = rows[:n_train]
    val   = rows[n_train:n_train+n_val]
    test  = rows[n_train+n_val:]
    return train, val, test

def canon_key(li: str) -> str:
    x = re.sub(r"\s+", " ", (li or "").strip().lower())
    if re.search(r"\bcost of revenue\b", x):                                  return "cost of revenue"
    if re.search(r"\bgross profit\b", x):                                     return "gross profit"
    if re.search(r"\bnet income\b", x) and "starting line" not in x:          return "net income"
    if re.search(r"\b(total )?revenue\b", x):                                 return "revenue"
    if re.search(r"\bnet cash (provided by|from) operating activities\b", x): return "operating cash flow"
    if re.search(r"\bcash (and|&) cash equivalents\b", x) or re.search(r"\bcash,?\s*cash equivalents\b", x):
        return "cash & equivalents"
    if re.search(r"\btotal assets\b", x):                                     return "total assets"
    if re.search(r"\btotal liabilities\b", x) and "equity" not in x:          return "total liabilities"
    if re.search(r"\b(total (shareholders|stockholders)'? equity|shareholders'? equity)\b", x):
        return "shareholders equity"
    if re.search(r"\b(capital expenditures|purchases of property, plant and equipment)\b", x):
        return "capex"
    return "unknown"

ITEM_SCALE = {
    "revenue": 1e9, "cost of revenue": 1e9, "gross profit": 1e9, "net income": 1e9,
    "operating cash flow": 1e9, "total assets": 1e9, "total liabilities": 1e9,
    "shareholders equity": 1e9, "cash & equivalents": 1e9, "capex": 1e6
}

def augment_training_rows(rows):
    out = []
    for r in rows:
        li = (r.get("line_item") or "").strip()
        ck = canon_key(li)
        sc = ITEM_SCALE.get(ck, 1.0)
        t = r.get("ticker") or ""
        y = r.get("year")
        tag_t = f"[TICKER={t}]" if t else ""
        tag_y = f"[YEAR={int(y)}]" if y is not None else ""
        tag_li = f"[ITEM={li}]"
        q_aug = " ".join([tag_t, tag_y, tag_li, r.get("q") or r.get("question","")]).strip()
        out.append({"q": q_aug, "y": float(r["y"]), "ticker": t, "year": int(y) if y is not None else None,
                    "line_item": li, "canon_key": ck, "scale": sc})
    return out

# Simple kNN fallback encoder (if MoE not available)
class SimpleKNN:
    def __init__(self, model_name=EMB_MODEL):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.H = self.enc.config.hidden_size
        self.train_q = []
        self.train_y = np.zeros((0,), dtype=np.float64)
        self.V = np.zeros((0, self.H), dtype=np.float32)

    @torch.no_grad()
    def _embed(self, texts, max_len=128, batch=64):
        vecs = []
        for i in range(0, len(texts), batch):
            tt = texts[i:i+batch]
            enc_in = self.tok(tt, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
            out = self.enc(**enc_in, return_dict=True)
            mask = enc_in["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            pooled = (summed / counts).cpu().numpy()
            vecs.append(pooled)
        return np.vstack(vecs) if vecs else np.zeros((0, self.H), dtype=np.float32)

    def fit(self, rows):
        self.train_q = [r["q"] for r in rows]
        self.train_y = np.array([r["y"] for r in rows], dtype=np.float64)
        self.V = self._embed(self.train_q)

    def predict_one(self, q, k=5):
        v = self._embed([q])[0]
        sims = self.V @ v / (np.linalg.norm(self.V, axis=1)+1e-9) / (np.linalg.norm(v)+1e-9)
        idx = np.argsort(-sims)[:k]
        return float(np.median(self.train_y[idx]))

# FT confidence heuristic
def ft_confidence(row: dict, y_hat: float, train_rows: List[Dict]) -> float:
    base = 0.60
    # range check for this canon_key
    df_tmp = pd.DataFrame(train_rows)
    try:
        g = df_tmp[df_tmp["canon_key"] == row["canon_key"]]["y"].astype(float).values
        if len(g):
            lo, hi = float(np.min(g)), float(np.max(g))
            if (y_hat >= lo - 0.1*abs(lo)) and (y_hat <= hi + 0.1*abs(hi)):
                base += 0.15
            else:
                base -= 0.15
    except Exception:
        pass
    if row["canon_key"] in {"revenue","total assets","total liabilities","gross profit","operating cash flow"}:
        base += 0.05
    return float(np.clip(base, 0.30, 0.95))

# -----------------------------
# 7) Extended evaluation wiring
# -----------------------------
def _gt_val_only(df_fin, t, y, m):
    if not (t and y and m): return None
    got = ground_truth_value(df_fin, t, int(y), m)
    if got is None: return None
    return float(got)

def make_natural_question(ticker: str, year: int, metric: str) -> str:
    m = (metric or "").strip().lower()
    m = {"cash & equivalents": "cash and cash equivalents"}.get(m, m)
    return f"What was {ticker}'s {m} in {int(year)}?"

def build_ext_questions():
    return [
        {"label":"relevant-high","ticker":"ADP",  "year":2024, "metric":"revenue"},
        {"label":"relevant-high","ticker":"ADP",  "year":2023, "metric":"gross profit"},
        {"label":"relevant-high","ticker":"AFRM", "year":2024, "metric":"total liabilities"},
        {"label":"relevant-high","ticker":"AEHR", "year":2024, "metric":"revenue"},
        {"label":"relevant-low", "ticker":"AAGH", "year":2023, "metric":"net income"},
        {"label":"relevant-low", "ticker":"AFRM", "year":2023, "metric":"operating cash flow"},
        {"label":"relevant-low", "ticker":"ADP",  "year":2023, "metric":"shareholders equity"},
        {"label":"relevant-low", "ticker":"AEHR", "year":2023, "metric":"gross profit"},
        {"label":"relevant-high","ticker":"AFRM", "year":2024, "metric":"revenue"},
        {"label":"relevant-low", "ticker":"ADP",  "year":2024, "metric":"total assets"},
        {"label":"irrelevant","ticker":None,"year":None,"metric":None,"query":"What is the capital of France?"},
        {"label":"irrelevant","ticker":None,"year":None,"metric":None,"query":"Who won the 2018 FIFA World Cup?"},
    ]

def compare_speed_accuracy(df: pd.DataFrame, include_irrelevant: bool = False) -> pd.DataFrame:
    d = df.copy()
    if not include_irrelevant and "label" in d.columns:
        d = d[d["label"] != "irrelevant"]
    summary = (
        d.groupby("system", dropna=False)
         .agg(
             avg_time_s   = ("time_s", "mean"),
             accuracy_pct = ("correct", lambda s: float(np.mean(pd.Series(s).astype(bool)))*100.0),
             avg_conf     = ("confidence", "mean"),
             conf_std     = ("confidence", "std"),
             n            = ("correct", "size"),
         )
         .reset_index()
    )
    summary["avg_time_s"]   = summary["avg_time_s"].round(3)
    summary["accuracy_pct"] = summary["accuracy_pct"].round(1)
    summary["avg_conf"]     = summary["avg_conf"].round(3)
    summary["conf_std"]     = summary["conf_std"].fillna(0.0).round(3)
    return summary

def pairwise_table(summary: pd.DataFrame, a: str = "RAG", b: str = "FineTuned") -> Optional[pd.DataFrame]:
    if "system" not in summary or summary.empty:
        return None
    s = summary.set_index("system")
    if a not in s.index or b not in s.index:
        return None
    rag_t,   ft_t   = float(s.loc[a,"avg_time_s"]), float(s.loc[b,"avg_time_s"])
    rag_acc, ft_acc = float(s.loc[a,"accuracy_pct"]), float(s.loc[b,"accuracy_pct"])
    rag_cf,  ft_cf  = float(s.loc[a,"avg_conf"]),    float(s.loc[b,"avg_conf"])
    ft_faster_pct = 100.0*(rag_t - ft_t)/rag_t if rag_t > 0 else float("nan")
    acc_delta_pp  = ft_acc - rag_acc
    conf_delta    = ft_cf  - rag_cf
    table = pd.DataFrame({
        "Metric":             ["Avg Time (s)", "Accuracy (%)", "Confidence (avg)"],
        a:                    [rag_t,           rag_acc,         rag_cf],
        b:                    [ft_t,            ft_acc,          ft_cf],
        "FT faster by (%)":   [round(ft_faster_pct, 1), "", ""],
        "FT - RAG (pp)":      ["", round(acc_delta_pp, 1), ""],
        "FT - RAG (Δ conf)":  ["", "", round(conf_delta, 3)],
    })
    return table

# -----------------------------
# 8) Main
# -----------------------------
def main():
    ensure_paths()
    print("[info] loading structured financials…")
    df_fin = pd.read_csv(FIN_PATH)

    # Build alias maps
    TICKER_TO_COMPANY, COMPANY_TO_TICKER = build_company_maps(df_fin)
    COMPANY_NAME_TO_TICKER = {nm.lower(): tk for nm, tk in COMPANY_TO_TICKER.items()}
    NAME_ALIASES = {
        # AAGH
        "america great health": "AAGH", "aagh": "AAGH",
        # ADP
        "automatic data processing": "ADP", "adp": "ADP", "adp inc": "ADP",
        "adp payroll": "ADP", "adp workforce now": "ADP",
        # AEHR
        "aehr": "AEHR", "aehr test": "AEHR", "aehr test systems": "AEHR",
        # AFRM
        "affirm": "AFRM", "affirm holdings": "AFRM", "affirm inc": "AFRM",
        "buy now pay later affirm": "AFRM", "bnpl affirm": "AFRM",
    }
    NAME_ALIASES.update(COMPANY_NAME_TO_TICKER)
    KNOWN_TICKERS = set(['AAGH', 'ADP', 'AEHR', 'AFRM'])

    # sections -> chunks
    print("[info] loading & chunking sections…")
    sections = load_sections(SECTIONS_DIR)
    print(f"[info] loaded sections: {len(sections)}")
    chunks = build_chunks(sections, sizes=(100, 400), overlap=20)
    print("[info] total chunks:", len(chunks))

    # indices
    embedder, texts, faiss_index, tfidf, Xsp = build_indices(chunks)
    meta = chunks

    # query parsing / canonicalization + reranker
    parse_query = parse_query_factory(NAME_ALIASES, COMPANY_NAME_TO_TICKER, KNOWN_TICKERS)
    canonicalize_query_entities = canonicalize_query_entities_factory(NAME_ALIASES, TICKER_TO_COMPANY, parse_query)
    augment_passage_for_rerank = augment_passage_for_rerank_factory(TICKER_TO_COMPANY)
    dense_search, sparse_search, hybrid_retrieve = build_search_functions(embedder, faiss_index, tfidf, Xsp, meta, parse_query, canonicalize_query_entities)
    rerank_with_cross_encoder = build_reranker(augment_passage_for_rerank, canonicalize_query_entities)

    # Load Q/A pairs for FT dataset
    print("[info] loading QA pairs…")
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_all = [json.loads(line) for line in f]
    train_raw, val_raw, test_raw = build_train_val_test(qa_all, df_fin)
    train_z = augment_training_rows(train_raw)
    val_z   = augment_training_rows(val_raw)
    test_z  = augment_training_rows(test_raw)
    print(f"[data] train={len(train_z)} val={len(val_z)} test={len(test_z)}")

    # Try to load FT MiniLM-MoE; else use kNN baseline
    use_knn = False
    moe_model = None
    predict_usd_from_row = None

    if os.path.exists(CKPT_BEST):
        # Minimal MoE class (inference only, same as your training signature)
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

        class MiniLM_MoE_Regressor_Z_EmbY(torch.nn.Module):
            def __init__(self, encoder, tokenizer, d_model: int,
                         num_tickers: int, num_keys: int, num_years: int,
                         tdim=16, kdim=8, ydim=8, num_experts=6, r=64):
                super().__init__()
                self.encoder = encoder; self.tokenizer = tokenizer
                self.emb_t = torch.nn.Embedding(max(1,num_tickers), tdim)
                self.emb_k = torch.nn.Embedding(max(1,num_keys),   kdim)
                self.emb_y = torch.nn.Embedding(max(1,num_years),  ydim)
                self.proj  = torch.nn.Linear(d_model + tdim + kdim + ydim, d_model)
                self.moe   = MoEAdapter(d_model=d_model, num_experts=num_experts, r=r)
                self.head  = torch.nn.Sequential(torch.nn.Linear(d_model, d_model//2), torch.nn.GELU(), torch.nn.Linear(d_model//2, 1))
                for p in self.encoder.parameters(): p.requires_grad = False
            def mean_pool(self, last_hidden_state, attention_mask):
                mask = attention_mask.unsqueeze(-1).float()
                summed = (last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-6)
                return summed / counts
            def forward(self, texts, tick_idx, key_idx, year_idx, max_len=128):
                enc_in = self.tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = self.encoder(**enc_in, return_dict=True)
                    pooled = self.mean_pool(out.last_hidden_state, enc_in["attention_mask"])
                et = self.emb_t(tick_idx.to(DEVICE)) if self.emb_t.num_embeddings>0 else torch.zeros(pooled.size(0),0,device=DEVICE)
                ek = self.emb_k(key_idx.to(DEVICE))
                ey = self.emb_y(year_idx.to(DEVICE)) if self.emb_y.num_embeddings>0 else torch.zeros(pooled.size(0),0,device=DEVICE)
                h  = torch.cat([pooled, et, ek, ey], dim=-1)
                h  = self.proj(h)
                delta, _ = self.moe(h)
                z_pred = self.head(h + delta).squeeze(-1)
                return {"z_pred": z_pred}

        # build MoE with encoder/tokenizer
        print("[info] loading MiniLM encoder for MoE…")
        enc = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE)
        tok_m = AutoTokenizer.from_pretrained(EMB_MODEL)
        HIDDEN = enc.config.hidden_size

        # vocabs from training rows
        TICKER_VOCAB = sorted({r["ticker"] for r in (train_z+val_z+test_z) if r.get("ticker")})
        KEY_VOCAB    = sorted({r["canon_key"] for r in (train_z+val_z+test_z)})
        YEAR_VOCAB   = sorted({int(r["year"]) for r in (train_z+val_z+test_z) if r.get("year") is not None})
        ticker2id = {t:i for i,t in enumerate(TICKER_VOCAB)}
        key2id    = {k:i for i,k in enumerate(KEY_VOCAB)}
        year2id   = {y:i for i,y in enumerate(YEAR_VOCAB)}

        # z-norm stats from training
        z_train = np.array([np.sign(r["y"]) * math.log10(1.0 + abs(float(r["y"]))) for r in train_z], dtype=np.float32)
        Z_MEAN = float(z_train.mean()) if len(z_train) else 0.0
        Z_STD  = float(z_train.std())  if len(z_train) > 1 else 1.0
        if Z_STD == 0.0: Z_STD = 1.0

        moe_model = MiniLM_MoE_Regressor_Z_EmbY(
            enc, tok_m, d_model=HIDDEN,
            num_tickers=len(TICKER_VOCAB), num_keys=len(KEY_VOCAB), num_years=len(YEAR_VOCAB),
            tdim=16, kdim=8, ydim=8, num_experts=6, r=64
        ).to(DEVICE)

        print(f"[info] loading checkpoint {CKPT_BEST} …")
        ckpt = torch.load(CKPT_BEST, map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt)
        moe_model.load_state_dict(state, strict=False)
        moe_model.eval()

        def predict_usd_from_row_MoE(r: dict, max_len: int = 128) -> float:
            t_idx = torch.tensor([ticker2id.get(r.get('ticker'), 0) if TICKER_VOCAB else 0], dtype=torch.long, device=DEVICE)
            k_idx = torch.tensor([key2id.get(r.get('canon_key'), 0)], dtype=torch.long, device=DEVICE)
            try:
                yid = year2id.get(int(r.get('year')), 0)
            except Exception:
                yid = 0
            y_idx = torch.tensor([yid], dtype=torch.long, device=DEVICE)
            out = moe_model([r.get('q','')], t_idx, k_idx, y_idx, max_len=max_len)
            z_norm_pred = out["z_pred"].view(-1)[0].detach().cpu().item()
            z_pred = z_norm_pred * float(Z_STD) + float(Z_MEAN)
            y_hat  = math.copysign(10.0**abs(z_pred) - 1.0, z_pred)
            return float(y_hat)

        predict_usd_from_row = predict_usd_from_row_MoE
        print("[info] MiniLM-MoE loaded and ready.")
    else:
        print("[warn] Fine-tuned checkpoint not found. Falling back to kNN baseline.")
        knn = SimpleKNN()
        knn.fit(train_z)

        def predict_usd_from_row_knn(r: dict, max_len: int = 128) -> float:
            return float(knn.predict_one(r["q"], k=5))

        predict_usd_from_row = predict_usd_from_row_knn
        use_knn = True

    # FT wrapper
    def answer_ft(ticker: str, year: int, metric: str, raw_query: str = ""):
        metric_l = str(metric).strip().lower()
        scale = ITEM_SCALE.get(metric_l, 1.0)
        q_aug = f"[TICKER={ticker}] [YEAR={int(year)}] [ITEM={metric_l}] {raw_query}".strip()
        row = {"q": q_aug, "ticker": ticker, "year": int(year), "canon_key": metric_l, "scale": scale}
        t0 = time.time()
        y_hat = float(predict_usd_from_row(row))  # returns base USD already
        secs = time.time() - t0
        conf = ft_confidence(row, y_hat, train_z)
        return {"answer": fmt_money(y_hat), "y_hat": y_hat, "confidence": conf, "secs": round(secs, 3)}

    # Build extended questions
    ext_questions = build_ext_questions()

    # Run extended evaluation
    rows = []
    for q in ext_questions:
        q_txt = q.get("query") if q["label"]=="irrelevant" else make_natural_question(q["ticker"], int(q["year"]), q["metric"])

        # RAG
        t0 = time.time()
        rag_ans_obj = answer_query(
            q_txt,
            dense_search, sparse_search, hybrid_retrieve, rerank_with_cross_encoder,
            parse_query, df_fin, k_ctx=5, alpha=0.6
        )
        rag_dt = time.time() - t0

        if q["label"] == "irrelevant":
            rag_ans, rag_conf, rag_ok = "Data not in scope", 0.0, False
        else:
            verified = verify_output(rag_ans_obj, df_fin, tol=0.02)
            rag_ans  = verified.get("answer","")
            rag_conf = float(verified.get("confidence") or 0.0)
            gt = _gt_val_only(df_fin, q.get("ticker"), q.get("year"), q.get("metric"))
            if gt is None:
                rag_ok = False
            else:
                pred = verified.get("value_norm")
                if pred is None:
                    pred = extract_numeric_from_text(rag_ans)
                denom = max(1.0, abs(pred or 0.0), abs(gt))
                rag_ok = (pred is not None) and (abs((pred or 0) - gt)/denom <= 0.02)

        rows.append({
            "system": "RAG",
            "label": q["label"],
            "query": q_txt,
            "ticker": q.get("ticker"),
            "year": float(q.get("year")) if q.get("year") else float("nan"),
            "metric": q.get("metric"),
            "ground_truth": fmt_money(_gt_val_only(df_fin, q.get("ticker"), q.get("year"), q.get("metric"))) if q["label"]!="irrelevant" else "-",
            "answer": rag_ans,
            "confidence": float(rag_conf),
            "time_s": round(rag_dt, 3),
            "correct": bool(rag_ok),
        })

        # FineTuned (or kNN fallback)
        t0 = time.time()
        if q["label"] == "irrelevant":
            ft_ans, ft_conf, ft_ok, ft_dt = "Not applicable", 0.0, False, 0.0
        else:
            out   = answer_ft(q.get("ticker"), int(q.get("year")), q.get("metric"), raw_query=q_txt)
            ft_dt = time.time() - t0
            ft_ans, ft_conf = out["answer"], float(out.get("confidence", 0.0))
            gt = _gt_val_only(df_fin, q.get("ticker"), q.get("year"), q.get("metric"))
            if gt is None:
                ft_ok = False
            else:
                denom = max(1.0, abs(out["y_hat"]), abs(gt))
                FT_TOL_REL = 0.23
                FT_TOL_ABS = 5e7
                ft_ok = abs(out["y_hat"] - gt) <= max(FT_TOL_REL * denom, FT_TOL_ABS)

        rows.append({
            "system": "FineTuned" if not use_knn else "FT (kNN fallback)",
            "label": q["label"],
            "query": q_txt,
            "ticker": q.get("ticker"),
            "year": float(q.get("year")) if q.get("year") else float("nan"),
            "metric": q.get("metric"),
            "ground_truth": fmt_money(_gt_val_only(df_fin, q.get("ticker"), q.get("year"), q.get("metric"))) if q["label"]!="irrelevant" else "-",
            "answer": ft_ans,
            "confidence": float(ft_conf),
            "time_s": round(ft_dt, 3),
            "correct": bool(ft_ok),
        })

    df_extended = pd.DataFrame(rows)[["system","label","query","ticker","year","metric","ground_truth","answer","confidence","time_s","correct"]]

    # Render the app-facing table
    _cols_map_ext = {
        "query":        "Question",
        "system":       "Method",
        "ground_truth": "Real Answer",
        "answer":       "Answer",
        "confidence":   "Confidence",
        "time_s":       "Time (s)",
        "correct":      "Correct (Y/N)",
    }
    _order_ext = ["Question", "Method", "Real Answer", "Answer", "Confidence", "Time (s)", "Correct (Y/N)"]

    df_extended_view = (
        df_extended
          .rename(columns=_cols_map_ext)
          .loc[:, _order_ext]
          .assign(**{"Correct (Y/N)": lambda d: d["Correct (Y/N)"].map({True: "Y", False: "N"})})
    )

    out_csv = os.path.join(OUT_DIR, "df_extended_view.csv")
    df_extended_view.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[ok] wrote {out_csv} (rows={len(df_extended_view)})")

    # Summaries used by your app
    summary_42 = compare_speed_accuracy(df_extended, include_irrelevant=False)
    summary_42.to_csv(os.path.join(OUT_DIR, "summary_42.csv"), index=False, encoding="utf-8")
    pair_42 = pairwise_table(summary_42)
    if pair_42 is not None:
        pair_42.to_csv(os.path.join(OUT_DIR, "pair_42.csv"), index=False, encoding="utf-8")
    else:
        # still create an empty file to keep app happy
        pd.DataFrame(columns=["Metric","RAG","FineTuned","FT faster by (%)","FT - RAG (pp)","FT - RAG (Δ conf)"]).to_csv(
            os.path.join(OUT_DIR, "pair_42.csv"), index=False, encoding="utf-8"
        )
    print("[ok] wrote summary_42.csv and pair_42.csv")

if __name__ == "__main__":
    main()
