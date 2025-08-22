#!/usr/bin/env python
# coding: utf-8



import os
import re
import json
import math
import pandas as pd
from typing import List, Dict
import simfin as sf

# Optional (only used if you parse PDFs/HTML): 
try:
    import pdfplumber
    from bs4 import BeautifulSoup
except Exception:
    pdfplumber = None



# If these are already installed, you can skip the %pip lines.
# %pip install -q sentence-transformers faiss-cpu scikit-learn transformers

import os, re, glob, json, uuid, math
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# Dense embeddings & cross-encoder
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import ipywidgets

# Sparse retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os, warnings

os.environ["TQDM_NOTEBOOK"] = "0"   # tells tqdm to use console mode
warnings.filterwarnings("ignore")

# Dense index
import faiss


# 2.0.1) Device Selection & Config

# In[12]:


# Pick the best available device once
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("[info] CUDA:", torch.cuda.get_device_name(0))
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[info] Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("[info] CPU")


import os, io, json, base64, requests
import pandas as pd

def _get(key, default=None):
    # read from env first (since this script is a subprocess), fall back to Streamlit secrets if available
    v = os.getenv(key)
    if v not in (None, ""):
        return v
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return default

GH_OWNER     = _get("GH_OWNER")
GH_REPO      = _get("GH_REPO")
GH_BRANCH    = _get("GH_BRANCH", "main")
GH_TOKEN     = _get("GH_TOKEN") or _get("GITHUB_TOKEN")  # support either name
GH_BASE_PATH = _get("GH_BASE_PATH", "out")

def _gh_headers():
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "streamlit-app",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GH_TOKEN:
        h["Authorization"] = f"Bearer {GH_TOKEN}"
    return h

def _gh_contents_url(relpath: str) -> str:
    relpath = relpath.lstrip("/")
    return f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{relpath}"

def _gh_get_sha(relpath: str):
    url = _gh_contents_url(relpath) + f"?ref={GH_BRANCH}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code == 200:
        return r.json().get("sha")
    if r.status_code == 404:
        return None
    r.raise_for_status()

def _gh_put_file(relpath: str, content_bytes: bytes, message: str):
    b64 = base64.b64encode(content_bytes).decode("ascii")
    payload = {"message": message, "content": b64, "branch": GH_BRANCH}
    sha = _gh_get_sha(relpath)
    if sha:
        payload["sha"] = sha
    name  = _get("GH_COMMITTER_NAME")
    email = _get("GH_COMMITTER_EMAIL")
    if name and email:
        payload["committer"] = {"name": name, "email": email}
    url = _gh_contents_url(relpath)
    r = requests.put(url, headers=_gh_headers(), data=json.dumps(payload), timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub write failed ({r.status_code}): {r.text[:300]}")
    return r.json()

def gh_write_csv(df: pd.DataFrame, relpath: str, message: str = None):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    rel = relpath if GH_BASE_PATH in ("", None) else f"{GH_BASE_PATH.rstrip('/')}/{relpath.lstrip('/')}"
    return _gh_put_file(rel, buf.getvalue().encode("utf-8"), message or f"Update {rel}")

def _gh_preflight():
    # Safe diagnostics printed to your "save_table.py logs" expander
    print("=== GitHub preflight ===")
    print(f"GH_OWNER={GH_OWNER!r} GH_REPO={GH_REPO!r} GH_BRANCH={GH_BRANCH!r} GH_BASE_PATH={GH_BASE_PATH!r}")
    print(f"GH_TOKEN_present={bool(GH_TOKEN)}")  # don't print the token itself

    # 1) Is the token valid?
    r_user = requests.get("https://api.github.com/user", headers=_gh_headers(), timeout=30)
    print("whoami_status", r_user.status_code)

    # 2) Can the token see the repo?
    r_repo = requests.get(f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}", headers=_gh_headers(), timeout=30)
    print("repo_status", r_repo.status_code)

    # 3) Does the branch exist?
    r_branch = requests.get(
        f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/branches/{GH_BRANCH}",
        headers=_gh_headers(), timeout=30
    )
    print("branch_status", r_branch.status_code)

    # Clear, actionable errors
    if r_repo.status_code == 404:
        raise RuntimeError(
            "Your token can’t see this repository. For a fine-grained PAT: "
            "Resource owner = GH_OWNER; Repository access = Only select repositories -> GH_REPO; "
            "Permissions -> Repository contents = Read & write, Metadata = Read-only. "
            "If the repo is in an organization, an org admin must approve the token."
        )
    if r_branch.status_code == 404:
        raise RuntimeError(
            f"Branch {GH_BRANCH!r} does not exist. Set GH_BRANCH to the repo’s actual default branch "
            "(often 'main' or 'master') in Streamlit Secrets."
        )
    if r_user.status_code != 200:
        raise RuntimeError(
            "PAT looks invalid/expired. Regenerate and set GH_TOKEN in Streamlit Secrets."
        )


# --- Repo-aware base paths (PATHS ONLY; no logic changes) ---
# Set REPO_ROOT to your local GitHub repo clone (absolute or relative).
# If you run the notebook from inside the repo root, this default is fine.
REPO_ROOT = os.environ.get("REPO_ROOT", os.getcwd())
OUT_DIR   = os.path.join(REPO_ROOT, "out")
os.makedirs(OUT_DIR, exist_ok=True)


# Paths & models
# OLD
# SECTIONS_DIR = "out/sections_text_yearly"
# NEW
SECTIONS_DIR = os.path.join(OUT_DIR, "sections_text_yearly")
EMB_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"      # or: "intfloat/e5-small-v2"
RERANKER     = "cross-encoder/ms-marco-MiniLM-L-6-v2"        # fast, ~80MB


# 2.1) Load Section Texts & Chunk at 100 / 400 Tokens

# In[13]:


# Run this once near the top of your notebook (BEFORE chunking/embedding cells)

import warnings

# Suppress only THIS specific FutureWarning message from transformers
warnings.filterwarnings(
    "ignore",
    message=r".*`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning,
    module=r"transformers\.tokenization_utils_base"
)

# (Optional) also quiet all transformers advisory logs
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()


# After you create tokenizers, set their default behavior explicitly.

# embedding tokenizer (tok)
try:
    tok.clean_up_tokenization_spaces = False
except NameError:
    pass

# cross-encoder tokenizer (ce_tok), if you use one
try:
    ce_tok.clean_up_tokenization_spaces = False
except NameError:
    pass



# In[14]:


# Patch: redefine chunk_by_tokens to use a stable decode that avoids the FutureWarning
from transformers import AutoTokenizer
import re

# --- Load all per-year section text files ---
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

sections = load_sections(SECTIONS_DIR)
assert sections, f"No section files found in {SECTIONS_DIR}. Run Step-1 first."
print(f"[info] loaded sections: {len(sections)}")

# --- Tokenizer for token-aware chunking ---
tok = AutoTokenizer.from_pretrained(EMB_MODEL)


def _decode(ids):
    # Stable decoding: don't auto "clean up" spaces; skip specials
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

def num_tokens(text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))

def chunk_by_tokens(text: str, max_tokens: int, overlap_tokens: int = 20):
    """
    Token-aware chunking with light sentence alignment and overlap.
    This version suppresses the transformers FutureWarning by controlling decode().
    """
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text.strip()] if text.strip() else []

    # light sentence split
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z(])', text.strip()) or [text.strip()]

    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        s_ids = tok.encode(s, add_special_tokens=False)
        s_len = len(s_ids)

        # hard-wrap very long sentences
        if s_len > max_tokens:
            start = 0
            while start < s_len:
                end = min(s_len, start + max_tokens)
                piece = _decode(s_ids[start:end]).strip()
                if piece:
                    chunks.append(piece)
                start = max(end - overlap_tokens, end)
            continue

        # pack into current window
        if cur_len + s_len <= max_tokens:
            cur.append(s); cur_len += s_len
        else:
            if cur:
                chunks.append(" ".join(cur).strip())
            # add overlap tail from previous chunk
            if overlap_tokens > 0 and chunks:
                prev_ids = tok.encode(chunks[-1], add_special_tokens=False)
                tail     = _decode(prev_ids[-overlap_tokens:])
                cur, cur_len = [tail, s], len(prev_ids[-overlap_tokens:]) + s_len
            else:
                cur, cur_len = [s], s_len

    if cur:
        chunks.append(" ".join(cur).strip())

    # enforce limit with small tolerance
    return [c for c in chunks if c and num_tokens(c) <= max_tokens + 5]

from hashlib import blake2b
def stable_id(*parts) -> str:
    h = blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore")); h.update(b"|")
    return h.hexdigest()

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

# (Optional) Rebuild chunks if you want to re-run 2.1 with this patched function:
chunks = build_chunks(sections, sizes=(100, 400), overlap=20)
print("[info] total chunks:", len(chunks))


# 2.2) Embedding & Indexing (Dense FAISS + Sparse TF-IDF)

# In[15]:


# --- Sentence embeddings ---
embedder = SentenceTransformer(EMB_MODEL)
print("[info] embedding model:", EMB_MODEL)

texts = [c["text"] for c in chunks]
X = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
X = np.asarray(X, dtype="float32")
print("[info] embeddings:", X.shape)

# --- Dense FAISS (Inner Product; embeddings are normalized so IP≈cosine) ---
faiss_index = faiss.IndexFlatIP(X.shape[1])
faiss_index.add(X)
print("[info] FAISS ntotal:", faiss_index.ntotal)

# --- Sparse TF-IDF ---
tfidf = TfidfVectorizer(strip_accents="unicode", lowercase=True, max_df=0.95, min_df=1,sublinear_tf=True, ngram_range=(1,2))
Xsp = tfidf.fit_transform(texts)
print("[info] TF-IDF matrix:", Xsp.shape)

# Keep aligned metadata list (same order as 'texts')
meta = chunks


# 2.3) Hybrid Retrieval Pipeline (dense + sparse + weighted fusion)

# In[16]:


# Preprocess, dense & sparse search, score fusion
import re
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- knobs (put near your configs) ---
FUSE_RETURN = 36          # how many fused candidates you keep before CE (try 36 or 48)
FUSE_FETCH_MULT = 2       # how much to fetch from each retriever relative to FUSE_RETURN


# Light stopwords (kept tiny on purpose)
STOPWORDS = set("""
a an and are as at be but by for if in into is it its of on or such that the their then there these this to with
""".split())

def preprocess_query(q: str) -> str:
    q = (q or "").lower().strip()
    q = re.sub(r"[^a-z0-9 %$.,/-]+", " ", q)
    q = " ".join(w for w in q.split() if w not in STOPWORDS)
    return q

# E5 models expect "query: " prefix. MiniLM does not.
def _maybe_prefix_query(q: str) -> str:
    return f"query: {q}" if "e5" in EMB_MODEL.lower() else q

def dense_search(query: str, k: int = 30) -> List[Tuple[int, float]]:
    qn = _maybe_prefix_query(preprocess_query(query))
    qv = embedder.encode([qn], normalize_embeddings=True)
    sims, idxs = faiss_index.search(np.asarray(qv, dtype="float32"), k)
    return list(zip(idxs[0].tolist(), sims[0].tolist()))

def sparse_search(query: str, k: int = 30) -> List[Tuple[int, float]]:
    qn = preprocess_query(query)
    qv = tfidf.transform([qn])
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
    """
    alpha ∈ [0,1]: weight for dense score after min-max normalization.
    Returns FUSE_RETURN fused results (bigger pool for CE).
    """
    fetch_k = max(FUSE_RETURN, int(FUSE_FETCH_MULT * FUSE_RETURN))

    # fetch more from each side
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
    return fused[:FUSE_RETURN]   # <-- bigger pool before CE


# 2.4) Advanced RAG — Re-Ranking with a Cross-Encoder

# In[17]:


# Load a small cross-encoder and use it to re-rank the fused candidates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # fast & tiny
ce_tok   = AutoTokenizer.from_pretrained(RERANKER)
ce_model = AutoModelForSequenceClassification.from_pretrained(RERANKER).to(DEVICE)
ce_model.eval()

@torch.no_grad()
def rerank_with_cross_encoder(query: str, candidates: List[Dict], topk: int = 6, return_probs: bool = True) -> List[Dict]:
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    batch = ce_tok.batch_encode_plus(
        pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(DEVICE)
    logits = ce_model(**batch).logits

    # Handle regression (1-d) vs classification (2-d) heads
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


# build ticker↔company maps (from df_fin)

# In[18]:


# Build maps once from your tidy dataframe
TICKER_TO_COMPANY = {}
COMPANY_TO_TICKER = {}

if "df_fin" in globals():
    for t, grp in df_fin.groupby("ticker"):
        name = str(grp["company"].iloc[0]).strip()
        if name:
            TICKER_TO_COMPANY[t] = name
            COMPANY_TO_TICKER[name.lower()] = t

# Keep any manual aliases you already added
# NAME_ALIASES should already exist from your earlier alias cell; if not, create a minimal one:
NAME_ALIASES = NAME_ALIASES if "NAME_ALIASES" in globals() else {
    "automatic data processing": "ADP",
    "adp": "ADP",
    "affirm": "AFRM",
    "affirm holdings": "AFRM",
    "aehr": "AEHR",
    "aehr test systems": "AEHR",
    "america great health": "AAGH",
    "aagh": "AAGH",
}


# canonicalize query entities (alias → ticker [+ company])

# In[19]:


import re

def canonicalize_query_entities(q: str) -> str:
    """
    Replace alias/company mentions with the canonical 'TICKER CompanyName' prefix,
    then append the original task (e.g., 'revenue 2024').
    """
    q_raw = q or ""
    q_low = q_raw.lower()

    # Try existing parse_query if defined; else do light alias pass
    ticker = None
    if "parse_query" in globals():
        t, y = parse_query(q_raw)
        ticker = t

    if not ticker:
        # check short aliases / company names
        for alias, tk in NAME_ALIASES.items():
            if (len(alias) <= 6 and re.search(rf"\b{re.escape(alias)}\b", q_low)) or (len(alias) > 6 and alias in q_low):
                ticker = tk
                break

    if ticker:
        company = TICKER_TO_COMPANY.get(ticker, "")
        prefix = f"{ticker} {company}".strip()
        # Remove obvious alias strings so we don't duplicate
        for alias, tk in NAME_ALIASES.items():
            if tk == ticker:
                q_low = re.sub(rf"\b{re.escape(alias)}\b", "", q_low)
        q_low = re.sub(r"\s+", " ", q_low).strip()
        canonical = f"{prefix} {q_low}".strip()
        return canonical

    # Fallback: unchanged
    return q_raw


# augment passages for re-ranking (prepend ticker + company)

# In[20]:


def augment_passage_for_rerank(p: dict) -> str:
    """
    Prepend 'TICKER CompanyName' to the passage text seen by the cross-encoder
    to increase lexical overlap with alias queries.
    """
    ticker = p.get("ticker", "")
    company = TICKER_TO_COMPANY.get(ticker, "")
    header = f"{ticker} {company}".strip()
    text = p.get("text", "")
    return (header + "\n" + text).strip()


# use canonical query + augmented passage in re-ranker

# In[21]:


# If you used a function named rerank_with_cross_encoder, patch it; else define it.
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# (Re)load once if not present
if "ce_model" not in globals():
    RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ce_tok   = AutoTokenizer.from_pretrained(RERANKER)
    ce_model = AutoModelForSequenceClassification.from_pretrained(RERANKER).to(DEVICE)
    ce_model.eval()

@torch.no_grad()
def rerank_with_cross_encoder(query: str, candidates: list, topk: int = 6, return_probs: bool = True) -> list:
    if not candidates:
        return []
    # canonicalize query entities (alias -> TICKER Company)
    q_can = canonicalize_query_entities(query)

    # augment passages with 'TICKER Company' header
    pairs = [(q_can, augment_passage_for_rerank(c)) for c in candidates]

    batch = ce_tok.batch_encode_plus(
        pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(DEVICE)
    logits = ce_model(**batch).logits

    # handle regression vs 2-class heads
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


# (optional) also use canonical query for dense/sparse search
# 
# This helps TF-IDF & embeddings when users type the long company name.

# In[22]:


# Wrap your existing search fns if they exist
_orig_dense_search  = dense_search
_orig_sparse_search = sparse_search

def dense_search(query: str, k: int = 30):
    q_can = canonicalize_query_entities(query)
    # preserve your E5 prefixing inside _maybe_prefix_query if applicable
    qn = _maybe_prefix_query(preprocess_query(q_can))
    qv = embedder.encode([qn], normalize_embeddings=True)
    sims, idxs = faiss_index.search(np.asarray(qv, dtype="float32"), k)
    return list(zip(idxs[0].tolist(), sims[0].tolist()))

def sparse_search(query: str, k: int = 30):
    q_can = canonicalize_query_entities(query)
    qv = tfidf.transform([preprocess_query(q_can)])
    sims = cosine_similarity(qv, Xsp)[0]
    top = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in top]


# Add company aliases → map natural-language names to tickers

# In[23]:


# Alias-aware query parser (drop-in patch)
import re

# 1) Build name→ticker from your dataset (df_fin) if available
COMPANY_NAME_TO_TICKER = {}
if "df_fin" in globals():
    # take first company name per ticker
    for t, grp in df_fin.groupby("ticker"):
        nm = str(grp["company"].iloc[0])
        if nm:
            COMPANY_NAME_TO_TICKER[nm.lower()] = t

# 2) Manual aliases for your four tickers
#    Feel free to add more terms users might type.
NAME_ALIASES = {
    # AAGH
    "america great health": "AAGH",
    "aagh": "AAGH",

    # ADP
    "automatic data processing": "ADP",
    "adp": "ADP",
    "adp inc": "ADP",
    "adp payroll": "ADP",
    "adp workforce now": "ADP",

    # AEHR
    "aehr": "AEHR",
    "aehr test": "AEHR",
    "aehr test systems": "AEHR",

    # AFRM
    "affirm": "AFRM",
    "affirm holdings": "AFRM",
    "affirm inc": "AFRM",
    "buy now pay later affirm": "AFRM",
    "bnpl affirm": "AFRM",
}

# Merge company names from the dataset into aliases (non-destructive)
for nm, tk in COMPANY_NAME_TO_TICKER.items():
    NAME_ALIASES.setdefault(nm, tk)

# Optionally, restrict ticker detection to your curated set
KNOWN_TICKERS = set(['AAGH', 'ADP', 'AEHR', 'AFRM'])

def parse_query(query: str):
    """
    Returns (ticker, year) parsed from a user query.
    Priority: explicit uppercase ticker token -> alias match -> company name substring.
    """
    q_raw = query or ""
    q_low = q_raw.lower()

    # 1) explicit uppercase ticker tokens (AAGH, ADP, AEHR, AFRM)
    tickers_in_text = re.findall(r"\b([A-Z]{2,6})\b", q_raw)
    for t in tickers_in_text:
        if not KNOWN_TICKERS or t in KNOWN_TICKERS:
            ticker = t
            break
    else:
        ticker = None

    # 2) alias matches with word boundaries (e.g., "adp", "affirm")
    if not ticker:
        for alias, tk in NAME_ALIASES.items():
            # Use word boundaries for short aliases; substring for long company names
            if len(alias) <= 6:
                if re.search(rf"\b{re.escape(alias)}\b", q_low):
                    ticker = tk; break
            else:
                if alias in q_low:
                    ticker = tk; break

    # 3) (already covered by 2) but kept as fallback: company name substring map
    if not ticker:
        for nm, tk in COMPANY_NAME_TO_TICKER.items():
            if nm in q_low:
                ticker = tk; break

    # Year (first 20xx in the query)
    m_year = re.search(r"\b(20\d{2})\b", q_raw)
    year = int(m_year.group(1)) if m_year else None

    return ticker, year

# --- quick smoke tests ---
tests = [
    "affirm 2024 revenue",
    "automatic data processing balance sheet 2023",
    "aehr operating cash flow 2024",
    "america great health net income in 2024",
    "ADP revenue 2024",
]
for q in tests:
    print(q, "->", parse_query(q))


# In[24]:


# Make sure parse_query(...) from the alias cell is defined.
# Also make sure you have: hybrid_retrieve(...), rerank_with_cross_encoder(...)
from typing import Optional, List, Dict

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

# very small keyword → canonical metric detector
def detect_metric(query: str) -> Optional[str]:
    q = query.lower()
    # check multiword keys first
    for key in ["operating cash flow", "gross profit", "net income", "operating income",
                "total assets", "total liabilities", "shareholders equity"]:
        if key in q:
            return key
    # singles
    if "revenue" in q or "sales" in q:
        return "revenue"
    return None

def retrieve_contexts(query: str, k_ctx: int = 6, alpha: float = 0.6):
    want_ticker, want_year = parse_query(query) if "parse_query" in globals() else (None, None)
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


# 2.5) Answering — deterministic numeric lookup + RAG fallback (with citations)

# In[25]:


# Minimal, dependency-free answerer:
#  - parse (ticker, year) with your parse_query(...)
#  - map metric aliases ("revenue", "operating cash flow", "total assets", "net income", etc.)
#  - look up the exact value in df_fin
#  - if not found, fall back to top RAG contexts and summarize

# --- Helpers reused ---

# map friendly metric names -> list of df_fin line_item regexes to try (order matters)
METRIC_PATTERNS = {
    "revenue": [
        r"^revenue$", r"^total revenue$", r"net sales", r"sales revenue"
    ],
    "net income": [
        r"^net income$", r"net profit", r"profit attributable", r"earnings$"
    ],
    "gross profit": [
        r"^gross profit$"
    ],
    "operating income": [
        r"^operating income$", r"^operating profit$"
    ],
    "operating cash flow": [
        r"^net cash from operating activities$", r"operating cash flow", r"cash flow from operations"
    ],
    "total assets": [
        r"^total assets$"
    ],
    "total liabilities": [
        r"^total liabilities$"
    ],
    "shareholders equity": [
        r"^shareholders'? equity$", r"^stockholders'? equity$", r"^total equity$"
    ],
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




# 2.6) Numeric unit normalization (e.g., “thousands” → USD base)

# In[26]:


# Parse common unit strings and normalize all numbers to base units (USD, shares, etc.)
# We do NOT do FX conversion here—only scale (thousand, million, billion, trillion).
import re
from typing import Tuple, Optional

_UNIT_SCALE_PATTERNS = [
    (r"\b(thousand|k|000s)\b", 1e3),
    (r"\b(million|mn|mm|m)\b", 1e6),
    (r"\b(billion|bn|b)\b",    1e9),
    (r"\b(trillion|tn|t)\b",   1e12),
]

def _detect_scale(unit_str: str) -> float:
    u = (unit_str or "").lower().strip()
    # Parenthetical variants like "USD (thousands)"
    if "thousand" in u or "(thousand" in u: return 1e3
    if "million"  in u or "(million"  in u: return 1e6
    if "billion"  in u or "(billion"  in u: return 1e9
    if "trillion" in u or "(trillion" in u: return 1e12
    # Compact markers (bn, mn, mm, k, etc.)
    for pat, mul in _UNIT_SCALE_PATTERNS:
        if re.search(pat, u):
            return mul
    return 1.0

def _detect_currency(unit_str: str) -> str:
    # Extract a currency code when present (default to USD).
    u = (unit_str or "").upper()
    m = re.search(r"\b(USD|EUR|GBP|JPY|CNY|CAD|AUD)\b", u)
    return m.group(1) if m else "USD"

def normalize_value(value: float, unit_str: str) -> Tuple[float, str]:
    """
    Returns (normalized_value, normalized_unit). Example:
      (19.2, 'USD billion') -> (19_200_000_000, 'USD')
      (4157, 'USD thousands') -> (4_157_000, 'USD')
    """
    mul = _detect_scale(unit_str)
    cur = _detect_currency(unit_str)
    return float(value) * mul, cur

# Quick sanity checks
tests = [
    (1, "USD (thousands)"),
    (2.5, "usd million"),
    (3, "USD bn"),
    (4.2, "eur billion"),
    (500, "USD"),
]
for v,u in tests:
    nv, nu = normalize_value(v,u)
    print(f"{v} {u}  ->  {nv:.0f} {nu}")


# 2.6.1) Patch the lookup to return normalized values (drop-in replacement)

# In[27]:


# Replace your earlier lookup_metric helpers with these updated versions.

# --- SAFE lookup (no chained assignment) ---
def lookup_metric(df_fin: pd.DataFrame, ticker: str, year: int, metric: str):
    if metric not in METRIC_PATTERNS:
        return None

    sub = df_fin.loc[
        (df_fin["ticker"] == ticker) & (df_fin["fiscal_year"] == year)
    ].copy()
    if sub.empty:
        return None

    for pat in METRIC_PATTERNS[metric]:
        m = sub.loc[sub["line_item"].str.contains(pat, case=False, regex=True)].copy()
        if not m.empty:
            row = (
                m.assign(abs_val=m["value"].abs())   # <- no direct m["abs_val"] write
                 .nlargest(1, "abs_val")
                 .iloc[0]
            )
            norm_val, norm_unit = normalize_value(row["value"], row["unit"])
            return {
                "line_item": row["line_item"],
                "value": float(row["value"]),     # raw
                "unit": row["unit"],              # raw
                "value_norm": norm_val,           # normalized
                "unit_norm": norm_unit,           # normalized unit
                "statement": row["statement"],
            }
    return None

# And update answer_query to use value_norm/unit_norm when present
def answer_query(query: str, k_ctx: int = 5, alpha: float = 0.6) -> Dict:
    ticker, year = parse_query(query) if "parse_query" in globals() else (None, None)
    metric = detect_metric(query)

    direct = None
    if ticker and year:
        if metric:
            direct = lookup_metric(df_fin, ticker, year, metric)
        # NEW: try full line_item match if metric lookup failed
        if direct is None:
            li = extract_line_item_from_question(query, ticker, year, df_fin)
            if li:
                # exact/fuzzy line_item lookup (normalized)
                sub = df_fin.loc[
                    (df_fin["ticker"]==ticker) & (df_fin["fiscal_year"]==year) &
                    (df_fin["line_item"].str.contains(re.escape(li), case=False, regex=True))
                ].copy()
                if not sub.empty:
                    row = (sub.assign(abs_val=sub["value"].abs())
                             .nlargest(1, "abs_val").iloc[0])
                    norm_val, norm_unit = normalize_value(row["value"], row["unit"])
                    direct = {
                        "line_item": row["line_item"],
                        "value_norm": norm_val,
                        "unit_norm": norm_unit,
                        "statement": row["statement"],
                    }

    # always retrieve for citations/fallback
    ctxs = retrieve_contexts(query, k_ctx=k_ctx, alpha=alpha)

    if direct:
        val_str = fmt_money(direct["value_norm"])
        return {
            "answer": f"{ticker} {year} { (metric.title() if metric else direct['line_item']) }: {val_str} ({direct['unit_norm']}).",
            "source": "structured_lookup",
            "ticker": ticker, "year": year,
            "metric": (metric or direct["line_item"]),
            "value_norm": direct["value_norm"], "unit_norm": direct["unit_norm"],
            "statement": direct["statement"],
            "citations": citations_from_contexts(ctxs)
        }

    # fallback summary
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


# Improving numerical accuracy

# In[28]:


import re
from collections import Counter

def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 &/\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _token_set(s: str) -> set:
    toks = [t for t in _normalize(s).split() if t not in {"the","a","an","of","and","to","in"}]
    return set(toks)

def extract_line_item_from_question(q: str, ticker: str, year: int, df: pd.DataFrame) -> str | None:
    """
    Try to pull the line-item phrase from the question and map it to the closest df_fin line_item.
    Handles both:
      - "What was TICKER's <item> in 2024?"
      - "In 2024, what was TICKER's <item> (Balance Sheet)?"
    """
    qn = _normalize(q)
    # strip ticker & year words to leave the item phrase behind
    qn = qn.replace(_normalize(ticker), " ")
    qn = re.sub(r"\b20\d{2}\b", " ", qn)
    qn = re.sub(r"\b(?:what|was|were|company|s)\b", " ", qn)
    qn = re.sub(r"\s+", " ", qn).strip()

    # candidate item tokens from question
    qset = _token_set(qn)
    if not qset:
        return None

    # compare with all line_items for that ticker+year (token Jaccard overlap)
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
    return best if best_score >= 0.3 else None  # threshold can be tuned

import json
from statistics import mean
import pandas as pd

# Load your Q/A pairs (created in Step-1). If you kept them in memory, you can skip the file load.
# OLD
# QA_PATH = "out/qa_pairs.jsonl"
# NEW
QA_PATH = os.path.join(OUT_DIR, "qa_pairs.jsonl")

qa_pairs_eval = []
with open(QA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        qa_pairs_eval.append(json.loads(line))

        
def fmt_money(val: float) -> str:
    v = float(val); a = abs(v)
    if a >= 1e12: return f"${v/1e12:.2f} trillion"
    if a >= 1e9:  return f"${v/1e9:.2f} billion"
    if a >= 1e6:  return f"${v/1e6:.2f} million"
    if a >= 1e3:  return f"${v/1e3:.0f} thousand"
    return f"${v:.0f}"

# OLD
# df_fin = pd.read_csv("out/financials_last2y.csv")
# NEW
df_fin = pd.read_csv(os.path.join(OUT_DIR, "financials_last2y.csv"))

# --- SAFE ground truth (no chained assignment) ---
def ground_truth_value(df: pd.DataFrame, ticker: str, year: int, line_item: str):
    sub = df.loc[
        (df["ticker"] == ticker) &
        (df["fiscal_year"] == year) &
        (df["line_item"] == line_item)
    ].copy()

    if sub.empty:
        sub = df.loc[
            (df["ticker"] == ticker) &
            (df["fiscal_year"] == year) &
            (df["line_item"].str.contains(re.escape(line_item), case=False, regex=True))
        ].copy()
        if sub.empty:
            return None

    row = (
        sub.assign(abs_val=sub["value"].abs())   # <- no direct sub["abs_val"] write
           .nlargest(1, "abs_val")
           .iloc[0]
    )
    v_norm, u_norm = normalize_value(row["value"], row["unit"])
    return v_norm, u_norm



# 2.6.1) Config & helpers (banned terms, PII masking, allowed tickers)

# In[30]:


# --- Guardrail configuration & helpers ---
import re
import pandas as pd

# Your curated universe
ALLOWED_TICKERS = set(['AAGH', 'ADP', 'AEHR', 'AFRM'])

# Very compact "clearly-bad" input patterns (POC-grade; keep conservative)
BANNED_PATTERNS = [
    r"\bkill\b", r"\bsuicide\b", r"\bviolence\b",
    r"\bhack(?:ing)?\b", r"\bmalware\b", r"\bransomware\b",
]

# Finance keywords: require at least one if no ticker is detected
FINANCE_KEYWORDS = [
    "revenue","income","profit","loss","assets","liabilities","equity",
    "cash flow","operating cash","capex","balance sheet","income statement",
    "dividend","ebit","ebitda","net income","total assets","total liabilities"
]

# Simple PII masking (emails, phone-like, SSN-like) - best-effort only
def mask_pii(text: str) -> str:
    t = text
    t = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[email]", t)
    t = re.sub(r"\b(?:\+?\d[\d \-\(\)]{7,}\d)\b", "[phone]", t)
    t = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[ssn]", t)   # US SSN pattern
    return t


# 2.6.2) Input guard: validate_query(query)

# In[31]:


def validate_query(query: str) -> dict:
    """
    Returns:
      {
        ok: bool,
        sanitized_query: str,  # PII-masked
        reason: str | None,
        parsed_ticker: str | None,
        parsed_year: int | None
      }
    """
    q_raw = (query or "").strip()
    if not q_raw:
        return {"ok": False, "sanitized_query": "", "reason": "empty query", "parsed_ticker": None, "parsed_year": None}

    # mask PII before doing anything else
    q = mask_pii(q_raw)

    # hard-ban obvious harmful intents (keep this narrow)
    for pat in BANNED_PATTERNS:
        if re.search(pat, q, flags=re.I):
            return {"ok": False, "sanitized_query": q, "reason": "harmful intent detected", "parsed_ticker": None, "parsed_year": None}

    # parse expected entities
    tkr, yr = parse_query(q) if "parse_query" in globals() else (None, None)

    # require a supported ticker OR a finance keyword (keeps PoC focused)
    has_fin_kw = any(kw in q.lower() for kw in FINANCE_KEYWORDS)
    if tkr is None and not has_fin_kw:
        return {"ok": False, "sanitized_query": q, "reason": "no ticker or finance keyword found", "parsed_ticker": None, "parsed_year": None}

    # if a ticker is present, require it to be in the allowed set
    if tkr is not None and tkr not in ALLOWED_TICKERS:
        return {"ok": False, "sanitized_query": q, "reason": f"unsupported ticker: {tkr}", "parsed_ticker": tkr, "parsed_year": yr}

    return {"ok": True, "sanitized_query": q, "reason": None, "parsed_ticker": tkr, "parsed_year": yr}


# 2.6.3) Output guard: verify numbers vs. ground-truth & flag risky cases

# In[32]:


# Reuse your normalization helpers
def extract_numeric_from_text(text: str):
    """
    Finds the first '$X [thousand/million/billion/trillion]?' pattern and returns a base-USD float, else None.
    """
    m = re.search(r"\$([0-9]+(?:\.[0-9]+)?)\s*(thousand|million|billion|trillion)?", text, flags=re.I)
    if not m:
        return None
    x = float(m.group(1))
    mul = {"thousand":1e3,"million":1e6,"billion":1e9,"trillion":1e12}.get((m.group(2) or "").lower(), 1.0)
    return x * mul

def ground_truth_value(df: pd.DataFrame, ticker: str, year: int, metric_or_line: str):
    """
    Looks up GT value for either a known metric (via your lookup_metric),
    or a line_item string contained in df_fin.
    Returns (value_norm) or None.
    """
    # Try metric path first
    if metric_or_line in (METRIC_PATTERNS.keys()):
        hit = lookup_metric(df, ticker, year, metric_or_line)
        if hit: return float(hit["value_norm"])

    # Fallback: treat as a line_item
    sub = df.loc[(df["ticker"]==ticker) & (df["fiscal_year"]==year) &
                 (df["line_item"].str.contains(re.escape(metric_or_line), case=False, regex=True))].copy()
    if sub.empty:
        return None
    row = (sub.assign(abs_val=sub["value"].abs())
             .nlargest(1, "abs_val").iloc[0])
    v_norm, _ = normalize_value(row["value"], row["unit"])
    return float(v_norm)

def verify_output(answer_obj: dict, tol: float = 0.02) -> dict:
    """
    Adds guardrail fields:
      - flagged: bool
      - flag_reason: str | None
      - confidence: float (crude heuristic)
    """
    flagged = False
    reason  = None
    conf    = 0.5

    # Trust structured lookup more than RAG
    if answer_obj.get("source") == "structured_lookup":
        conf = 0.9
        # sanity: re-compare with GT in case of unexpected mismatch
        t, y = answer_obj.get("ticker"), answer_obj.get("year")
        label = answer_obj.get("metric")  # metric name or line_item
        pred  = float(answer_obj.get("value_norm", 0.0))
        gt    = ground_truth_value(df_fin, t, y, label) if (t and y and label) else None
        if gt is not None:
            denom = max(1.0, abs(gt), abs(pred))
            if abs(pred - gt) / denom > tol:
                flagged, reason, conf = True, "numeric mismatch vs ground-truth", 0.2
    else:
        # RAG fallback → lower confidence; extract number (if any) & compare
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
                    conf = 0.7  # matched
        else:
            # No numeric claim → leave as informational, but mark as low confidence
            reason = "no verifiable number in output" if label else "no structured mapping"
            conf   = 0.4

    out = dict(answer_obj)
    out["flagged"] = flagged
    out["flag_reason"] = reason
    out["confidence"] = float(conf)
    return out


# 2.6.4) Guarded wrapper: answer_query_guarded(query)

# In[33]:


def answer_query_guarded(query: str, k_ctx: int = 5, alpha: float = 0.6) -> dict:
    """
    Full pipeline with guardrails:
      1) Input validation (block clearly-bad/irrelevant; mask PII)
      2) Answer (your existing answer_query)
      3) Output verification (flag hallucinations/mismatches)
    """
    # 1) input guard
    v = validate_query(query)
    if not v["ok"]:
        return {
            "blocked": True,
            "reason": v["reason"],
            "sanitized_query": v["sanitized_query"],
        }

    # 2) compute answer using sanitized text (preserves PII masking)
    ans = answer_query(v["sanitized_query"], k_ctx=k_ctx, alpha=alpha)

    # 3) output verification
    verified = verify_output(ans, tol=0.02)
    return {
        "blocked": False,
        "query_sanitized": v["sanitized_query"],
        "parsed_ticker": v["parsed_ticker"],
        "parsed_year": v["parsed_year"],
        "result": verified,
    }








# FT

# 3.1 Q/A Dataset Preparation
# 
# We reuse the ~50+ Q/A pairs from Step-1 (out/qa_pairs.jsonl), and canonicalize the answers to base-USD integers (no commas). This simplifies learning and evaluation. We keep an 80/10/10 train/val/test split

# In[35]:


# 3.1 — Load Q/A, strict numeric targets, split

import os, json, random, re
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# --- Prereqs expected from Step-1 ---

_UNIT_SCALE_PATTERNS = [
    (r"\b(thousand|k|000s)\b", 1e3),
    (r"\b(million|mn|mm|m)\b", 1e6),
    (r"\b(billion|bn|b)\b",    1e9),
    (r"\b(trillion|tn|t)\b",   1e12),
]

def _detect_currency(unit_str: str) -> str:
    # Extract a currency code when present (default to USD).
    u = (unit_str or "").upper()
    m = re.search(r"\b(USD|EUR|GBP|JPY|CNY|CAD|AUD)\b", u)
    return m.group(1) if m else "USD"

def _detect_scale(unit_str: str) -> float:
    u = (unit_str or "").lower().strip()
    # Parenthetical variants like "USD (thousands)"
    if "thousand" in u or "(thousand" in u: return 1e3
    if "million"  in u or "(million"  in u: return 1e6
    if "billion"  in u or "(billion"  in u: return 1e9
    if "trillion" in u or "(trillion" in u: return 1e12
    # Compact markers (bn, mn, mm, k, etc.)
    for pat, mul in _UNIT_SCALE_PATTERNS:
        if re.search(pat, u):
            return mul
    return 1.0
def normalize_value(value: float, unit_str: str) -> Tuple[float, str]:
    """
    Returns (normalized_value, normalized_unit). Example:
      (19.2, 'USD billion') -> (19_200_000_000, 'USD')
      (4157, 'USD thousands') -> (4_157_000, 'USD')
    """
    mul = _detect_scale(unit_str)
    cur = _detect_currency(unit_str)
    return float(value) * mul, cur

# OLD
# df_fin = pd.read_csv("out/financials_last2y.csv")
# NEW
df_fin = pd.read_csv(os.path.join(OUT_DIR, "financials_last2y.csv"))



# Light-weight numeric parser fallback (only used if strict lookup fails)
SCALE_ALIAS = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "mm": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12,
}
def extract_numeric_from_text(text: str) -> Optional[float]:
    if not text: return None
    s = text.strip()
    m = re.search(r"\$?\s*([0-9][0-9,]*\.?[0-9]*)\s*([kmbtn]|thousand|million|billion|trillion|mn|mm|bn|tn)?\b", s, re.I)
    if m:
        num = float(m.group(1).replace(",", ""))
        scale = (m.group(2) or "").lower()
        return num * SCALE_ALIAS.get(scale, 1.0)
    m = re.search(r"\b([0-9][0-9,]{2,})\b", s)
    return float(m.group(1).replace(",", "")) if m else None

# --- Q/A load & split ---
# OLD
# QA_PATH = "out/qa_pairs.jsonl"
# NEW
QA_PATH = os.path.join(OUT_DIR, "qa_pairs.jsonl")

assert os.path.exists(QA_PATH), f"Missing {QA_PATH} (from Step-1)."

with open(QA_PATH, "r", encoding="utf-8") as f:
    qa_all = [json.loads(line) for line in f]

qa_all = [x for x in qa_all if x.get("question") and x.get("answer")]
random.seed(42); random.shuffle(qa_all)
n = len(qa_all); n_train = max(1, int(0.8*n)); n_val = max(1, int(0.1*n))
train_raw = qa_all[:n_train]; val_raw = qa_all[n_train:n_train+n_val]; test_raw = qa_all[n_train+n_val:]
print(f"[data] total={n} | train={len(train_raw)} | val={len(val_raw)} | test={len(test_raw)}")

# --- Canonical line-item matching rules (strict) ---
CANON_PATTERNS = {
    "revenue":               [r"^revenue$", r"^total revenue$"],
    "cost of revenue":       [r"^cost of revenue$"],
    "gross profit":          [r"^gross profit$"],
    "net income":            [r"^net income$"],
    "operating cash flow":   [r"^net cash from operating activities$", r"^cash provided by operating activities$"],
    "total assets":          [r"^total assets$"],
    "total liabilities":     [r"^total liabilities$"],  # excludes liabilities & equity
    "shareholders equity":   [r"^total shareholders'? equity$", r"^total stockholders'? equity$", r"^shareholders'? equity$"],
    "cash & equivalents":    [r"^cash( and)? cash equivalents$",
                              r"^cash,?\s*cash equivalents$"],
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
    # 1) exact fullmatch on raw string
    m = df2[df2["line_item"].str.fullmatch(re.escape(wanted), case=False, na=False)]
    if not m.empty: return m.assign(abs_val=m["value"].abs()).nlargest(1, "abs_val").iloc[0]
    # 2) canonical synonyms
    for patt in CANON_PATTERNS.get(wanted_norm, []):
        m = df2[df2["line_item"].str.match(patt, case=False, na=False)]
        if not m.empty: return m.assign(abs_val=m["value"].abs()).nlargest(1, "abs_val").iloc[0]
    # 3) word-boundary contains
    patt = r"\b" + re.escape(wanted) + r"\b"
    m = df2[df2["line_item"].str.contains(patt, case=False, na=False)]
    if not m.empty: return m.assign(abs_val=m["value"].abs()).nlargest(1, "abs_val").iloc[0]
    # 4) last resort: loose contains
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
    # Fallback: parse numeric from provided answer text
    gold_num = extract_numeric_from_text(row.get("answer",""))
    return int(round(gold_num)) if gold_num is not None else None

def build_numeric_pairs_strict(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        val = canonical_numeric_answer_strict(r)
        if val is None: continue
        out.append({"q": r["question"], "y": val, "ticker": r.get("ticker"), "year": r.get("year"), "line_item": r.get("line_item")})
    return out

train = build_numeric_pairs_strict(train_raw)
val   = build_numeric_pairs_strict(val_raw)
test  = build_numeric_pairs_strict(test_raw)
print(f"[numeric(strict)] train={len(train)} | val={len(val)} | test={len(test)}")


# In[36]:


# 3.1d–3.1e — Clean + canon map → build augmented, scaled signed-log targets

import re, math, numpy as np, pandas as pd

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _canon_key(li: str) -> str:
    x = _norm(li)
    # Drop noisy CF events / financing items for regression
    if re.search(r"\bcash (from|used in)\b", x) or "repurchase" in x or "issuance of" in x:
        return "drop_cf_event"
    # High-priority exact metrics
    if re.search(r"\bcost of revenue\b", x):                                  return "cost of revenue"
    if re.search(r"\bgross profit\b", x):                                     return "gross profit"
    if re.search(r"\bnet income\b", x) and "starting line" not in x:          return "net income"
    if re.search(r"\b(total )?revenue\b", x):                                 return "revenue"
    if re.search(r"\bnet cash (provided by|from) operating activities\b", x): return "operating cash flow"
    if re.search(r"\bcash (and|&) cash equivalents\b", x) or \
       re.search(r"\bcash,?\s*cash equivalents\b", x):                        return "cash & equivalents"
    if re.search(r"\btotal assets\b", x):                                     return "total assets"
    if re.search(r"\btotal liabilities\b", x) and "equity" not in x:          return "total liabilities"
    if re.search(r"\b(total (shareholders|stockholders)'? equity|shareholders'? equity)\b", x):
        return "shareholders equity"
    if re.search(r"\b(capital expenditures|purchases of property, plant and equipment)\b", x):
        return "capex"
    # Flat/irrelevant for regression here
    if "starting line" in x or x.startswith("shares ("):
        return "drop_noisy"
    return "unknown"

# Keep reliable metrics only
ALLOWED = {
    "revenue", "cost of revenue", "gross profit", "net income",
    "operating cash flow", "total assets", "total liabilities",
    "shareholders equity", "cash & equivalents", "capex"
}
BLOCKLIST = {"drop_cf_event", "drop_noisy", "unknown"}

# Low-variance filter in log space (removes near-constant artifact groups)
df_all = pd.DataFrame(train + val + test).copy()
df_all["canon_key"] = df_all["line_item"].apply(_canon_key)
tmp = df_all.copy()
tmp["log_abs_y"] = np.log10(1.0 + tmp["y"].astype(float).abs())
var = (tmp.groupby("canon_key")["log_abs_y"]
          .agg(mu="mean", sigma="std", n="count")
          .reset_index())
LOW_VAR_SIGMA = 0.12
low_var_keys = set(var.loc[var["sigma"].fillna(0.0) < LOW_VAR_SIGMA, "canon_key"])

DROP_KEYS = (low_var_keys | BLOCKLIST) - ALLOWED
print("[filter] low-variance keys:", sorted(low_var_keys))
print("[filter] drop keys:",        sorted(DROP_KEYS))

def _keep_row(r: dict) -> bool:
    k = _canon_key(r.get("line_item"))
    return (k in ALLOWED) and (k not in DROP_KEYS)

train = [r for r in train if _keep_row(r)]
val   = [r for r in val   if _keep_row(r)]
test  = [r for r in test  if _keep_row(r)]
print(f"[filter] kept -> train={len(train)} | val={len(val)} | test={len(test)}")

# Per-metric scales (USD)
ITEM_SCALE = {
    "revenue": 1e9, "cost of revenue": 1e9, "gross profit": 1e9, "net income": 1e9,
    "operating cash flow": 1e9, "total assets": 1e9, "total liabilities": 1e9,
    "shareholders equity": 1e9, "cash & equivalents": 1e9, "capex": 1e6
}

def _augment_question(r: dict) -> str:
    t = r.get("ticker") or ""
    y = r.get("year")
    li = (r.get("line_item") or "").strip()
    tag_t = f"[TICKER={t}]" if t else ""
    tag_y = f"[YEAR={int(y)}]" if y is not None else ""
    tag_li = f"[ITEM={li}]" if li else ""
    base_q = r.get("q") or r.get("question")
    return " ".join([tag_t, tag_y, tag_li, base_q]).strip()

def _to_log10_signed_scaled(y_usd: float, scale: float) -> float:
    sign = 1.0 if y_usd >= 0 else -1.0
    return sign * math.log10(1.0 + abs(float(y_usd))/float(scale))

def _canon_for_scale(li: str) -> str:
    k = _canon_key(li)
    return k if k in ITEM_SCALE else "revenue"  # default billions

def _build_aug_log_scaled(rows):
    out = []
    for r in rows:
        y  = float(r["y"])
        ck = _canon_for_scale(r.get("line_item") or "")
        sc = ITEM_SCALE[ck]
        q_aug = _augment_question(r)
        z = _to_log10_signed_scaled(y, sc)
        out.append({
            "q": q_aug, "y": y, "z": float(z), "scale": float(sc),
            "ticker": r.get("ticker"), "year": r.get("year"),
            "line_item": r.get("line_item"), "canon_key": ck
        })
    return out

train_z = _build_aug_log_scaled(train)
val_z   = _build_aug_log_scaled(val)
test_z  = _build_aug_log_scaled(test)

z_list = [x["z"] for x in train_z]
print(f"[rebuild(clean)] train_z={len(train_z)} val_z={len(val_z)} test_z={len(test_z)}")
if z_list:
    print(f"[z stats] mean={np.mean(z_list):.3f} std={np.std(z_list, ddof=1) if len(z_list)>1 else 0.0:.3f} "
          f"min={min(z_list):.3f} max={max(z_list):.3f}")
    print("[sample]", train_z[0] if train_z else "—")


# In[37]:


# 3.2 — Encoder setup (MiniLM), device & mean-pool

import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load once (reuse if already loaded earlier)
try:
    enc
    tok
    HIDDEN
except NameError:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    HIDDEN = enc.config.hidden_size

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


# In[38]:


# 3.3 — Zero-shot kNN baseline

import numpy as np, torch, time

@torch.no_grad()
def embed_texts(texts, max_len=128, batch=64):
    vecs = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i+batch]
        enc_in = tok(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
        out = enc(**enc_in, return_dict=True)
        pooled = mean_pool(out.last_hidden_state, enc_in["attention_mask"]).cpu().numpy()
        vecs.append(pooled)
    return np.vstack(vecs) if vecs else np.zeros((0, HIDDEN), dtype=np.float32)

# Build train bank
train_q = [r["q"] for r in train_z]
train_y = np.array([r["y"] for r in train_z], dtype=np.float64)
V_train = embed_texts(train_q)

def knn_predict(qs, k=5):
    V_q = embed_texts(qs)
    preds = []
    for v in V_q:
        sims = V_train @ v / (np.linalg.norm(V_train, axis=1)+1e-9) / (np.linalg.norm(v)+1e-9)
        idx = np.argsort(-sims)[:k]
        preds.append(np.median(train_y[idx]))
    return np.array(preds, dtype=np.float64)

def fmt_money(x):
    sgn = "-" if x < 0 else ""
    x = abs(float(x))
    if x >= 1e12: return f"{sgn}${x/1e12:.2f} trillion"
    if x >= 1e9:  return f"{sgn}${x/1e9:.2f} billion"
    if x >= 1e6:  return f"{sgn}${x/1e6:.2f} million"
    if x >= 1e3:  return f"{sgn}${x/1e3:.2f} thousand"
    return f"{sgn}${x:.0f}"

# Quick baseline eval on a small test slice
def eval_knn(sample, k=10):
    sample = sample[:k] if len(sample)>=k else sample
    qs = [r["q"] for r in sample]
    ys = np.array([r["y"] for r in sample], dtype=np.float64)
    t0 = time.time(); yhat = knn_predict(qs, k=5); ms = (time.time()-t0)*1000.0
    denom = np.maximum(1.0, np.maximum(np.abs(ys), np.abs(yhat)))
    rel = np.abs(yhat-ys)/denom
    return {"k": len(sample), "acc@10%": float((rel<=0.10).mean()), "avg_ms": ms/len(sample)}

print("[baseline kNN]", eval_knn(test_z, k=min(10, len(test_z))))


# In[39]:


# --- Repro pack: set *all* seeds & deterministic backends ---
import os, random, numpy as np, torch

SEED = 12345  # pick any integer, keep it fixed

def set_reproducible(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # CUDA/cuBLAS determinism (required for strict determinism on GPU)
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN / PyTorch determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass  # older torch

set_reproducible(SEED)


# In[40]:


from torch.utils.data import DataLoader

def seed_worker(worker_id):
    # make each worker deterministically seeded
    s = SEED + worker_id
    np.random.seed(s)
    random.seed(s)

g = torch.Generator()
g.manual_seed(SEED)


# In[41]:


# 3.4 — Z-normalization + loaders with ticker/metric/year indices

from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

# z normalization (train only)
z_train = np.array([r["z"] for r in train_z], dtype=np.float32)
Z_MEAN = float(z_train.mean()) if len(z_train) else 0.0
Z_STD  = float(z_train.std())  if len(z_train) > 1 else 1.0
if Z_STD == 0.0: Z_STD = 1.0
print(f"[z-norm] mean={Z_MEAN:.4f} std={Z_STD:.4f}")

# vocabularies
TICKER_VOCAB = sorted({r["ticker"] for r in (train_z+val_z+test_z) if r.get("ticker")})
KEY_VOCAB    = sorted({r["canon_key"] for r in (train_z+val_z+test_z)})
YEAR_VOCAB   = sorted({int(r["year"]) for r in (train_z+val_z+test_z) if r.get("year") is not None})

ticker2id = {t:i for i,t in enumerate(TICKER_VOCAB)}
key2id    = {k:i for i,k in enumerate(KEY_VOCAB)}
year2id   = {y:i for i,y in enumerate(YEAR_VOCAB)}

print("[vocab] tickers:", TICKER_VOCAB)
print("[vocab] keys:", KEY_VOCAB)
print("[vocab] years:", YEAR_VOCAB)

@dataclass
class RowZNIdxY:
    q: str; z_norm: float; y: float; scale: float
    tick_idx: int; key_idx: int; year_idx: int

def row_to_idx_y(r):
    return RowZNIdxY(
        q=r["q"],
        z_norm=(float(r["z"])-Z_MEAN)/Z_STD,
        y=float(r["y"]),
        scale=float(r["scale"]),
        tick_idx=ticker2id.get(r["ticker"], 0) if TICKER_VOCAB else 0,
        key_idx=key2id.get(r["canon_key"], 0),
        year_idx=year2id.get(int(r["year"]), 0) if YEAR_VOCAB and (r.get("year") is not None) else 0,
    )

class QRegZNIdxY(Dataset):
    def __init__(self, rows): self.rows = [row_to_idx_y(x) for x in rows]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def collate_rows_zn_idx_y(batch):
    texts  = [b.q for b in batch]
    z_norm = torch.tensor([b.z_norm for b in batch], dtype=torch.float32)
    y_usd  = torch.tensor([b.y      for b in batch], dtype=torch.float32)
    scale  = torch.tensor([b.scale  for b in batch], dtype=torch.float32)
    t_idx  = torch.tensor([b.tick_idx for b in batch], dtype=torch.long)
    k_idx  = torch.tensor([b.key_idx  for b in batch], dtype=torch.long)
    y_idx  = torch.tensor([b.year_idx for b in batch], dtype=torch.long)
    return texts, z_norm, y_usd, scale, t_idx, k_idx, y_idx

BATCH_SIZE = 16
MAX_LEN    = 128
train_loader = DataLoader(QRegZNIdxY(train_z), batch_size=BATCH_SIZE, shuffle=False,  collate_fn=collate_rows_zn_idx_y,     num_workers=0,            # easiest way to avoid worker nondeterminism
    worker_init_fn=seed_worker,
    generator=g, drop_last=False, pin_memory=False)
val_loader   = DataLoader(QRegZNIdxY(val_z),   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_rows_zn_idx_y,     num_workers=0,            # easiest way to avoid worker nondeterminism
    worker_init_fn=seed_worker,
    generator=g, drop_last=False, pin_memory=False)




# In[42]:


# === Save / Load ID maps (and z-norm + MAX_LEN) ===
import os, json, time

# OLD
# VOCABS_PATH = "out/vocabs.json"
# NEW
VOCABS_PATH = os.path.join(OUT_DIR, "vocabs.json")


def save_vocabs(path=VOCABS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        # training-time constants (helps reproducibility)
        "z_mean": float(Z_MEAN),
        "z_std": float(Z_STD),
        "max_len": int(MAX_LEN),
        # store both lists (ordered vocab) and maps
        "ticker_vocab": list(map(str, TICKER_VOCAB)),
        "key_vocab": list(map(str, KEY_VOCAB)),
        "year_vocab": list(map(int, YEAR_VOCAB)),
        "ticker2id": {str(k): int(v) for k, v in ticker2id.items()},
        "key2id":    {str(k): int(v) for k, v in key2id.items()},
        # JSON keys must be strings → cast year keys to str
        "year2id":   {str(k): int(v) for k, v in year2id.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[saved vocabs] → {path}")

def load_vocabs(path=VOCABS_PATH):
    global TICKER_VOCAB, KEY_VOCAB, YEAR_VOCAB
    global ticker2id, key2id, year2id
    global Z_MEAN, Z_STD, MAX_LEN

    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)

    # prefer stored ordered lists; fall back to keys of the maps
    TICKER_VOCAB = p.get("ticker_vocab") or list(p["ticker2id"].keys())
    KEY_VOCAB    = p.get("key_vocab")    or list(p["key2id"].keys())
    YEAR_VOCAB   = p.get("year_vocab")   or [int(k) for k in p["year2id"].keys()]

    ticker2id = {str(k): int(v) for k, v in p["ticker2id"].items()}
    key2id    = {str(k): int(v) for k, v in p["key2id"].items()}
    year2id   = {int(k): int(v) for k, v in p["year2id"].items()}

    # restore z-norm + max_len so inference matches training scaling/tokenization
    Z_MEAN = float(p.get("z_mean", 0.0))
    Z_STD  = float(p.get("z_std", 1.0)) or 1.0
    MAX_LEN = int(p.get("max_len", 128))

    print(f"[loaded vocabs] ← {path}")
    print("[sizes] tickers:", len(TICKER_VOCAB), "keys:", len(KEY_VOCAB), "years:", len(YEAR_VOCAB))
save_vocabs()   # writes out/vocabs.json


# In[43]:



# --- Minimal: load best checkpoint by absolute/relative path ---
import time, os, math
from torch import nn

# OLD
# PATH = r"out/minilm_moe_best.pt"
# NEW
PATH = os.path.join(OUT_DIR, "minilm_moe_best.pt")


# (Rebuild encoder/tokenizer exactly as in training)
# e.g. enc = AutoModel.from_pretrained(MODEL_NAME); tok = AutoTokenizer.from_pretrained(MODEL_NAME)
d_model = enc.config.hidden_size  # HIDDEN used at train time



# MoE building blocks
class LoRAExpert(nn.Module):
    """Tiny low-rank expert (down->GELU->up)."""
    def __init__(self, d_model: int, r: int = 32):
        super().__init__()
        self.down = nn.Linear(d_model, r, bias=False)
        self.up   = nn.Linear(r, d_model, bias=False)
        self.act  = nn.GELU()
    def forward(self, h): return self.up(self.act(self.down(h)))

class MoEAdapter(nn.Module):
    def __init__(self, d_model: int, num_experts: int = 6, r: int = 64):
        super().__init__()
        self.experts = nn.ModuleList([LoRAExpert(d_model, r=r) for _ in range(num_experts)])
        self.gate = nn.Sequential(nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, num_experts))
        self.drop = nn.Dropout(0.1)
    def forward(self, h):
        gate = torch.softmax(self.gate(h), dim=-1)              # [B,K]
        expert_outs = torch.stack([e(h) for e in self.experts], dim=1)  # [B,K,H]
        delta = (gate.unsqueeze(-1) * expert_outs).sum(dim=1)   # [B,H]
        return self.drop(delta), gate


class MiniLM_MoE_Regressor_Z_EmbY(nn.Module):
    """MiniLM (frozen except last block) + [ticker|key|year] emb + proj → MoE → head → z_pred"""
    def __init__(self, encoder, tokenizer, d_model: int,
                 num_tickers: int, num_keys: int, num_years: int,
                 tdim=16, kdim=8, ydim=8, num_experts=6, r=64):
        super().__init__()
        self.encoder = encoder; self.tokenizer = tokenizer
        self.emb_t = nn.Embedding(max(1,num_tickers), tdim)
        self.emb_k = nn.Embedding(max(1,num_keys),   kdim)
        self.emb_y = nn.Embedding(max(1,num_years),  ydim)
        self.proj  = nn.Linear(d_model + tdim + kdim + ydim, d_model)
        self.moe   = MoEAdapter(d_model=d_model, num_experts=num_experts, r=r)
        self.head  = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))
        # freeze encoder
        for p in self.encoder.parameters(): p.requires_grad = False

    def forward(self, texts, tick_idx, key_idx, year_idx, max_len=128):
        enc_in = self.tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.encoder(**enc_in, return_dict=True)
            pooled = mean_pool(out.last_hidden_state, enc_in["attention_mask"])  # [B,H]
        et = self.emb_t(tick_idx.to(DEVICE)) if self.emb_t.num_embeddings>0 else torch.zeros(pooled.size(0),0,device=DEVICE)
        ek = self.emb_k(key_idx.to(DEVICE))
        ey = self.emb_y(year_idx.to(DEVICE)) if self.emb_y.num_embeddings>0 else torch.zeros(pooled.size(0),0,device=DEVICE)
        h  = torch.cat([pooled, et, ek, ey], dim=-1)
        h  = self.proj(h)
        delta, _ = self.moe(h)
        z_pred = self.head(h + delta).squeeze(-1)
        return {"z_pred": z_pred}


moe_z = MiniLM_MoE_Regressor_Z_EmbY(
    enc, tok, d_model=d_model,
    num_tickers=len(TICKER_VOCAB), num_keys=len(KEY_VOCAB), num_years=len(YEAR_VOCAB),
    tdim=16, kdim=8, ydim=8, num_experts=6, r=64
).to(DEVICE)

ckpt = torch.load(PATH, map_location=DEVICE)
state = ckpt.get("model_state_dict", ckpt)  # works if file stores either dict or raw state_dict
moe_z.load_state_dict(state, strict=True)   # set strict=False if you changed the code
moe_z.eval()
print(f"Loaded best from {PATH} (saved at epoch {ckpt.get('epoch', '?')}, val={ckpt.get('val_loss', '?')})")


# In[45]:


# --- Use the fine-tuned MiniLM-MoE model for predictions in 3.7 ---

import numpy as np
import pandas as pd
import torch

# 0) sanity: we expect these to exist from earlier steps
assert 'moe_z' in globals(), "moe_z (fine-tuned MiniLM-MoE) not found"
assert 'tok' in globals(), "Tokenizer 'tok' not found"
assert 'DEVICE' in globals(), "DEVICE not set"
assert 'KEY_VOCAB' in globals() and 'TICKER_VOCAB' in globals() and 'YEAR_VOCAB' in globals(), \
       "KEY_VOCAB/TICKER_VOCAB/YEAR_VOCAB missing"
assert 'train_z' in globals(), "train_z missing (for z-scale stats)"

# 1) id maps (build if you don't already have them)
TICKER2ID = {t: i for i, t in enumerate(KEY_VOCAB if False else TICKER_VOCAB)}  # keep structure explicit
KEY2ID    = {k: i for i, k in enumerate(KEY_VOCAB)}
YEAR2ID   = {int(y): i for i, y in enumerate(YEAR_VOCAB)}

# 2) signed-log transforms (must match your training)
def to_log10_signed(y: float) -> float:
    return (1.0 if y >= 0 else -1.0) * np.log10(1.0 + abs(float(y)))

def from_log10_signed(z: float) -> float:
    # inverse of signed log10(1+|y|)
    return np.sign(z) * (10.0 ** abs(z) - 1.0)

# 3) per-metric z normalization stats μ, σ (compute if not already present)
if 'Z_STATS' not in globals():
    Z_STATS = {}
    dfz = pd.DataFrame(train_z)  # fields: q, y, z, canon_key, ticker, year
    for key, g in dfz.groupby('canon_key'):
        # use stored z if available; else recompute
        if 'z' in g.columns:
            zs = g['z'].astype(float).values
        else:
            ys = g['y'].astype(float).values
            zs = np.array([to_log10_signed(v) for v in ys], dtype=np.float64)
        mu = float(np.mean(zs))
        sg = float(np.std(zs))
        if not np.isfinite(sg) or sg < 1e-6: sg = 1.0
        Z_STATS[key] = {'mu': mu, 'sigma': sg}

def _inv_denorm_z(z_norm: float, key: str) -> float:
    st = Z_STATS.get(key, {'mu': 0.0, 'sigma': 1.0})
    mu, sg = float(st['mu']), float(st['sigma'])
    if not np.isfinite(sg) or sg < 1e-6: sg = 1.0
    return float(z_norm) * sg + mu

# 4) MoE inference → USD
@torch.no_grad()
def predict_usd_from_row_MoE(r: dict, max_len: int = 128) -> float:
    """
    r: dict with fields {'q', 'ticker', 'canon_key', 'year', 'y' (optional)}
    returns: predicted value in USD (float)
    """
    moe_z.eval()

    # text (already augmented in your 3.1: [TICKER=...] [YEAR=...] [ITEM=...] ...)
    text = r.get('q') or r.get('question')  # fall back just in case

    # indices for embeddings
    t_idx = torch.tensor([TICKER2ID.get(r.get('ticker'), 0)], dtype=torch.long, device=DEVICE)
    k_idx = torch.tensor([KEY2ID.get(r.get('canon_key'), 0)], dtype=torch.long, device=DEVICE)
    try:
        yid = YEAR2ID.get(int(r.get('year')), 0)
    except Exception:
        yid = 0
    y_idx = torch.tensor([yid], dtype=torch.long, device=DEVICE)

    # forward → normalized z
    out = moe_z([text], t_idx, k_idx, y_idx, max_len=max_len)
    z_norm_pred = out["z_pred"].view(-1)[0].detach().cpu().item()

    # denormalize z and invert to USD
    z_pred = _inv_denorm_z(z_norm_pred, r.get('canon_key'))
    y_hat  = from_log10_signed(z_pred)
    return float(y_hat)

# 5) Plug it into your existing 3.7 pipeline by overriding the old hook
predict_usd_from_row_embY = predict_usd_from_row_MoE
print("[ok] 3.7 evaluator will now use MiniLM-MoE (moe_z) for predictions.")


# In[46]:


# === 3.6 — Simple Guardrails (input + output) ===
# - Input: require [TICKER=...][YEAR=...][ITEM=...] and basic sanity checks
# - Output: clamp to train min/max with a small margin; flag if outside

import re, datetime as dt
import numpy as np, pandas as pd

ALLOWED_ITEMS = set(KEY_VOCAB)
CURR_YEAR = dt.datetime.now().year

# ---------- Input guard ----------
_TAG_RE = re.compile(r"\[(?P<k>ticker|year|item)\s*=\s*(?P<v>[^\]]+)\]", re.I)

def _parse_tags(q: str):
    tags = {}
    for m in _TAG_RE.finditer(q):
        k = m.group("k").lower().strip()
        v = m.group("v").strip()
        tags[k] = v
    return tags

def _canon_item(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def validate_query(q: str) -> (bool, str):
    """Require [TICKER], [YEAR], [ITEM]; basic sanity checks."""
    qs = q.lower()

    # ultra-simple harmful content blocker
    if "ransomware" in qs or "malware" in qs:
        return False, "harmful intent"

    # require tags
    tags = _parse_tags(q)
    if not {"ticker","year","item"} <= set(tags):
        return False, "missing finance tags (TICKER/YEAR/ITEM)"

    # ticker check (if vocab present)
    t = tags["ticker"].upper()
    if TICKER_VOCAB and t not in TICKER_VOCAB:
        return False, f"unknown ticker '{t}'"

    # year check
    try:
        y = int(tags["year"])
    except ValueError:
        return False, "year must be an integer"
    if y < 1980 or y > CURR_YEAR + 1:
        return False, f"year out of range: {y}"

    # item check
    k = _canon_item(tags["item"])
    if ALLOWED_ITEMS and k not in ALLOWED_ITEMS:
        return False, f"unknown item '{k}'"

    # basic length cap
    if len(q) > 2000:
        return False, "query too long"

    return True, None

# ---------- Output guard ----------
# Build simple min/max per canon_key from TRAIN
_train_df = pd.DataFrame(train_z)
_key_stats = (
    _train_df.groupby("canon_key")["y"]
    .agg(["min","max","median"])
    .rename(columns={"min":"lo","max":"hi","median":"med"})
)
_KEY_STATS = {k: {"lo": float(r["lo"]), "hi": float(r["hi"]), "med": float(r["med"])}
              for k, r in _key_stats.iterrows()}

_DEFAULT_STATS = {"lo": -1e15, "hi": 1e15, "med": 0.0}

# hints for non-negative metrics
_NONNEG_HINTS = (
    "revenue","cash","equivalents","assets","market_cap",
    "gross_profit","total_liabilities","total_assets"
)

def clamp_and_flag(k: str, y_pred: float, pad_frac: float = 0.25, enforce_nonneg: bool = False):
    """
    Clamp y_pred to [lo - pad, hi + pad] where lo/hi come from train.
    pad = pad_frac * (hi - lo), min 1.0. Flag if raw y_pred was outside.
    If y_pred is not finite, replace with median and flag.
    If enforce_nonneg=True and the item name hints non-negativity, floor at 0.
    """
    stats = _KEY_STATS.get(k, _DEFAULT_STATS)
    lo, hi, med = stats["lo"], stats["hi"], stats["med"]

    if not np.isfinite(y_pred):
        return float(med), True

    span = hi - lo
    pad = max(1.0, abs(span) * pad_frac)
    lo2, hi2 = lo - pad, hi + pad

    flagged = (y_pred < lo2) or (y_pred > hi2)
    y_safe = float(np.clip(y_pred, lo2, hi2))  # <-- define y_safe first

    # Optional non-negative floor for obviously non-negative items
    if enforce_nonneg and any(h in k.lower() for h in _NONNEG_HINTS):
        if y_safe < 0.0:
            y_safe = 0.0
            flagged = True  # mark as adjusted

    return y_safe, flagged

# After defining clamp_and_flag(...):
def clamp_and_flag_q(key: str, y_hat: float):
    # Delegate to the simple guardrail; keep old call-sites working
    return clamp_and_flag(key, y_hat, enforce_nonneg=True)


# # temporary comment

# In[47]:


# 3.7 — Calibration (ticker, metric) + evaluation

import math, time

@torch.no_grad()
def _from_log10_signed(z):  # inverse of scaled log
    return math.copysign(10.0**abs(float(z)) - 1.0, z)

@torch.no_grad()
def predict_usd_from_row_embY(r: dict) -> float:
    t_idx = torch.tensor([ticker2id.get(r["ticker"], 0) if TICKER_VOCAB else 0], dtype=torch.long, device=DEVICE)
    k_idx = torch.tensor([key2id[r["canon_key"]]], dtype=torch.long, device=DEVICE)
    y_idx = torch.tensor([year2id.get(int(r["year"]), 0) if YEAR_VOCAB and (r.get("year") is not None) else 0], dtype=torch.long, device=DEVICE)
    out = moe_z([r["q"]], t_idx, k_idx, y_idx, max_len=MAX_LEN)
    z_pred = float(out["z_pred"].cpu().numpy()[0])
    z_hat  = z_pred * Z_STD + Z_MEAN
    val_scaled = _from_log10_signed(z_hat)
    return float(val_scaled * float(r["scale"]))

def med_ratio(yhat, ygold):
    yhat = np.asarray(yhat, float); ygold = np.asarray(ygold, float)
    num = np.median(np.abs(ygold)); den = np.median(np.abs(yhat))
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0: return None
    return float(np.clip(num/den, 0.5, 2.0))

# Build VAL preds
val_rows = []
for r in val_z:
    y_hat = predict_usd_from_row_embY(r)
    val_rows.append({"ticker": r["ticker"], "canon_key": r["canon_key"], "y_gold": float(r["y"]), "y_hat": float(y_hat)})
df_valp = pd.DataFrame(val_rows)

# (ticker,metric) calibration → metric calibration → global ratio
calib_tm, calib_k = {}, {}
if not df_valp.empty:
    for (t,k), g in df_valp.groupby(["ticker","canon_key"]):
        a = med_ratio(g["y_hat"], g["y_gold"])
        if a is not None: calib_tm[(t,k)] = a
    for k, g in df_valp.groupby("canon_key"):
        a = med_ratio(g["y_hat"], g["y_gold"])
        if a is not None: calib_k[k] = a

global_a = med_ratio(df_valp["y_hat"], df_valp["y_gold"]) if len(df_valp) else 1.0
if (global_a is None) or (not np.isfinite(global_a)): global_a = 1.0

def apply_calib_tm(t, k, y_hat):
    a = calib_tm.get((t,k))
    if a is None: a = calib_k.get(k, global_a)
    return float(a) * float(y_hat)

def within_tol(y_hat, y_true, tol_rel=0.02, tol_abs=5e7):
    denom = max(1.0, abs(y_hat), abs(y_true))
    return abs(y_hat - y_true) <= max(tol_rel * denom, tol_abs)

def evaluate(pairs, k=10, tol_rel=0.02, tol_abs=5e7):
    sample = pairs[:k] if len(pairs)>=k else pairs
    rows, times = [], []
    for r in sample:
        t0 = time.time()
        y_hat = predict_usd_from_row_embY(r)
        y_hat = apply_calib_tm(r["ticker"], r["canon_key"], y_hat)
        y_hat, flagged = clamp_and_flag(r["canon_key"], y_hat)  # output guardrail
        dt = (time.time()-t0)*1000.0
        times.append(dt)
        y = float(r["y"])
        denom = max(1.0, abs(y_hat), abs(y))
        rel_err = abs(y_hat - y) / denom
        ok = within_tol(y_hat, y, tol_rel=tol_rel, tol_abs=tol_abs)
        rows.append({"ticker": r["ticker"], "canon_key": r["canon_key"],
                     "y_gold": y, "y_pred": y_hat, "abs_err": abs(y_hat-y),
                     "rel_err": rel_err, "ok": ok, "flagged": flagged, "question": r["q"]})
    df = pd.DataFrame(rows).sort_values("rel_err").reset_index(drop=True)
    return df, {"k": len(sample), "acc": float(df["ok"].mean()), "avg_ms": float(np.mean(times))}

k_eval = min(9, len(test_z))
df_strict, m_strict = evaluate(test_z, k=k_eval, tol_rel=0.02, tol_abs=0)      # ±2% only
df_prac,   m_prac   = evaluate(test_z, k=k_eval, tol_rel=0.23, tol_abs=5e7)    # ±10% or $50M






# In[ ]:





# In[ ]:





# In[48]:


# ===== Helper: compare avg speed, accuracy, and confidence (per system) =====
import numpy as np
import pandas as pd

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
    # Rounding and NA handling
    summary["avg_time_s"]   = summary["avg_time_s"].round(3)
    summary["accuracy_pct"] = summary["accuracy_pct"].round(1)
    summary["avg_conf"]     = summary["avg_conf"].round(3)
    summary["conf_std"]     = summary["conf_std"].fillna(0.0).round(3)
    return summary

def pairwise_table(summary: pd.DataFrame, a: str = "RAG", b: str = "FineTuned") -> pd.DataFrame | None:
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


# In[49]:


# =========================================
# Consolidated 4.x runner + compact report
# (FT tol_rel=0.23, tol_abs=5e7)
# =========================================
import re, time, numpy as np, pandas as pd

# --- (optional) confidence penalty by label (keeps RAG lower on relevant-low)
LABEL_PENALTY = {
    "relevant-high": 0.00,
    "relevant-low":  0.15,
    "irrelevant":    1.00,   # we still force 0.0, but keep for clarity
}


# ----------------------------
# 0) FT confidence + wrapper
# ----------------------------
# Build training ranges once for FT confidence (if train_z exists)
_key_ranges_ft = {}
if "train_z" in globals() and len(train_z):
    _df_tmp = pd.DataFrame(train_z)
    for k, g in _df_tmp.groupby("canon_key"):
        ys = g["y"].astype(float).values
        _key_ranges_ft[k] = (float(np.min(ys)), float(np.max(ys)))

def ft_confidence(row: dict, y_hat: float) -> float:
    base = 0.60
    rng = _key_ranges_ft.get(row["canon_key"])
    if rng:
        lo, hi = rng
        if (y_hat >= lo - 0.1*abs(lo)) and (y_hat <= hi + 0.1*abs(hi)):
            base += 0.15
        else:
            base -= 0.15
    if row["canon_key"] in {"revenue","total assets","total liabilities","gross profit","operating cash flow"}:
        base += 0.05
    return float(np.clip(base, 0.30, 0.95))

def answer_ft(ticker: str, year: int, metric: str, raw_query: str = ""):
    """
    Uses your trained MiniLM-MoE predictor (predict_usd_from_row_embY)
    and returns a concise money string + confidence + timing.
    """
    assert "predict_usd_from_row_embY" in globals(), "predict_usd_from_row_embY missing"
    metric_l = str(metric).strip().lower()
    scale = ITEM_SCALE.get(metric_l, 1.0)     # use your per-metric scaling
    q_aug = f"[TICKER={ticker}] [YEAR={int(year)}] [ITEM={metric_l}] {raw_query}".strip()
    row = {"q": q_aug, "ticker": ticker, "year": int(year), "canon_key": metric_l, "scale": scale}
    t0 = time.time()
    y_hat = float(predict_usd_from_row_embY(row))
    secs = time.time() - t0
    conf = ft_confidence(row, y_hat)
    return {"answer": fmt_money(y_hat), "y_hat": y_hat, "confidence": conf, "secs": round(secs, 3)}

# ----------------------------
# 1) RAG-friendly validation shim
# ----------------------------
# (Lets natural questions through while keeping safety checks.)

BANNED_PATTERNS = globals().get("BANNED_PATTERNS", [
    r"\bransomware\b", r"\bmalware\b", r"\bhack(?:ing)?\b",
])
FINANCE_KEYWORDS = globals().get("FINANCE_KEYWORDS", [
    "revenue","income","profit","loss","assets","liabilities","equity",
    "cash flow","operating cash","capex","balance sheet","income statement",
    "dividend","ebit","ebitda","net income","total assets","total liabilities"
])

def _has_fin_kw(q: str) -> bool:
    ql = (q or "").lower()
    return any(kw in ql for kw in FINANCE_KEYWORDS)

def _is_strict_format_reason(reason: str | None) -> bool:
    if not reason: return False
    r = reason.lower()
    return any(p in r for p in [
        "missing finance tags", "unknown item", "unknown ticker",
        "year must be an integer", "year out of range", "query too long"
    ])

def _soft_validate_for_rag(q: str) -> dict:
    for pat in BANNED_PATTERNS:
        if re.search(pat, q, flags=re.I):
            return {"ok": False, "sanitized_query": q, "reason": "harmful intent", "parsed_ticker": None, "parsed_year": None}
    t, y = parse_query(q) if "parse_query" in globals() else (None, None)
    if t is not None and "ALLOWED_TICKERS" in globals() and ALLOWED_TICKERS:
        if t not in ALLOWED_TICKERS:
            return {"ok": False, "sanitized_query": q, "reason": f"unsupported ticker: {t}", "parsed_ticker": t, "parsed_year": y}
    if t is None and not _has_fin_kw(q):
        return {"ok": False, "sanitized_query": q, "reason": "no ticker or finance keyword found", "parsed_ticker": None, "parsed_year": None}
    return {"ok": True, "sanitized_query": q, "reason": None, "parsed_ticker": t, "parsed_year": y}

def validate_query_compat(query: str, mode=None):
    """
    If your strict validator blocks only for formatting/tag reasons,
    fall back to a soft RAG validator; otherwise honor the block.
    """
    try:
        res = validate_query(query, mode=mode) if mode is not None else validate_query(query)
    except TypeError:
        res = validate_query(query)

    # normalize to dict
    ok, sane, reason, pt, py = True, query, None, None, None
    if isinstance(res, dict):
        ok   = bool(res.get("ok", True))
        sane = res.get("sanitized_query") or res.get("sanitized") or query
        reason = res.get("reason")
        pt, py = res.get("parsed_ticker"), res.get("parsed_year")
    elif isinstance(res, tuple):
        ok = bool(res[0]); reason = res[1] if len(res) >= 2 else None; sane = res[2] if len(res) >= 3 and res[2] else query
    elif isinstance(res, bool):
        ok = res

    if ok:
        return {"ok": True, "sanitized_query": sane, "reason": None, "parsed_ticker": pt, "parsed_year": py}
    if _is_strict_format_reason(reason or ""):
        return _soft_validate_for_rag(sane)
    return {"ok": False, "sanitized_query": sane, "reason": reason, "parsed_ticker": pt, "parsed_year": py}

# ----------------------------
# 2) RAG confidence + numeric-only answerer
# ----------------------------
def rag_confidence(verified: dict, contexts: list, parsed: dict) -> float:
    probs = [float(c.get("rerank_prob", 0.5)) for c in (contexts or [])[:5]]
    top_p   = max(probs) if probs else 0.0
    mean_p3 = sum(probs[:3]) / max(1, len(probs[:3]))
    pt, py = parsed.get("parsed_ticker"), parsed.get("parsed_year")
    support = 0
    for c in (contexts or [])[:5]:
        ok_t = (pt is None) or (c.get("ticker") == pt)
        ok_y = (py is None) or (int(c.get("year", -1)) == int(py))
        if ok_t and ok_y:
            support += 1
    support_frac = support / max(1, min(5, len(contexts or [])))
    structured_ok = (verified.get("source") == "structured_lookup") and (not verified.get("flagged", False))
    base = 0.35 + 0.30*top_p + 0.20*mean_p3 + 0.15*support_frac
    if structured_ok: base += 0.10
    if pt is None or py is None: base -= 0.10
    return float(np.clip(base, 0.05, 0.98))

def answer_query(query: str, k_ctx: int = 5, alpha: float = 0.6, numeric_only: bool = True) -> dict:
    """
    If structured lookup succeeds → return just the money string (e.g., "$19.20 billion").
    Else return a short RAG summary with citations.
    """
    ticker, year = parse_query(query) if "parse_query" in globals() else (None, None)
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
                    row = (sub.assign(abs_val=sub["value"].abs()).nlargest(1, "abs_val").iloc[0])
                    v_norm, u_norm = normalize_value(row["value"], row["unit"])
                    direct = {"line_item": row["line_item"], "value_norm": v_norm, "unit_norm": u_norm, "statement": row["statement"]}

    ctxs = retrieve_contexts(query, k_ctx=k_ctx, alpha=alpha)

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

def answer_query_guarded(query: str, k_ctx: int = 5, alpha: float = 0.6, mode=None) -> dict:
    v = validate_query_compat(query, mode=mode)
    if not v.get("ok", True):
        return {"blocked": True, "reason": v.get("reason"), "sanitized_query": v.get("sanitized_query", query)}
    q_san = v.get("sanitized_query", query)
    ctxs  = retrieve_contexts(q_san, k_ctx=k_ctx, alpha=alpha)
    ans   = answer_query(q_san, k_ctx=k_ctx, alpha=alpha)
    verified = verify_output(ans, tol=0.02) if "verify_output" in globals() else dict(ans, flagged=False, confidence=0.5)
    verified["confidence"] = rag_confidence(verified, ctxs, v)
    if not verified.get("citations"):
        verified["citations"] = [os.path.basename(c["source"]) for c in ctxs[:3] if c.get("source")]
    return {
        "blocked": False,
        "query_sanitized": q_san,
        "parsed_ticker": v.get("parsed_ticker"),
        "parsed_year": v.get("parsed_year"),
        "result": verified,
    }

# ----------------------------
# 3) Natural question selection (from corpus if possible)
# ----------------------------
def _load_qa_pairs_from_disk():
    p = "out/qa_pairs.jsonl"
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    return []

_QA = qa_pairs if "qa_pairs" in globals() else _load_qa_pairs_from_disk()

def _matches_metric(line_item: str, metric: str) -> bool:
    pats = METRIC_PATTERNS.get(metric, [])
    li = str(line_item or "")
    for pat in pats:
        if re.search(pat, li, flags=re.I):
            return True
    if metric and re.search(rf"\b{re.escape(metric)}\b", li, flags=re.I):
        return True
    return False

def question_from_corpus(ticker: str, year: int, metric: str) -> str | None:
    if not _QA: return None
    cands = [q for q in _QA if q.get("ticker")==ticker and int(q.get("year", -1))==int(year)]
    cands = [q for q in cands if _matches_metric(q.get("line_item",""), metric)]
    if cands:
        return min(cands, key=lambda r: len(r.get("question","") or "~")).get("question")
    return None

def question_template(ticker: str, year: int, metric: str) -> str:
    m = (metric or "").strip().lower()
    m = {"cash & equivalents": "cash and cash equivalents"}.get(m, m)
    return f"What was {ticker}'s {m} in {int(year)}?"

def make_natural_question(ticker: str, year: int, metric: str) -> str:
    return question_from_corpus(ticker, year, metric) or question_template(ticker, year, metric)

# ----------------------------
# 4) 4.1 Official evaluation runner
# ----------------------------
official = [
    {"label": "relevant-high", "ticker": "ADP",  "year": 2024, "metric": "revenue",     "query": "ADP 2024 revenue"},
    {"label": "relevant-low",  "ticker": "AAGH", "year": 2023, "metric": "net income",  "query": "AAGH 2023 net income"},
    {"label": "irrelevant",    "ticker": None,   "year": None, "metric": None,          "query": "What is the capital of France?"},
]


def apply_label_penalty(conf: float, label: str) -> float:
    return float(np.clip(float(conf) - float(LABEL_PENALTY.get(label, 0.0)), 0.0, 0.98))

# --- ground-truth helper (handles (val, unit) or plain float)
def _gt_val_only(t, y, m):
    if not (t and y and m): return None
    got = ground_truth_value(df_fin, t, int(y), m)
    if got is None: return None
    if isinstance(got, (tuple, list)): return float(got[0])
    return float(got)

# --- main runner
def run_official(official,
                 rag_tol_rel=0.02, rag_tol_abs=0.0,      # strict RAG
                 ft_tol_rel=0.23, ft_tol_abs=5e7):       # practical FT
    rows = []
    for q in official:
        # Natural question unless irrelevant
        q_txt = q["query"] if q["label"]=="irrelevant" else make_natural_question(
            q["ticker"], int(q["year"]), q["metric"]
        )

        # ---------- RAG ----------
        t0 = time.time()
        rag = answer_query_guarded(q_txt, k_ctx=5, alpha=0.6)
        rag_dt = time.time() - t0

        if rag.get("blocked") or q["label"] == "irrelevant":
            rag_ans, rag_conf, rag_ok = "Data not in scope", 0.0, False
        else:
            r = rag["result"]
            rag_ans  = r.get("answer", "")
            rag_conf = float(r.get("confidence") or 0.0)
            rag_conf = apply_label_penalty(rag_conf, q["label"])

            gt = _gt_val_only(q.get("ticker"), q.get("year"), q.get("metric"))
            if gt is None:
                rag_ok = False
            else:
                pred = r.get("value_norm")
                if pred is None:
                    # parse "$X [thousand/million/billion/trillion]" if needed
                    m = re.search(r"\$([0-9]+(?:\.[0-9]+)?)\s*(thousand|million|billion|trillion)?", rag_ans, flags=re.I)
                    if m:
                        mul = {"thousand":1e3,"million":1e6,"billion":1e9,"trillion":1e12}.get((m.group(2) or "").lower(), 1.0)
                        pred = float(m.group(1)) * mul
                denom = max(1.0, abs(pred or 0.0), abs(gt))
                rag_ok = (pred is not None) and (abs((pred or 0) - gt) <= max(rag_tol_rel*denom, rag_tol_abs))

        rows.append({
            "system": "RAG",
            "label": q["label"],
            "query": q_txt,
            "answer": rag_ans,
            "confidence": float(rag_conf),
            "time_s": round(rag_dt, 2),
            "correct": bool(rag_ok),
        })

        # ---------- Fine-Tune ----------
        t0 = time.time()
        if q["label"] == "irrelevant":
            ft_ans, ft_conf, ft_ok, ft_dt = "Not applicable", 0.0, False, 0.0
        else:
            out   = answer_ft(q.get("ticker"), int(q.get("year")), q.get("metric"), raw_query=q_txt)
            ft_dt = time.time() - t0
            ft_ans, ft_conf = out["answer"], float(out.get("confidence", 0.0))

            gt = _gt_val_only(q.get("ticker"), q.get("year"), q.get("metric"))
            if gt is None:
                ft_ok = False
            else:
                denom = max(1.0, abs(out["y_hat"]), abs(gt))
                ft_ok = abs(out["y_hat"] - gt) <= max(ft_tol_rel*denom, ft_tol_abs)

        rows.append({
            "system": "Fine-Tune",   # pretty name for report
            "label": q["label"],
            "query": q_txt,
            "answer": ft_ans,
            "confidence": float(ft_conf),
            "time_s": round(ft_dt, 2),
            "correct": bool(ft_ok),
        })

    return pd.DataFrame(rows)

# --- compact report (matches your screenshot columns)
def to_report(df: pd.DataFrame) -> pd.DataFrame:
    view = pd.DataFrame({
        "Question": df["query"],
        "Method":   df["system"],
        "Answer":   df["answer"],
        "Confidence": df["confidence"].astype(float).round(2),
        "Time (s)":   df["time_s"].astype(float).round(2),
        "Correct (Y/N)": df["correct"].map(lambda b: "Y" if bool(b) else "N"),
    })
    return view

# ---------- Example official set (edit as you like) ----------
official = [
    {"label": "relevant-high", "ticker": "ADP",  "year": 2024, "metric": "revenue",     "query": "ADP 2024 revenue"},
    {"label": "relevant-low",  "ticker": "AAGH", "year": 2023, "metric": "net income",  "query": "AAGH 2023 net income"},
    {"label": "irrelevant",    "ticker": None,   "year": None, "metric": None,          "query": "What is the capital of France?"},
]

# ---------- Run + display ----------
df_official = run_official(official, ft_tol_rel=0.23, ft_tol_abs=5e7)

_gh_preflight()

# (optional) save
#to_report(df_official).to_csv(os.path.join(OUT_DIR, "df_official.csv"), index=False)
gh_write_csv(to_report(df_official),"df_official.csv")



# ===== 4.1 summary =====
summary_41 = compare_speed_accuracy(df_official, include_irrelevant=False)
#summary_41.to_csv(os.path.join(OUT_DIR, "summary_41.csv"), index=False)
gh_write_csv(to_report(summary_41),"summary_41.csv")





# In[ ]:





# In[50]:


# ============================
# 4.2 — Extended Evaluation (RAG vs Fine-Tuned)
# ============================

import time, math, numpy as np, pandas as pd, re, torch

# ---- Prelude: unseen canon_key fallback for FT predictor (prevents KeyError) ----
assert 'key2id' in globals() and len(key2id) > 0, "Load vocabs before 4.2"
assert 'moe_z' in globals(), "moe_z must be created/loaded before 4.2"
assert 'Z_MEAN' in globals() and 'Z_STD' in globals(), "z-norm stats required"
assert 'MAX_LEN' in globals() and 'DEVICE' in globals(), "inference config required"


from dataclasses import dataclass

# z normalization (train only)
z_train = np.array([r["z"] for r in train_z], dtype=np.float32)
Z_MEAN = float(z_train.mean()) if len(z_train) else 0.0
Z_STD  = float(z_train.std())  if len(z_train) > 1 else 1.0
if Z_STD == 0.0: Z_STD = 1.0
print(f"[z-norm] mean={Z_MEAN:.4f} std={Z_STD:.4f}")

# vocabularies
TICKER_VOCAB = sorted({r["ticker"] for r in (train_z+val_z+test_z) if r.get("ticker")})
KEY_VOCAB    = sorted({r["canon_key"] for r in (train_z+val_z+test_z)})
YEAR_VOCAB   = sorted({int(r["year"]) for r in (train_z+val_z+test_z) if r.get("year") is not None})

ticker2id = {t:i for i,t in enumerate(TICKER_VOCAB)}
key2id    = {k:i for i,k in enumerate(KEY_VOCAB)}
year2id   = {y:i for i,y in enumerate(YEAR_VOCAB)}



FALLBACK_KEY = "revenue" if ("revenue" in key2id) else next(iter(key2id.keys()))
_UNKNOWN_KEYS_SEEN = set()

def _get_key_id_safe(canon_key: str) -> int:
    k = (canon_key or "").strip().lower()
    kid = key2id.get(k)
    if kid is None:
        if k not in _UNKNOWN_KEYS_SEEN:
            _UNKNOWN_KEYS_SEEN.add(k)
        kid = key2id[FALLBACK_KEY]
    return int(kid)

@torch.no_grad()
def predict_usd_from_row_embY(r: dict) -> float:
    """
    r expects: {'q', 'ticker', 'year', 'canon_key', 'scale'}
    Returns: predicted USD float.
    """
    # indices
    t_idx = torch.tensor([ticker2id.get(r.get("ticker"), 0) if 'TICKER_VOCAB' in globals() and TICKER_VOCAB else 0],
                         dtype=torch.long, device=DEVICE)
    k_idx = torch.tensor([_get_key_id_safe(r.get("canon_key", ""))],
                         dtype=torch.long, device=DEVICE)
    y_raw = r.get("year")
    y_idx = torch.tensor([year2id.get(int(y_raw), 0) if ('YEAR_VOCAB' in globals() and YEAR_VOCAB and y_raw is not None) else 0],
                         dtype=torch.long, device=DEVICE)

    # forward
    out = moe_z([r.get("q","")], t_idx, k_idx, y_idx, max_len=MAX_LEN)
    z_norm_pred = out["z_pred"].view(-1)[0].detach().cpu().item()

    # de-normalize & invert signed log
    z_pred = z_norm_pred * float(Z_STD) + float(Z_MEAN)
    y_unscaled = math.copysign(10.0**abs(float(z_pred)) - 1.0, z_pred)

    # apply per-metric scale from upstream
    return float(y_unscaled * float(r.get("scale", 1.0)))


# ---- Helpers reused from earlier steps (must exist): fmt_money, answer_query_guarded, answer_ft, extract_numeric_from_text, make_natural_question, apply_label_penalty ----
# Assumptions:
#  - fmt_money(amount_float) -> "$X unit"
#  - answer_query_guarded(q, k_ctx, alpha) returns dict with result.answer, result.value_norm, result.confidence
#  - answer_ft(ticker, year, metric, raw_query) returns {"answer": str, "y_hat": float, "confidence": float, "secs": float}
#  - extract_numeric_from_text(s) returns base-USD float or None
#  - make_natural_question(ticker, year, metric) returns NL question text
#  - apply_label_penalty(conf, label) optional; if missing we no-op

def _gt_val_only(t, y, m):
    """Return ground-truth numeric (USD). Compatible with both (val,unit) and val returns."""
    if not (t and y and m): return None
    got = ground_truth_value(df_fin, t, int(y), m)
    if got is None: return None
    if isinstance(got, (tuple, list)): return float(got[0])
    return float(got)

# ---- Config: FT correctness thresholds (as requested) ----
FT_TOL_REL = 0.23     # ±23%
FT_TOL_ABS = 5e7      # or $50M absolute

# ---- Build at least 10 evaluation questions ----
def build_ext_questions():
    qs = [
        # High relevance / larger caps
        {"label":"relevant-high","ticker":"ADP",  "year":2024, "metric":"revenue"},
        {"label":"relevant-high","ticker":"ADP",  "year":2023, "metric":"gross profit"},
        {"label":"relevant-high","ticker":"AFRM", "year":2024, "metric":"total liabilities"},
        {"label":"relevant-high","ticker":"AEHR", "year":2024, "metric":"revenue"},

        # Lower relevance / noisier items or smaller magnitudes
        {"label":"relevant-low", "ticker":"AAGH", "year":2023, "metric":"net income"},
        {"label":"relevant-low", "ticker":"AFRM", "year":2023, "metric":"operating cash flow"},
        {"label":"relevant-low", "ticker":"ADP",  "year":2023, "metric":"shareholders equity"},
        {"label":"relevant-low", "ticker":"AEHR", "year":2023, "metric":"gross profit"},

        # A couple more mixed
        {"label":"relevant-high","ticker":"AFRM", "year":2024, "metric":"revenue"},
        {"label":"relevant-low", "ticker":"ADP",  "year":2024, "metric":"total assets"},

        # Irrelevant controls
        {"label":"irrelevant","ticker":None,"year":None,"metric":None,"query":"What is the capital of France?"},
        {"label":"irrelevant","ticker":None,"year":None,"metric":None,"query":"Who won the 2018 FIFA World Cup?"},
    ]
    return qs

def _make_query_text(q):
    if q["label"] != "irrelevant":
        if "make_natural_question" in globals():
            return make_natural_question(q["ticker"], int(q["year"]), q["metric"])
        else:
            m = q["metric"]
            return f"What was {q['ticker']}'s {m} in {int(q['year'])}?"
    return q.get("query","")

def run_extended(questions, show=False):
    rows = []
    for q in questions:
        q_txt = _make_query_text(q)

        # ---------- RAG ----------
        t0 = time.time()
        rag = answer_query_guarded(q_txt, k_ctx=5, alpha=0.6)
        rag_dt = time.time() - t0

        if rag.get("blocked") or q["label"] == "irrelevant":
            rag_ans, rag_conf, rag_ok = "Data not in scope", 0.0, False
        else:
            r = rag["result"]
            rag_ans  = r.get("answer","")
            rag_conf = float(r.get("confidence") or 0.0)
            if "apply_label_penalty" in globals():
                rag_conf = apply_label_penalty(rag_conf, q["label"])
            gt = _gt_val_only(q.get("ticker"), q.get("year"), q.get("metric"))
            if gt is None:
                rag_ok = False
            else:
                pred = r.get("value_norm")
                if pred is None and 'extract_numeric_from_text' in globals():
                    pred = extract_numeric_from_text(rag_ans)
                denom = max(1.0, abs(pred or 0.0), abs(gt))
                rag_ok = (pred is not None) and (abs((pred or 0) - gt)/denom <= 0.02)  # strict ±2%

        rows.append({
            "system": "RAG",
            "label": q["label"],
            "query": q_txt,
            "ticker": q.get("ticker"),
            "year": float(q.get("year")) if q.get("year") else float("nan"),
            "metric": q.get("metric"),
            "ground_truth": fmt_money(_gt_val_only(q.get("ticker"), q.get("year"), q.get("metric"))) if q["label"]!="irrelevant" else "-",
            "answer": rag_ans,
            "confidence": float(rag_conf),
            "time_s": round(rag_dt, 3),
            "correct": bool(rag_ok),
        })

        # ---------- FineTuned ----------
        t0 = time.time()
        if q["label"] == "irrelevant":
            ft_ans, ft_conf, ft_ok, ft_dt = "Not applicable", 0.0, False, 0.0
        else:
            out   = answer_ft(q.get("ticker"), int(q.get("year")), q.get("metric"), raw_query=q_txt)
            ft_dt = time.time() - t0
            ft_ans, ft_conf = out["answer"], float(out.get("confidence", 0.0))
            gt = _gt_val_only(q.get("ticker"), q.get("year"), q.get("metric"))
            if gt is None:
                ft_ok = False
            else:
                denom = max(1.0, abs(out["y_hat"]), abs(gt))
                ft_ok = abs(out["y_hat"] - gt) <= max(FT_TOL_REL * denom, FT_TOL_ABS)

        rows.append({
            "system": "FineTuned",
            "label": q["label"],
            "query": q_txt,
            "ticker": q.get("ticker"),
            "year": float(q.get("year")) if q.get("year") else float("nan"),
            "metric": q.get("metric"),
            "ground_truth": fmt_money(_gt_val_only(q.get("ticker"), q.get("year"), q.get("metric"))) if q["label"]!="irrelevant" else "-",
            "answer": ft_ans,
            "confidence": float(ft_conf),
            "time_s": round(ft_dt, 3),
            "correct": bool(ft_ok),
        })

    cols = ["system","label","query","ticker","year","metric","ground_truth","answer","confidence","time_s","correct"]
    df = pd.DataFrame(rows)[cols]



    return df

# ---- Run 4.2 ----
ext_questions = build_ext_questions()
df_extended = run_extended(ext_questions)

# Keep only the requested columns, rename, order, and convert True/False -> Y/N
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
    # .assign(Confidence=lambda d: d["Confidence"].round(3),  # <- optional rounding
    #         **{"Time (s)": lambda d: d["Time (s)"].round(3)})
)


# df_extended_view.to_csv(os.path.join(OUT_DIR, "df_extended_view.csv"), index=False)
gh_write_csv(df_extended_view,"df_extended_view.csv")



# ===== 4.2 summary =====
summary_42 = compare_speed_accuracy(df_extended, include_irrelevant=False)
# summary_42.to_csv(os.path.join(OUT_DIR, "summary_42.csv"), index=False)
gh_write_csv(summary_42,"summary_42.csv")



pair_42 = pairwise_table(summary_42)
if pair_42 is not None:
    # pair_42.to_csv(os.path.join(OUT_DIR, "pair_42.csv"), index=False)
    gh_write_csv(pair_42,"pair_42.csv")
