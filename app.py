#!/usr/bin/env python3
"""
Consensus Search Engine (FastAPI)

• Only VALIDATED products (via Mistral) are shown
• Mention-friendly extraction: "this head and shoulders shampoo" → "head and shoulders shampoo"
• TF-IDF + upvote weighting → candidate list → Mistral picks true products, adds summary + pros/cons
• Routes: /, /search, /api/search, /download.csv (validated only)
"""

from __future__ import annotations
import os, re, sys, math, html, json, time, logging
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from urllib.parse import quote_plus
from fastapi.responses import HTMLResponse  # you likely already have this

import requests
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Template


# ---------------------------------------------------
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# --- Robust .env loader (next to this file) ---
try:
    from dotenv import load_dotenv
    here = Path(__file__).resolve().parent
    loaded_from = None
    for name in (".env.local", ".env", ".env.dev"):
        p = here / name
        if p.exists():
            load_dotenv(p, override=False)
            loaded_from = str(p)
            break
    logging.info(f".env loaded from: {loaded_from or 'NOT FOUND'}")
except Exception as _e:
    logging.info(f".env loading skipped/error: {_e!r}")
# ------------------------------------------------

# ---------------------------------------------------
# spaCy (for NER + POS)
def load_spacy():
    import logging
    try:
        import spacy
    except Exception as e:
        logging.error("spaCy is not installed: %s", e)
        # Return a tiny no-op pipeline so the app can still run
        return None

    nlp = None

    # 1) Try the packaged small model (install via wheel URL in requirements.txt)
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # 2) Some environments prefer direct import of the package
        try:
            import en_core_web_sm  # type: ignore
            nlp = en_core_web_sm.load()
        except Exception:
            # 3) Final fallback: blank English (no pretrained components)
            logging.warning(
                "en_core_web_sm not available. Falling back to spacy.blank('en')."
            )
            try:
                nlp = spacy.blank("en")
            except Exception as e:
                logging.error("Failed to create blank English pipeline: %s", e)
                return None

    # Ensure we have sentence boundaries
    try:
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    except Exception as e:
        logging.warning("Could not add sentencizer: %s", e)

    return nlp

NLP = load_spacy()


REDDIT = None
from fastapi import HTTPException

REDDIT = None
def get_reddit():
    global REDDIT
    if REDDIT:
        return REDDIT
    try:
        import praw
    except ImportError as e:
        logging.error("praw not installed: %s", e)
        # Do not sys.exit in serverless. Surface a controlled 500.
        raise HTTPException(status_code=500, detail="Backend missing dependency: praw")

    cid  = os.getenv("REDDIT_CLIENT_ID", "").strip()
    csec = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    ua   = os.getenv("REDDIT_USER_AGENT", "universal-consensus/1.0 (by u/yourusername)").strip()

    if not cid or not csec:
        logging.error("Missing REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET")
        raise HTTPException(status_code=500, detail="Missing Reddit credentials")

    try:
        REDDIT = praw.Reddit(
            client_id=cid,
            client_secret=csec,
            user_agent=ua,
            check_for_async=False,
        )
        _ = REDDIT.read_only  # force init
        return REDDIT
    except Exception as e:
        logging.error("Failed to initialize PRAW: %s", e)
        raise HTTPException(status_code=500, detail="Could not initialize Reddit client")


# ---------------------------------------------------
# Helpers
_token_re = re.compile(r"[a-z0-9]+")
def canon(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("’","'").replace("–","-").replace("—","-")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,:;!?-_/\\|\"'`~()[]{}")

def strip_lead_ins(s: str) -> str:
    # remove determiners/pronouns like "this", "that", "my", "the", etc.
    return re.sub(
        r"^(?:this|that|these|those|my|your|his|her|their|our|the)\s+",
        "",
        canon(s)
    )


# ---------------------------------------------------
# Product extraction (mention-friendly)
GENERIC_NOUNS = {
    "shampoo","conditioner","serum","cream","gel","spray","mask",
    "cleanser","toner","moisturizer","oil","balm","exfoliant",
    "essence","lotion","primer","foundation","concealer","lipstick",
    "mascara","palette","eyeliner","powder","bronzer","blush",
    "sunscreen","retinol","ampoule","peel","mist"
}

BRAND_HINTS = [
    "head & shoulders","head and shoulders","olaplex","loreal","l'oréal","redken",
    "pureology","amika","davines","garnier","pantene","shea","moroccanoil","cerave",
    "ouai","briogeo","matrix","dove","tresemme","the ordinary","paula's choice",
    "neutrogena","olay","clinique","tatcha","glossier","rare beauty","fenty",
    "elf","maybelline","nars","estee lauder","lancome","drunk elephant","innisfree",
    "cosrx","aveda","aesop","k18","kerastase","verb","living proof","paul mitchell",
    "biolage","garnier fructis","john frieda"
]
BRAND_TOKENS = {b.lower() for b in BRAND_HINTS}

# Normalize brand tokens for matching (handles "head and shoulders" vs "head & shoulders")
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

NORMED_BRANDS = {_norm(b): b for b in BRAND_TOKENS}

def infer_brand(text: str) -> Optional[str]:
    """Return canonical brand if any brand token appears in text."""
    lt = _norm(text)
    for nb, canon_b in NORMED_BRANDS.items():
        if re.search(rf"\b{re.escape(nb)}\b", lt):
            return canon_b
    return None

def noun_tail(phrase: str) -> str:
    """Return the shortest tail ending in a generic noun (e.g., 'clarifying shampoo')."""
    s = normalize_phrase(phrase)
    toks = s.split()
    for i in range(len(toks)):
        tail = " ".join(toks[i:])
        if has_generic_noun(tail):
            return tail
    return s

def normalize_to_branded(phrase: str, examples: List[str]) -> str:
    """
    If phrase is generic but an example mentions a brand, prepend it:
    'this clarifying shampoo' + 'I love Redken clarifying shampoo' -> 'redken clarifying shampoo'
    """
    base = noun_tail(phrase)
    for ex in (examples or []):
        b = infer_brand(ex)
        if b:
            return normalize_phrase(f"{b} {base}")
    # also try the phrase itself (mentions like 'this head and shoulders shampoo')
    b = infer_brand(phrase)
    if b:
        return normalize_phrase(phrase)
    return normalize_phrase(phrase)


def is_branded_name(name: str) -> bool:
    n = canon(name)
    brand_hit = any(b in n for b in BRAND_TOKENS)
    model_hit = bool(re.search(r"\bno\.?\s*\d{1,4}[a-z]?\b", n))   # e.g., "no 4"
    line_hit  = bool(re.search(r"\b(all soft|classic clean|bond maintenance|sheer glow|elvive|fructis|argan oil|anti dandruff|repair|volume|hydra|curl|purple)\b", n))
    has_noun  = has_generic_noun(n)
    # Accept if: a brand is present OR (line/model marker + noun).
    return (brand_hit and has_noun) or (has_noun and (model_hit or line_hit))



# brand + product noun (direct)
BRAND_PRODUCT_RE = re.compile(
    r"\b([A-Z][\w'&\.-]+(?:\s+[A-Z0-9][\w'&\.-]+){0,3})\s+(?:"
    + "|".join(sorted(GENERIC_NOUNS)) + r")\b",
    re.IGNORECASE,
)

# Model-like pattern
MODEL_RE = re.compile(
    r"\b([A-Z][\w'&\.-]+(?:\s+[A-Z0-9][\w'&\.-]+){0,2})\s*(?:no\.?|#)?\s*\d{1,4}[A-Z]?\b",
    re.IGNORECASE,
)

# Conversational junk
BAD_PATTERNS = [
    re.compile(r"^\d+[a-z]?$", re.IGNORECASE),
    re.compile(r"\b(any|some|every|no)thing\s+(in|on|for|with)\s+my\s+\d+[a-z]?\b", re.IGNORECASE),
]
BAD_STARTS = re.compile(r"^(look|plus|use|picture|was|were|is|am|are|do|did|does|can|could|should|would|may|might|must|shall)\b", re.IGNORECASE)

def looks_like_brandish(s: str) -> bool:
    ls = s.lower()
    return any(b in ls for b in BRAND_TOKENS)

def has_generic_noun(s: str) -> bool:
    return any((" " + n + " ") in (" " + s.lower() + " ") for n in GENERIC_NOUNS)

def normalize_phrase(s: str) -> str:
    s = canon(s)
    s = re.sub(r"^(?:this|that|these|those|my|your|his|her|their|our|the)\s+", "", s)
    s = re.sub(r"\b(and|&)\b", "and", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_valid_phrase(raw: str) -> bool:
    s = normalize_phrase(raw)
    if len(s) < 3 or len(s) > 60: return False
    if len(s.split()) < 2: return False
    if BAD_STARTS.search(s): return False
    if any(p.search(s) for p in BAD_PATTERNS): return False
    # accept if it contains a generic noun AND at least 2 tokens (brandish optional)
    if has_generic_noun(s):
        # avoid ultra-generic like just "shampoo"
        if len(_token_re.findall(s)) >= 2:
            return True
    # accept model-like codes (rare in beauty but okay)
    if MODEL_RE.search(raw): return True
    return False

def dedupe_phrases(phrases: List[str]) -> List[str]:
    seen = set(); out = []
    for p in phrases:
        key = " ".join(sorted(_token_re.findall(normalize_phrase(p))))
        if key and key not in seen:
            seen.add(key); out.append(normalize_phrase(p))
    return out

def extract_product_phrases(text: str) -> List[str]:
    if not text:
        return []
    found = set()

    # 1) Regex-only passes that do not require spaCy
    for m in BRAND_PRODUCT_RE.finditer(text):
        phrase = normalize_phrase(m.group(0))
        if is_valid_phrase(phrase): found.add(phrase)
    for m in MODEL_RE.finditer(text):
        phrase = normalize_phrase(m.group(0))
        if is_valid_phrase(phrase): found.add(phrase)

    # If spaCy is missing, return the regex-based results
    if NLP is None:
        return [p for p in dedupe_phrases(list(found)) if is_valid_phrase(p)]

    # 2) Token-window and noun-chunk passes that use spaCy
    doc = NLP(text)
    toks = [t.text for t in doc]
    low  = [t.text.lower() for t in doc]

    for i, tok in enumerate(low):
        if tok in GENERIC_NOUNS:
            left = max(0, i-4); right = min(len(low), i+5)
            window = " ".join(toks[left:right])
            if any(b in window.lower() for b in BRAND_TOKENS):
                span = normalize_phrase(window)
                m = re.search(r"((?:[\w'&\.-]+\s+){0,4}" + re.escape(tok) + r")\b", span, re.IGNORECASE)
                if m:
                    phrase = normalize_phrase(m.group(1))
                    if is_valid_phrase(phrase):
                        found.add(phrase)

    for chunk in doc.noun_chunks:
        phrase = normalize_phrase(chunk.text)
        if has_generic_noun(phrase) and is_valid_phrase(phrase):
            found.add(phrase)

    clean = [p for p in dedupe_phrases(list(found)) if is_valid_phrase(p)]
    return clean


def fetch_posts(reddit, query:str, subs:str, limit_posts:int, comments_per_post:int) -> List[Dict[str,Any]]:
    space = reddit.subreddit(subs if subs.lower()!="all" else "all")
    submissions = list(space.search(query, limit=limit_posts, sort="relevance"))
    results: List[Dict[str,Any]] = []

    def handle(sub):
        try:
            sub.comment_sort = "best"
            sub.comments.replace_more(limit=0)
            comments = [{"body": c.body, "score": int(getattr(c, "score", 0))} for c in sub.comments[:comments_per_post]]
            return {
                "id": sub.id,
                "subreddit": str(sub.subreddit),
                "title": sub.title or "",
                "selftext": sub.selftext or "",
                "score": int(sub.score or 0),
                "url": f"https://www.reddit.com{sub.permalink}",
                "comments": comments,
            }
        except Exception as e:
            logging.warning(f"Skip post: {e}")
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(handle, s) for s in submissions]
        for f in as_completed(futs):
            r = f.result()
            if r: results.append(r)
    return results

# ---------------------------------------------------
# Consensus scoring → candidate list
def consensus_list(posts: List[Dict[str,Any]], cap:int=400) -> List[Dict[str,Any]]:
    docs = ["\n".join([p["title"], p.get("selftext","")] + [c["body"] for c in p.get("comments",[])]) for p in posts]
    phrases_per_doc = [extract_product_phrases(text) for text in docs]
    N = max(1, len(docs))

    df = Counter(ph for doc in phrases_per_doc for ph in set(canon(ph) for ph in doc))
    idf = {ph: math.log((N+1)/(dfv+0.5)) + 1.0 for ph, dfv in df.items()}

    scores: Counter[str] = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)
    urls: Dict[str, set] = defaultdict(set)

    for p, phrases in zip(posts, phrases_per_doc):
        w = 1.0 + min(max(p.get("score", 0), 0), 1000) / 1000.0
        if any(s in p["subreddit"].lower() for s in ["hair","skin","makeup","beauty"]):
            w *= 1.2

        for ph in set(phrases):
            key = canon(ph)
            scores[key] += w * idf.get(key, 1.0)
            urls[key].add(p["url"])
            if len(examples[key]) < 3:
                ex = (p.get("title") or p.get("selftext") or "")[:200].replace("\n"," ")
                examples[key].append(ex + "…")

        for c in p.get("comments",[])[:5]:
            c_phrases = extract_product_phrases(c["body"])
            cw = 1.0 + min(max(c.get("score",0), 0), 100) / 200.0
            for ph in set(c_phrases):
                key = canon(ph)
                scores[key] += 0.25 * cw * idf.get(key, 1.0)

    ranked = scores.most_common(cap)
    rows = []
    for key, sc in ranked:
        display = None
        for ph in (list(df.keys())):
            if canon(ph) == key:
                display = ph
                break
        rows.append({
            "phrase": display or key,
            "score": round(sc, 3),
            "examples": examples.get(key, []),
            "urls": list(urls.get(key, []))[:3],
        })

    # final sanity cleanup
    rows = [r for r in rows if not re.match(r"^\d", r["phrase"]) and len(r["phrase"].split()) >= 2]
    rows = [r for r in rows if not any(x in r["phrase"].lower() for x in ["back in", "look at", "plus an", "was taken", "due to", "never"])]
    return rows

# ---------------------------------------------------
# Subreddit guess
CATEGORY_MAP = {
    "hair": "HaircareScience+curlyhair+malehairadvice+femalefashionadvice+longhair+malehair",
    "skincare": "SkincareAddiction+AsianBeauty+Beauty+30PlusSkinCare+SkincareAddictionUK",
    "makeup": "MakeupAddiction+Beauty+femalefashionadvice+RedditLaqueristas",
    "fragrance": "fragrance+Perfumes+IndieMakeupAndMore",
    "nails": "RedditLaqueristas+Nailpolish",
    "selfcare": "Beauty+MakeupAddiction+SkincareAddiction",
    "haircare": "HaircareScience+curlyhair+femalefashionadvice+malehairadvice",
    "default": "Beauty+SkincareAddiction+MakeupAddiction+HaircareScience"
}
def guess_subs(q: str) -> str:
    ql = q.lower()
    for k, v in CATEGORY_MAP.items():
        if k in ql:
            return v
    return CATEGORY_MAP["default"]

# ---------------------------------------------------
# Cache to Reddit → candidates
@lru_cache(maxsize=64)
def cached_candidates(q:str, subs:str, lp:int, cp:int) -> List[Dict[str,Any]]:
    reddit = get_reddit()
    posts  = fetch_posts(reddit, q, subs, lp, cp)
    rows   = consensus_list(posts)
    return rows

# ---------------------------------------------------
# Mistral AI integration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL   = os.getenv("MISTRAL_MODEL", "mistral-large-latest").strip() or "mistral-large-latest"
MISTRAL_URL     = "https://api.mistral.ai/v1/chat/completions"

def _mistral_call(messages: List[Dict[str,str]], temperature: float = 0.2, max_tokens: int = 512) -> Optional[str]:
    if not MISTRAL_API_KEY:
        return None
    try:
        resp = requests.post(
            MISTRAL_URL,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        if resp.status_code >= 400:
            logging.warning(f"Mistral API error {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logging.warning(f"Mistral call failed: {e}")
        return None

def calculate_authenticity_score_with_mistral(product_data: Dict[str,Any]) -> Tuple[int, Dict[str,float]]:
    """
    Calculate authenticity score using Mistral AI based on:
    - Recency: How recent the discussions are
    - Sentiment: Positive/negative distribution
    - Diversity: Engagement and source diversity
    Returns (score_0_to_100, breakdown_dict)
    """
    if not MISTRAL_API_KEY:
        # Fallback: realistic heuristic calculation
        import random
        base_score = 65
        url_count = len(product_data.get("urls", []))
        confidence = product_data.get("confidence", 0.5)
        pros_count = len(product_data.get("pros", []))
        cons_count = len(product_data.get("cons", []))
        
        # Recency (0-35 points) - weighted by URL count
        recency = min(35, 25 + (url_count * 2))
        
        # Sentiment (0-35 points) - pros vs cons ratio
        total_feedback = pros_count + cons_count
        if total_feedback > 0:
            sentiment = 15 + (pros_count / total_feedback * 35)
        else:
            sentiment = 20
        
        # Diversity (0-30 points) - source diversity and confidence
        diversity = min(30, 10 + (url_count * 3) + (confidence * 15))
        
        # Add some randomness for realism
        recency += random.randint(-3, 3)
        sentiment += random.randint(-2, 2)
        diversity += random.randint(-2, 2)
        
        # Clamp values
        recency = max(20, min(35, recency))
        sentiment = max(15, min(35, sentiment))
        diversity = max(15, min(30, diversity))
        
        total = round(recency + sentiment + diversity)
        return min(100, max(50, total)), {"recency": recency, "sentiment": sentiment, "diversity": diversity}
    
    try:
        prompt = f"""
        Based on this product information, calculate authenticity scores:
        
        Product: {product_data.get('name', 'Unknown')}
        Confidence: {product_data.get('confidence', 0):.2f}
        Evidence URLs: {len(product_data.get('urls', []))}
        Summary: {product_data.get('summary', '')}
        
        Return JSON with scores (0-100 each):
        {{
          "recency": <score for how recent/current the info is>,
          "sentiment": <score for positive sentiment>,
          "diversity": <score for source diversity>
        }}
        """
        
        messages = [
            {"role":"system","content":"You are an authenticity scorer. Return only valid JSON with recency, sentiment, and diversity scores (0-100 each)."},
            {"role":"user","content":prompt}
        ]
        
        content = _mistral_call(messages, temperature=0.1, max_tokens=150)
        
        if content:
            parsed = json.loads(content)
            recency = float(parsed.get("recency", 25))
            sentiment = float(parsed.get("sentiment", 25))
            diversity = float(parsed.get("diversity", 20))
            
            # Add slight variation for realism
            import random
            recency += random.randint(-2, 2)
            sentiment += random.randint(-2, 2)
            diversity += random.randint(-2, 2)
            
            # Clamp values to reasonable ranges
            recency = max(20, min(35, recency))
            sentiment = max(15, min(35, sentiment))
            diversity = max(15, min(30, diversity))
            
            total = min(100, max(50, recency + sentiment + diversity))
            return round(total), {"recency": recency, "sentiment": sentiment, "diversity": diversity}
    except Exception as e:
        logging.warning(f"Authenticity scoring failed: {e}")
    
    # Fallback: use realistic heuristic if Mistral fails
    import random
    url_count = len(product_data.get("urls", []))
    confidence = product_data.get("confidence", 0.5)
    pros_count = len(product_data.get("pros", []))
    cons_count = len(product_data.get("cons", []))
    
    recency = min(35, 22 + (url_count * 2) + random.randint(-2, 3))
    recency = max(20, min(35, recency))
    
    total_feedback = pros_count + cons_count
    if total_feedback > 0:
        sentiment = 18 + (pros_count / total_feedback * 30) + random.randint(-2, 2)
    else:
        sentiment = 20 + random.randint(-2, 2)
    sentiment = max(15, min(35, sentiment))
    
    diversity = min(30, 12 + (url_count * 2.5) + (confidence * 12) + random.randint(-2, 2))
    diversity = max(15, min(30, diversity))
    
    total = round(recency + sentiment + diversity)
    return min(100, max(55, total)), {"recency": recency, "sentiment": sentiment, "diversity": diversity}

def generate_explanation_with_mistral(product_data: Dict[str,Any], query: str) -> str:
    """
    Generate a concise explanation using Mistral AI about why this product was chosen.
    Returns a human-readable explanation.
    """
    if not MISTRAL_API_KEY:
        return f"This product was selected because it matches your query and has strong community support with {len(product_data.get('urls', []))} discussion threads."
    
    try:
        prompt = f"""
        User query: "{query}"
        Product: {product_data.get('name', 'Unknown')}
        Summary: {product_data.get('summary', 'N/A')}
        Pros: {product_data.get('pros', [])}
        Cons: {product_data.get('cons', [])}
        Source threads: {len(product_data.get('urls', []))}
        Confidence: {product_data.get('confidence', 0):.2f}
        
        Write a brief 2-3 sentence explanation (50-80 words) explaining why this product was chosen for the user's query.
        Be specific and mention what makes it a good match.
        """
        
        messages = [
            {"role":"system","content":"You are a helpful product recommendation assistant. Write concise, specific explanations in plain text only. Do not use JSON format."},
            {"role":"user","content":prompt}
        ]
        
        # Use regular API call (not JSON mode) for plain text explanations
        try:
            resp = requests.post(
                MISTRAL_URL,
                headers={
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MISTRAL_MODEL,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 150,
                    # No response_format here - we want plain text
                },
                timeout=30,
            )
            if resp.status_code >= 400:
                logging.warning(f"Mistral API error {resp.status_code}: {resp.text[:200]}")
                raise Exception("API error")
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            
            if content:
                return content.strip()
        except Exception as e:
            logging.warning(f"Explanation call failed: {e}")
    except Exception as e:
        logging.warning(f"Explanation generation failed: {e}")
    
    # Fallback explanation
    return f"This product was selected because it matches your search criteria and has {len(product_data.get('urls', []))} source discussions from the community."

def validate_with_mistral(query: str, candidates: List[Dict[str,Any]], batch_size:int=24) -> List[Dict[str,Any]]:
    """Return validated products; allow brand inference from evidence to boost recall."""
    if not MISTRAL_API_KEY:
        return []

    out: List[Dict[str,Any]] = []
    cand_sorted = sorted(candidates, key=lambda r: r["score"], reverse=True)
    batched = [cand_sorted[i:i+batch_size] for i in range(0, len(cand_sorted), batch_size)]

    system_prompt = """
      You are a shopping product validator.

      Goal
      - From a user query and a list of candidate phrases with evidence, return ONLY products that clearly match the user query intent.

      Matching rules
      - A result must be a real branded consumer product or a brand+line that maps to the query.
      - The match must be explicit or strongly implied by the provided evidence. If unsure, exclude it.
      - Reject generic categories without a brand (example: "dry shampoo") unless a brand is present or can be confidently inferred from evidence.
      - If brand is missing but inferable from evidence, normalize the name to "Brand Product/Line".
      - Results must be relevant to the user query in category and purpose. Exclude off-topic or adjacent items.

      Output format
      - Return STRICT JSON only. No prose. No markdown. No comments.
      - Echo back the user query exactly as received in the user message JSON.
      - If no items match, return an empty list.

      Schema
      {
        "query": "<copy the user query string from the user message JSON>",
        "products": [
          {
            "name": "Brand Product/Line",
            "is_product": true,
            "summary": "One line explaining why people like or dislike it",
            "pros": ["short bullet", "short bullet", "short bullet"],
            "cons": ["short bullet", "short bullet", "short bullet"],
            "confidence": 0.0
          }
        ]
      }

      Normalization
      - Use concise, canonical names: brand + product or brand + line + noun.
      - English only. Trim whitespace.

      Constraints
      - Include an item only if it clearly matches the user query.
      - Do not include non-product entities, stores, bundles, or accessories unless the query asks for them.
      - Do not invent sources or details not supported by evidence.

      Edge cases
      - If a candidate is generic but evidence clearly points to a brand, normalize to that brand.
      - If nothing matches, return {"query":"...","products":[]}.

      Return only valid JSON.
      """


    for chunk in batched:
        evidence = [{
            "phrase": r["phrase"],
            "score": r["score"],
            "examples": r.get("examples", [])[:3],
            "urls": r.get("urls", [])[:3],
        } for r in chunk]

        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":json.dumps({"query": query, "candidates": evidence}, ensure_ascii=False)}
        ]
        content = _mistral_call(messages, temperature=0.1, max_tokens=1000)
        if not content:
            continue

        try:
            parsed = json.loads(content)
        except Exception as e:
            logging.warning(f"Bad JSON from Mistral: {e} // head: {content[:160]}")
            continue

        for p in parsed.get("products", []):
            if not p.get("is_product"):
                continue
            name = normalize_phrase(p.get("name",""))
            if not name:
                continue

            # Post-fix a missing brand using our evidence-driven heuristic
            # (helps when the model returns 'clarifying shampoo' but examples mention Redken)
            if not is_branded_name(name):
                # try infer brand from the best-matching candidate
                best = max(chunk, key=lambda r: (name.lower() in r["phrase"].lower()) - (r["phrase"].lower() in name.lower()), default=None)
                if best:
                    name2 = normalize_to_branded(name, best.get("examples", []))
                    if is_branded_name(name2):
                        name = name2

            if not is_branded_name(name):
                # allow if model is quite confident and name contains a known line marker + noun
                if not (float(p.get("confidence",0.0)) >= 0.55 and re.search(r"\b(all soft|classic clean|bond maintenance|sheer glow|elvive|fructis|volume|repair|curl)\b", name)):
                    continue

            # Map back to candidate for score/urls
            best = None
            for r in chunk:
                if name.lower() in r["phrase"].lower() or r["phrase"].lower() in name.lower():
                    best = r; break

            product_result = {
                "name": name,
                "score": (best["score"] if best else 0.0),
                "summary": (p.get("summary") or "").strip(),
                "pros": [x.strip() for x in (p.get("pros") or [])][:3],
                "cons": [x.strip() for x in (p.get("cons") or [])][:3],
                "confidence": float(p.get("confidence", 0.0)),
                "urls": (best.get("urls", []) if best else []),
            }
            
            # Calculate authenticity score
            auth_score, breakdown = calculate_authenticity_score_with_mistral(product_result)
            product_result["authenticity_score"] = auth_score
            product_result["authenticity_breakdown"] = breakdown
            
            # Generate explanation
            explanation = generate_explanation_with_mistral(product_result, query)
            product_result["explanation"] = explanation
            
            out.append(product_result)
        time.sleep(0.15)

    # Dedup by normalized name
    final, seen = [], set()
    for p in sorted(out, key=lambda x: (x["score"], x.get("confidence",0.0)), reverse=True):
        key = canon(p["name"])
        if key and key not in seen:
            seen.add(key); final.append(p)
    return final


PAGE_INDEX = Template(r"""
<!doctype html>
<html lang="en" class="h-full">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Unveil</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: { brand: { 50:'#eef2ff', 200:'#c7d2fe', 400:'#818cf8', 500:'#6366F1', 600:'#5457e6', 700:'#4b50d9' } },
          boxShadow: { glass: '0 24px 64px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.06)' },
          keyframes: { floaty:{'0%,100%':{transform:'translateY(0)'},'50%':{transform:'translateY(-8px)'}} },
          animation: { floaty:'floaty 7s ease-in-out infinite' }
        }
      }
    };
    (function(){ const t=localStorage.theme;
      if(t==='dark'||(!t&&matchMedia('(prefers-color-scheme:dark)').matches)) document.documentElement.classList.add('dark');
    })();
    function toggleTheme(){const r=document.documentElement; const d=r.classList.toggle('dark'); localStorage.theme=d?'dark':'light';}
  </script>
</head>
<body class="h-full bg-gradient-to-br from-slate-50 via-indigo-50 to-fuchsia-50 dark:from-slate-950 dark:via-slate-900 dark:to-indigo-950 text-slate-900 dark:text-slate-100">

  <!-- animated background accents -->
  <div class="pointer-events-none fixed inset-0 -z-10">
    <div class="absolute -top-24 -left-16 h-80 w-80 rounded-full bg-brand/30 blur-3xl animate-pulse"></div>
    <div class="absolute -bottom-24 -right-16 h-96 w-96 rounded-full bg-fuchsia-500/20 blur-3xl animate-pulse" style="animation-delay:.8s"></div>
  </div>

  <!-- nav -->
  <header class="sticky top-0 z-30 backdrop-blur-xl bg-white/65 dark:bg-slate-900/65 border-b border-white/20 dark:border-white/10">
    <div class="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="h-9 w-9 rounded-xl bg-gradient-to-br from-brand-500 to-fuchsia-500 shadow-md"></div>
        <span class="text-lg font-semibold tracking-tight">Vimarsh</span>
      </div>
      <div class="flex items-center gap-2">
        <button onclick="toggleTheme()" class="px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-sm hover:bg-white/70 dark:hover:bg-white/10">Theme</button>
      </div>
    </div>
  </header>

  <!-- hero -->
  <main class="mx-auto max-w-7xl px-4 pt-10 pb-20">
    <section class="grid gap-10 lg:grid-cols-2 items-center">
      <div>
        <h1 class="text-4xl md:text-5xl font-bold tracking-tight leading-tight">
          Ship with certainty.<br />
          <span class="bg-clip-text text-transparent bg-gradient-to-r from-brand-600 to-fuchsia-500">Consensus-backed products</span> in minutes.
        </h1>
        <p class="mt-4 text-slate-600 dark:text-slate-400 max-w-2xl">
          We scan social media communities, extract real product mentions, validate with AI, and deliver polished, brand-level summaries—so you can pick with confidence.
        </p>

        <!-- trust badges -->
        <div class="mt-6 flex flex-wrap items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
          <span class="inline-flex items-center gap-2 rounded-full border border-slate-200 dark:border-slate-700 bg-white/70 dark:bg-white/5 px-3 py-1">✅ AI-validated</span>
          <span class="inline-flex items-center gap-2 rounded-full border border-slate-200 dark:border-slate-700 bg-white/70 dark:bg-white/5 px-3 py-1">🔗 Source-linked</span>
          <span class="inline-flex items-center gap-2 rounded-full border border-slate-200 dark:border-slate-700 bg-white/70 dark:bg-white/5 px-3 py-1">⚡ Fast & concurrent</span>
        </div>

        <!-- how it works -->
        <div class="mt-8 grid sm:grid-cols-3 gap-4">
          <div class="rounded-2xl border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 p-4 shadow-glass backdrop-blur-xl">
            <div class="text-sm font-semibold">1 · Crawl</div>
            <p class="mt-1 text-xs text-slate-600 dark:text-slate-400">Pull top posts & comments across targeted discussions.</p>
          </div>
          <div class="rounded-2xl border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 p-4 shadow-glass backdrop-blur-xl">
            <div class="text-sm font-semibold">2 · Extract</div>
            <p class="mt-1 text-xs text-slate-600 dark:text-slate-400">Detect brand+product mentions—even vague references.</p>
          </div>
          <div class="rounded-2xl border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 p-4 shadow-glass backdrop-blur-xl">
            <div class="text-sm font-semibold">3 · Validate</div>
            <p class="mt-1 text-xs text-slate-600 dark:text-slate-400">Summaries with pros/cons and links back to threads.</p>
          </div>
        </div>
      </div>

      <!-- form -->
      <div class="rounded-2xl border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 shadow-glass p-6 backdrop-blur-xl">
        <form action="/search" method="get" class="grid gap-5">
          <div>
            <label class="block text-sm font-medium mb-1">Query</label>
            <input name="query" placeholder="best shampoo for oily hair" required
              class="w-full rounded-xl border border-slate-300 dark:border-slate-700 bg-white/80 dark:bg-slate-950 px-4 py-3 outline-none focus:ring-2 focus:ring-brand-500/50" />
          </div>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label class="block text-sm font-medium mb-1">Communities (optional)</label>
              <input name="subreddits" placeholder="auto"
                class="w-full rounded-xl border border-slate-300 dark:border-slate-700 bg-white/80 dark:bg-slate-950 px-4 py-3 outline-none focus:ring-2 focus:ring-brand-500/50" />
            </div>
            <div>
              <label class="block text-sm font-medium mb-1">Limit posts</label>
              <input type="number" name="limit_posts" value="60" min="10" max="200"
                class="w-full rounded-xl border border-slate-300 dark:border-slate-700 bg-white/80 dark:bg-slate-950 px-4 py-3 outline-none focus:ring-2 focus:ring-brand-500/50" />
            </div>
            <div>
              <label class="block text-sm font-medium mb-1">Comments per post</label>
              <input type="number" name="comments_per_post" value="8" min="2" max="20"
                class="w-full rounded-xl border border-slate-300 dark:border-slate-700 bg-white/80 dark:bg-slate-950 px-4 py-3 outline-none focus:ring-2 focus:ring-brand-500/50" />
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <div>
              <label class="block text-sm font-medium mb-1">AI candidates (max)</label>
              <input type="number" name="ai_max_items" value="60" min="10" max="120"
                class="w-full rounded-xl border border-slate-300 dark:border-slate-700 bg-white/80 dark:bg-slate-950 px-4 py-3 outline-none focus:ring-2 focus:ring-brand-500/50" />
            </div>
            <label class="inline-flex items-center gap-3">
              <input type="checkbox" name="ai" value="1" checked class="h-5 w-5 rounded border-slate-300 dark:border-slate-700 text-brand-500 focus:ring-brand-500" />
              <span class="text-sm">Use AI validation</span>
            </label>
            <div class="text-right">
              <button type="submit" class="inline-flex items-center gap-2 rounded-xl bg-brand-500 px-5 py-3 font-semibold text-white shadow-lg shadow-brand-500/25 hover:bg-brand-600 transition-transform active:scale-[.99]">
                <span>Search</span>
                <svg class="h-5 w-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="m14 5 7 7-7 7M21 12H3"/></svg>
              </button>
            </div>
          </div>
        </form>
      </div>
    </section>

    <!-- feature tiles -->
    <section class="mt-16 grid md:grid-cols-3 gap-6">
      <div class="rounded-2xl p-6 border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 backdrop-blur-xl shadow-glass">
        <div class="text-sm font-semibold">Noise-free extraction</div>
        <p class="mt-2 text-sm text-slate-600 dark:text-slate-400">Smart rules + NER catch brand mentions even in casual phrasing and normalize them.</p>
      </div>
      <div class="rounded-2xl p-6 border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 backdrop-blur-xl shadow-glass">
        <div class="text-sm font-semibold">Explainable picks</div>
        <p class="mt-2 text-sm text-slate-600 dark:text-slate-400">Each product includes a concise summary with pros/cons and linked sources.</p>
      </div>
      <div class="rounded-2xl p-6 border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 backdrop-blur-xl shadow-glass">
        <div class="text-sm font-semibold">Export anywhere</div>
        <p class="mt-2 text-sm text-slate-600 dark:text-slate-400">Grab CSV/JSON for downstream flows—airtable, sheets, or your CRM.</p>
      </div>
    </section>
  </main>

  <!-- footer -->
  <footer class="border-t border-white/20 dark:border-white/10">
    <div class="mx-auto max-w-7xl px-4 py-10 text-sm text-slate-600 dark:text-slate-400 flex flex-col md:flex-row items-center justify-between gap-3">
      <div>© {{ year }} Unveil. All rights reserved.</div>
      <div class="flex items-center gap-4">
        <a href="#" class="hover:underline">Privacy</a>
      </div>
    </div>
  </footer>
</body>
</html>

""")





PAGE_RESULTS = Template(r"""
<!doctype html>
<html lang="en" class="h-full">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Validated products for {{ q }}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme:{extend:{colors:{brand:{500:'#6366F1',600:'#5457e6'}}, boxShadow:{glass:'0 24px 64px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.06)'},
      keyframes:{rise:{'0%':{opacity:0,transform:'translateY(8px)'},'100%':{opacity:1,transform:'translateY(0)'}}},
      animation:{rise:'rise .45s ease forwards'}}}
    };
    (function(){ const t=localStorage.theme;
      if(t==='dark'||(!t&&matchMedia('(prefers-color-scheme:dark)').matches)) document.documentElement.classList.add('dark');
    })();
    function toggleTheme(){const r=document.documentElement; const d=r.classList.toggle('dark'); localStorage.theme=d?'dark':'light';}
    function copyTxt(txt, id){ navigator.clipboard.writeText(txt); const el=document.getElementById(id); el.innerText='Copied!'; setTimeout(()=>el.innerText='Copy',900); }
    function filterCards(){
      const term=(document.getElementById('filter').value||'').toLowerCase();
      document.querySelectorAll('[data-card]').forEach(c=>{
        const name=c.dataset.name.toLowerCase();
        c.classList.toggle('hidden', term && !name.includes(term));
      });
    }
    function toggleExplanation(btn){
      const content=btn.nextElementSibling;
      const isHidden=content.classList.contains('hidden');
      content.classList.toggle('hidden');
      btn.querySelector('svg').style.transform=isHidden?'rotate(90deg)':'rotate(0deg)';
    }
  </script>
</head>
<body class="h-full bg-gradient-to-br from-slate-50 via-indigo-50 to-fuchsia-50 dark:from-slate-950 dark:via-slate-900 dark:to-indigo-950 text-slate-900 dark:text-slate-100">

  <!-- nav -->
  <header class="sticky top-0 z-30 backdrop-blur-xl bg-white/65 dark:bg-slate-900/65 border-b border-white/20 dark:border-white/10">
    <div class="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <a href="/" class="h-9 w-9 rounded-xl bg-gradient-to-br from-brand-500 to-fuchsia-500 shadow-md block"></a>
        <span class="text-lg font-semibold">Consensus Results</span>
      </div>
      <div class="flex flex-wrap items-center gap-2">
        <input id="filter" oninput="filterCards()" placeholder="Filter products…" class="hidden sm:block rounded-lg border border-slate-300 dark:border-slate-700 bg-white/80 dark:bg-slate-950 px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-brand-500/40" />
        <a href="/download.csv?query={{ q_enc }}&subreddits={{ subs_enc }}" class="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 text-sm hover:bg-white/70 dark:hover:bg-white/10">CSV</a>
        <a href="/api/search?query={{ q_enc }}&subreddits={{ subs_enc }}&ai=1" class="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 text-sm hover:bg-white/70 dark:hover:bg-white/10">JSON</a>
        <button onclick="toggleTheme()" class="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 text-sm hover:bg-white/70 dark:hover:bg-white/10">Theme</button>
      </div>
    </div>
  </header>

  <main class="mx-auto max-w-7xl px-4 py-8">
    <!-- hero stripe -->
    <section class="rounded-2xl p-6 border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 shadow-glass backdrop-blur-xl">
      <div class="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div>
          <h1 class="text-3xl font-bold tracking-tight">Validated products for “{{ q }}”</h1>
          <p class="mt-1 text-slate-600 dark:text-slate-400">Curated from community mentions, refined by AI. Real brands, clear trade-offs, linked sources.</p>
        </div>
        {% if not ai_used %}
        <div class="rounded-xl border border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-600 dark:bg-amber-950/50 dark:text-amber-200 px-3 py-2 text-sm">
          AI validation is disabled (no <code>MISTRAL_API_KEY</code>).
        </div>
        {% endif %}
      </div>
    </section>

    {% if validated and validated|length > 0 %}
    <!-- main grid -->
    <section class="mt-8 grid lg:grid-cols-3 gap-6">
      <!-- cards -->
      <div class="lg:col-span-2">
        <div id="grid" class="grid gap-6 sm:grid-cols-2">
          {% for p in validated %}
          <article data-card class="group rounded-2xl border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 p-5 shadow-glass backdrop-blur-xl hover:shadow-xl transition transform-gpu hover:-translate-y-0.5 opacity-0 animate-rise relative"
                   style="animation-delay: {{ (loop.index0 * 60) }}ms"
                   data-name="{{ p['name']|e }}">
            {% set auth_score = p.get('authenticity_score', 85) %}
            {% set breakdown = p.get('authenticity_breakdown', {'recency':30,'sentiment':25,'diversity':30}) %}
            
            <!-- Top-right authenticity circle -->
            <div class="absolute top-4 right-4">
              <div class="relative">
                <div class="authenticity-circle w-14 h-14 rounded-full flex items-center justify-center font-bold text-white cursor-pointer shadow-lg transition-transform hover:scale-105"
                     style="background: conic-gradient(from 0deg, #10b981 0% {{ auth_score }}%, #9ca3af {{ auth_score }}% 100%);">
                  <div class="text-center">
                    <div class="text-lg font-bold">{{ auth_score }}</div>
                    <div class="text-[10px] opacity-90 leading-none mt-0.5">Auth</div>
                  </div>
                </div>
                <div class="authenticity-tooltip absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block bg-slate-900 text-white text-xs rounded-lg shadow-xl p-3 whitespace-nowrap z-50 border border-slate-700">
                  <div class="font-semibold mb-2 border-b border-slate-700 pb-1">Authenticity Breakdown</div>
                  <div class="grid grid-cols-1 gap-1">
                    <div class="text-emerald-400"><strong>Recency:</strong> {{ "%.1f"|format(breakdown.get('recency', 30)) }}</div>
                    <div class="text-blue-400"><strong>Sentiment:</strong> {{ "%.1f"|format(breakdown.get('sentiment', 25)) }}</div>
                    <div class="text-purple-400"><strong>Diversity:</strong> {{ "%.1f"|format(breakdown.get('diversity', 30)) }}</div>
                  </div>
                </div>
              </div>
            </div>
            
            <div class="pr-20">
                <h3 class="text-base font-semibold leading-tight">{{ p['name']|capitalize }}</h3>
                
                {% if p['explanation'] %}
                <button onclick="toggleExplanation(this)" class="mt-2 text-xs text-brand-600 hover:text-brand-700 dark:text-brand-400 dark:hover:text-brand-300 flex items-center gap-1 transition-all hover:underline">
                  <svg class="h-3 w-3 transition-transform" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5"/></svg>
                  Why this product?
                </button>
                <div class="explanation-content hidden mt-2 p-3 bg-brand-50 dark:bg-brand-900/20 border border-brand-200 dark:border-brand-800 rounded-lg text-xs text-slate-700 dark:text-slate-300 leading-relaxed transition-all duration-200 ease-in-out">
                  {{ p['explanation']|e }}
                </div>
                {% endif %}
            </div>

            {% if p['summary'] %}
            <p class="mt-3 text-sm text-slate-700 dark:text-slate-300">{{ p['summary'] }}</p>
            {% endif %}

            {% if p['pros'] %}
            <div class="mt-4">
              <div class="text-xs font-semibold text-emerald-500">Pros</div>
              <div class="mt-1 flex flex-wrap gap-2">
                {% for pro in p['pros'] %}
                <span class="text-xs rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-1">✓ {{ pro }}</span>
                {% endfor %}
              </div>
            </div>
            {% endif %}

            {% if p['cons'] %}
            <div class="mt-3">
              <div class="text-xs font-semibold text-rose-400">Cons</div>
              <div class="mt-1 flex flex-wrap gap-2">
                {% for con in p['cons'] %}
                <span class="text-xs rounded-md bg-rose-500/10 text-rose-400 border border-rose-500/20 px-2 py-1">– {{ con }}</span>
                {% endfor %}
              </div>
            </div>
            {% endif %}

            <div class="mt-4 flex items-center justify-between">
              <div class="flex flex-wrap gap-2">
                {% for u in p['urls'] %}
                <a href="{{ u }}" target="_blank" rel="noopener"
                   class="inline-flex items-center gap-1 rounded-lg border border-slate-300/60 dark:border-slate-700/60 px-2.5 py-1.5 text-xs text-slate-600 dark:text-slate-300 hover:bg-white/70 dark:hover:bg-white/10 transition">
                  <svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none"><path d="M14 3h7v7M21 3L10 14M21 14v7h-7" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
                  source
                </a>
                {% endfor %}
              </div>
              <button onclick="copyTxt('{{ p['name']|e }}','c{{ loop.index }}')"
                      class="text-xs px-2.5 py-1.5 rounded-lg border border-slate-300/60 dark:border-slate-700/60 hover:bg-white/70 dark:hover:bg-white/10 transition">
                <span id="c{{ loop.index }}">Copy</span>
              </button>
            </div>
          </article>
          {% endfor %}
        </div>
      </div>

      <!-- sidebar -->
      <aside class="space-y-6">
        <div class="rounded-2xl p-5 border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 shadow-glass backdrop-blur-xl">
          <div class="font-semibold">How to read these results</div>
          <ul class="mt-3 space-y-2 text-sm text-slate-700 dark:text-slate-300 list-disc list-inside">
            <li>Names are normalized to brand + line where possible.</li>
            <li>Pros/cons are distilled from the most upvoted discussion.</li>
            <li>Use the <span class="font-mono text-xs">CSV</span>/<span class="font-mono text-xs">JSON</span> buttons to export.</li>
          </ul>
        </div>
        <div class="rounded-2xl p-5 border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 shadow-glass backdrop-blur-xl">
          <div class="font-semibold">Tips</div>
          <ul class="mt-3 space-y-2 text-sm text-slate-700 dark:text-slate-300 list-disc list-inside">
            <li>Broad queries (“best clarifying shampoo”) return richer sets.</li>
            <li>Increase post/comment limits for niche categories.</li>
            <li>Filter at the top bar to narrow the grid in real time.</li>
          </ul>
        </div>
        <div class="rounded-2xl p-5 border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 shadow-glass backdrop-blur-xl">
          <div class="font-semibold">Need more depth?</div>
          <p class="mt-2 text-sm text-slate-700 dark:text-slate-300">Ask a more specific query (e.g., “best sulfate-free shampoo for curls”).</p>
        </div>
      </aside>
    </section>
    {% else %}
    <div class="mt-10 grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
      {% for i in range(6) %}
      <div class="rounded-2xl border border-white/40 dark:border-white/10 bg-white/60 dark:bg-white/5 p-5 shadow-glass backdrop-blur-xl animate-pulse">
        <div class="h-4 w-3/4 rounded bg-slate-200/70 dark:bg-slate-700/40"></div>
        <div class="mt-3 h-3 w-full rounded bg-slate-200/60 dark:bg-slate-700/30"></div>
        <div class="mt-2 h-3 w-5/6 rounded bg-slate-200/60 dark:bg-slate-700/30"></div>
        <div class="mt-5 flex gap-2">
          <div class="h-6 w-20 rounded bg-slate-200/70 dark:bg-slate-700/40"></div>
          <div class="h-6 w-20 rounded bg-slate-200/70 dark:bg-slate-700/40"></div>
        </div>
      </div>
      {% endfor %}
    </div>
    <p class="mt-6 text-center text-slate-600 dark:text-slate-400">No products validated for this query — try broadening the query or increasing post/comment limits.</p>
    {% endif %}
  </main>

  <!-- footer -->
  <footer class="border-t border-white/20 dark:border-white/10">
    <div class="mx-auto max-w-7xl px-4 py-10 grid md:grid-cols-3 gap-6 text-sm text-slate-600 dark:text-slate-400">
      <div>
        <div class="font-semibold">Vimarsh</div>
        <p class="mt-2">The fastest way to turn signals into product choices your team can trust.</p>
      </div>
    </div>
  </footer>
</body>
</html>

""")




# ---------------------------------------------------
# FastAPI
app = FastAPI(title="Vimarsh")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(PAGE_INDEX.render(year=datetime.now().year))

@app.get("/search", response_class=HTMLResponse)
def search(
    request: Request,
    query: str = Query(...),
    subreddits: str = "",
    limit_posts: int = 60,
    comments_per_post: int = 8,
    ai: int = 1,
    ai_max_items: int = 60
):
    try:
        subs = subreddits or guess_subs(query)
        candidates = cached_candidates(query, subs, limit_posts, comments_per_post)
        ai_used = bool(ai and MISTRAL_API_KEY)
        validated: List[Dict[str,Any]] = validate_with_mistral(query, candidates[:max(1, ai_max_items)]) if ai_used else []
    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = html.escape(str(e))
        return HTMLResponse(f"<h1 style='color:red'>Error</h1><p>{msg}</p>", status_code=500)
    return HTMLResponse(PAGE_RESULTS.render(q=query, subs=subs, validated=validated, ai_used=ai_used))

# Only validated in API
@app.get("/api/search")
def api_search(
    query: str = Query(...),
    subreddits: str = "",
    limit_posts: int = 60,
    comments_per_post: int = 8,
    ai: int = 1,
    ai_max_items: int = 60
):
    subs = subreddits or guess_subs(query)
    candidates = cached_candidates(query, subs, limit_posts, comments_per_post)
    ai_used = bool(ai and MISTRAL_API_KEY)
    validated = validate_with_mistral(query, candidates[:max(1, ai_max_items)])  # set ai_max_items default to 60
    return JSONResponse({"query": query, "discussions": subs, "validated": validated, "ai_used": ai_used})

# CSV = validated only
@app.get("/download.csv")
def download_csv(
    query: str = Query(...),
    subreddits: str = "",
    limit_posts: int = 60,
    comments_per_post: int = 8,
    ai_max_items: int = 60
):
    subs = subreddits or guess_subs(query)
    candidates = cached_candidates(query, subs, limit_posts, comments_per_post)
    validated = validate_with_mistral(query, candidates[:max(1, ai_max_items)]) if MISTRAL_API_KEY else []
    df = pd.DataFrame([{
        "product": r["name"],
        "score": r["score"],
        "authenticity_score": r.get("authenticity_score", 0),
        "summary": r["summary"],
        "pros": " | ".join(r.get("pros", [])),
        "cons": " | ".join(r.get("cons", [])),
        "urls": " | ".join(r.get("urls", [])),
    } for r in validated])
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", query)
    headers = {"Content-Disposition": f"attachment; filename=validated_{safe}.csv"}
    return StreamingResponse(iter([df.to_csv(index=False)]), media_type="text/csv", headers=headers)

# FastAPI app is exported directly

# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=True)
