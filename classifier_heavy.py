#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cse_relevance_heavy_v3.py
-------------------------------------------------
Improved Heavy Classifier for Critical Sector Entity (CSE) Relevance

Performs in-depth analysis on domains flagged by the prefilter.

Checks:
  ✅ HTTP Reachability & Redirects
  ✅ HTML Title, Body & Keyword Matches
  ✅ Semantic Similarity with CSE Names (SentenceTransformer)
  ✅ Favicon Hash Matching (Visual Clone Detection)
  ✅ RDAP Info (Registrar, Org, Domain Age)
  ✅ Lexical Cues (keyword overlap, suspicious patterns)

Input:
    prefilter_scored.csv (with Domain & CSE_Score)
Output:
    CSE_Relevance_Output_v3.csv
"""

import os
import re
import io
import json
import math
import socket
import asyncio
import aiohttp
import pandas as pd
import requests
import tldextract
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
import imagehash
from urllib.parse import urljoin

# Optional semantic model
from sentence_transformers import SentenceTransformer, util

# ---------- CONFIG ----------
OUT_FILE = "CSE_Relevance_Output.csv"
CACHE_FILE = Path("cache/cse_relevance_cache_v3.json")
CONCURRENCY = 60
TIMEOUT = aiohttp.ClientTimeout(total=15, connect=6, sock_read=8)

CSE_DEFS = {
    "SBI": {"official_domains": ["onlinesbi.sbi", "sbi.co.in", "sbicard.com"], "keywords": ["sbi", "onlinesbi", "sbicard"]},
    "ICICI": {"official_domains": ["icicibank.com", "icicidirect.com"], "keywords": ["icici", "icicibank"]},
    "HDFC": {"official_domains": ["hdfcbank.com", "hdfclife.com"], "keywords": ["hdfc", "hdfcbank"]},
    "PNB": {"official_domains": ["pnbindia.in", "netpnb.com"], "keywords": ["pnb", "pnbindia"]},
    "BoB": {"official_domains": ["bankofbaroda.in", "bankofbaroda.com"], "keywords": ["bankofbaroda", "bob"]},
    "NIC": {"official_domains": ["nic.in", "gov.in"], "keywords": ["nic", "gov"]},
    "IRCTC": {"official_domains": ["irctc.co.in"], "keywords": ["irctc"]},
    "Airtel": {"official_domains": ["airtel.in"], "keywords": ["airtel"]},
    "IOCL": {"official_domains": ["iocl.com", "indianoil.in"], "keywords": ["iocl", "indianoil"]}
}

SEM_MODEL = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
CSE_NAMES = list(CSE_DEFS.keys())
CSE_EMBEDS = SEM_MODEL.encode(CSE_NAMES, convert_to_tensor=True, normalize_embeddings=True)

# ---------- HELPERS ----------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def safe_host(domain):
    d = str(domain).lower().strip()
    d = re.sub(r"^https?://", "", d)
    return d.split("/")[0]

def levenshtein_ratio(a, b):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()

async def fetch_html(session, domain):
    """Fetch HTML content for domain"""
    for scheme in ("https", "http"):
        url = f"{scheme}://{domain}"
        try:
            async with session.get(url, timeout=TIMEOUT, allow_redirects=True) as r:
                if r.status != 200:
                    continue
                html = await r.text(errors="ignore")
                return html, str(r.url), r.status
        except Exception:
            continue
    return "", "", 0

async def fetch_favicon(session, domain):
    """Fetch and hash favicon"""
    try:
        url = f"https://{domain}/favicon.ico"
        async with session.get(url, timeout=TIMEOUT) as r:
            if r.status == 200:
                img = Image.open(io.BytesIO(await r.read())).convert("RGB")
                return str(imagehash.phash(img))
    except Exception:
        pass
    return None

def semantic_similarity(text):
    """Compute semantic similarity between text and known CSE names"""
    try:
        text_emb = SEM_MODEL.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(text_emb, CSE_EMBEDS)[0]
        best_idx = int(sims.argmax())
        return CSE_NAMES[best_idx], float(sims[best_idx])
    except Exception:
        return None, 0.0

def rdap_lookup(domain):
    """Fetch WHOIS/RDAP info"""
    result = {"org": "", "creation_date": "", "registrar": ""}
    try:
        r = requests.get(f"https://rdap.org/domain/{domain}", timeout=6)
        if r.status_code != 200:
            return result
        data = r.json()
        for e in data.get("events", []):
            if e.get("eventAction") == "registration":
                result["creation_date"] = e.get("eventDate")
        for ent in data.get("entities", []):
            if "registrar" in (ent.get("roles") or []):
                result["registrar"] = ent.get("handle", "")
            if "registrant" in (ent.get("roles") or []):
                for row in ent.get("vcardArray", [[], []])[1]:
                    if row[0] == "org":
                        result["org"] = row[3]
        return result
    except Exception:
        return result

def build_official_fav_lib():
    """Compute and cache official favicons"""
    lib = {}
    for cse, meta in CSE_DEFS.items():
        lib[cse] = []
        for d in meta["official_domains"]:
            try:
                r = requests.get(f"https://{d}/favicon.ico", timeout=5)
                if r.status_code == 200:
                    ph = str(imagehash.phash(Image.open(io.BytesIO(r.content)).convert("RGB")))
                    lib[cse].append(ph)
            except Exception:
                continue
    return lib

# ---------- ANALYZE ONE ----------
async def analyze_domain(session, domain, fav_lib):
    html, final_url, status = await fetch_html(session, domain)
    if not html:
        return {"Domain": domain, "Status": "inactive", "Relation_Score": 0}

    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    text = " ".join([title] + [t.get_text(" ", strip=True) for t in soup.find_all(["h1","h2","p"])])
    text_low = text.lower()

    # --- CSE semantic similarity ---
    sem_cse, sem_score = semantic_similarity(title + " " + text_low)

    # --- Keyword match ---
    cse_kw_hits = sum(kw in text_low for meta in CSE_DEFS.values() for kw in meta["keywords"])
    phish_kw_hits = sum(k in text_low for k in ["login","verify","otp","password","account","update","kyc"])

    # --- Favicon similarity ---
    fav_hash = await fetch_favicon(session, domain)
    fav_best, fav_dist = None, 999
    if fav_hash:
        for cse, hashes in fav_lib.items():
            for h in hashes:
                try:
                    dist = imagehash.hex_to_hash(fav_hash) - imagehash.hex_to_hash(h)
                    if dist < fav_dist:
                        fav_dist, fav_best = dist, cse
                except Exception:
                    continue

    # --- RDAP info ---
    rdap = rdap_lookup(domain)
    age_days = 0
    if rdap.get("creation_date"):
        try:
            dt = datetime.fromisoformat(rdap["creation_date"].replace("Z",""))
            age_days = (datetime.now() - dt).days
        except Exception:
            pass

    # --- Weighted scoring ---
    score = 0
    if sem_score > 0.65: score += sem_score * 1.5
    if cse_kw_hits > 0: score += 0.5
    if fav_best and fav_dist <= 6: score += 1.0
    if "gov.in" in domain or rdap.get("org"): score += 0.5
    if age_days < 180: score += 0.3

    status_label = "related" if score >= 1.5 else "uncertain" if score >= 0.7 else "unrelated"

    return {
        "Domain": domain,
        "Final_URL": final_url,
        "HTTP_Status": status,
        "Title": title,
        "Content_CSE_Keyword_Count": cse_kw_hits,
        "Content_Phish_Keyword_Count": phish_kw_hits,
        "Semantic_CSE": sem_cse,
        "Semantic_Score": round(sem_score, 3),
        "Favicon_Match": fav_best or "",
        "Favicon_Distance": fav_dist if fav_dist != 999 else "",
        "RDAP_Org": rdap.get("org", ""),
        "RDAP_Registrar": rdap.get("registrar", ""),
        "Domain_Age_Days": age_days,
        "Relation_Score": round(score, 3),
        "Status": status_label,
    }

# ---------- MAIN ----------
async def main(input_csv):
    df = pd.read_csv(input_csv)
    if "Domain" not in df.columns:
        raise ValueError("Input CSV must have 'Domain' column")

    domains = df["Domain"].astype(str).dropna().unique().tolist()
    print(f"🧩 Loaded {len(domains)} domains")

    fav_lib = build_official_fav_lib()
    print("🧠 Loaded official favicon library")

    results = []
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(connector=connector, timeout=TIMEOUT) as session:
        tasks = [asyncio.create_task(analyze_domain(session, d, fav_lib)) for d in domains]
        for f in asyncio.as_completed(tasks):
            r = await f
            results.append(r)
            if len(results) % 200 == 0:
                pd.DataFrame(results).to_csv(OUT_FILE, mode="a", header=not Path(OUT_FILE).exists(), index=False)
                print(f"💾 checkpoint {len(results)}/{len(domains)}")

    pd.DataFrame(results).to_csv(OUT_FILE, index=False)
    print(f"✅ Done. Results written to {OUT_FILE}")

# ---------- ENTRY ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV (from prefilter)")
    args = ap.parse_args()
    asyncio.run(main(args.input))
