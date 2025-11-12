#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Extractor – EXTENDED + Caching + Resume
===============================================
Extracts advanced lexical, domain, HTML content, and favicon features from
both whitelisted and phishing URL datasets (.xlsx). 

✔ Lexical + Domain-level (your full list)
✔ HTML content analysis
✔ Favicon similarity to official CSE favicons
✔ CSE keyword presence
✔ Caching (resumes from last checkpoint)
"""

import os, re, io, math, json, hashlib, requests, ipaddress
import pandas as pd
import numpy as np
from tqdm import tqdm
from urllib.parse import urlsplit, urlunsplit, quote, quote_plus
from collections import Counter
from bs4 import BeautifulSoup
from tldextract import extract as tld_extract
from PIL import Image
import imagehash

# ------------------------------
# CONFIG
# ------------------------------
CACHE_FILE = "feature_cache.json"

CSE_KEYWORDS = [
    "sbi","sbicard","sbilife","hdfc","hdfclife","hdfcergo","icici","icicibank",
    "icicidirect","pnb","bankofbaroda","nic","gov","irctc","airtel","iocl","indianoil"
]

CSE_FAVICONS = {
    "SBI": ["https://sbi.co.in/favicon.ico"],
    "HDFC": ["https://www.hdfcbank.com/favicon.ico"],
    "ICICI": ["https://www.icicibank.com/favicon.ico"],
    "PNB": ["https://www.pnbindia.in/favicon.ico"],
    "BoB": ["https://www.bankofbaroda.in/favicon.ico"],
    "NIC": ["https://www.nic.in/favicon.ico"],
    "IRCTC": ["https://www.irctc.co.in/favicon.ico"],
    "Airtel": ["https://www.airtel.in/favicon.ico"],
    "IOCL": ["https://www.iocl.com/favicon.ico"]
}

# -------------------------------------
# UTILITY HELPERS
# -------------------------------------
def safe_url(url: str) -> str:
    """Normalize URLs and filter out malformed ones."""
    try:
        if not url or not isinstance(url, str):
            return ""
        url = url.strip().replace(" ", "").replace("\n", "")
        if not url:
            return ""
        if not re.match(r"^https?://", url, re.I):
            url = "http://" + url

        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if not host:
            return ""
        if ":" in host:
            try:
                ipaddress.ip_address(host)
                if ":" in host and not host.startswith("["):
                    host = f"[{host}]"
            except ValueError:
                return ""

        netloc = host
        if parsed.port:
            netloc += f":{parsed.port}"
        path = quote(parsed.path or "", safe="/")
        query = quote_plus(parsed.query or "", safe="=&")
        fragment = quote_plus(parsed.fragment or "")
        return urlunsplit((parsed.scheme, netloc, path, query, fragment))
    except Exception:
        return ""

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    probs = [f / len(s) for f in Counter(s).values()]
    return -sum(p * math.log2(p) for p in probs)

def has_ip_address(host: str) -> bool:
    return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))

def fetch_page_html(url: str, timeout: int = 8) -> str:
    """Fetch HTML safely; skip invalid or broken URLs."""
    url = safe_url(url)
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if "text/html" in r.headers.get("Content-Type", ""):
            return r.text
    except Exception:
        return ""
    return ""

def extract_html_features(html: str) -> dict:
    """Extract HTML content-based features."""
    feats = {
        "html_length": 0, "num_links": 0, "external_links_ratio": 0,
        "num_forms": 0, "has_login_form": 0, "num_scripts": 0,
        "num_iframes": 0, "title_length": 0, "meta_keyword_match": 0
    }
    if not html:
        return feats
    soup = BeautifulSoup(html, "html.parser")
    feats["html_length"] = len(html)
    if soup.title and soup.title.string:
        feats["title_length"] = len(soup.title.string)
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    feats["num_links"] = len(links)
    external = [l for l in links if l and not l.startswith(("/", "#"))]
    feats["external_links_ratio"] = len(external) / (len(links) + 1e-5)
    forms = soup.find_all("form")
    feats["num_forms"] = len(forms)
    feats["has_login_form"] = int(any("password" in str(f).lower() for f in forms))
    feats["num_scripts"] = len(soup.find_all("script"))
    feats["num_iframes"] = len(soup.find_all("iframe"))
    feats["meta_keyword_match"] = int(any(
        kw in (m.get("content") or "").lower() for m in soup.find_all("meta") for kw in CSE_KEYWORDS
    ))
    return feats

def get_favicon_hash(url: str) -> str | None:
    """Compute perceptual hash of a page's favicon."""
    try:
        if not url.startswith("http"):
            url = "http://" + url
        domain_info = tld_extract(urlsplit(url).netloc)
        base = f"https://{domain_info.domain}.{domain_info.suffix}"
        for suffix in ["/favicon.ico", "/favicon.png"]:
            try:
                r = requests.get(base + suffix, timeout=5)
                if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                    img = Image.open(io.BytesIO(r.content))
                    return str(imagehash.phash(img.convert("RGB")))
            except Exception:
                continue
    except Exception:
        pass
    return None

def compare_with_cse_favicons(fav_hash: str, cse_lib: dict) -> tuple[str, int]:
    """Compare site favicon hash with known CSE favicons."""
    if not fav_hash:
        return "", 999
    best_cse, best_dist = "", 999
    try:
        fav_h = imagehash.hex_to_hash(fav_hash)
    except Exception:
        return "", 999
    for cse, urls in cse_lib.items():
        for u in urls:
            try:
                r = requests.get(u, timeout=5)
                img = Image.open(io.BytesIO(r.content))
                dist = fav_h - imagehash.phash(img.convert("RGB"))
                if dist < best_dist:
                    best_dist, best_cse = dist, cse
            except Exception:
                continue
    return best_cse, best_dist

# -------------------------------------
# MAIN FEATURE EXTRACTOR
# -------------------------------------
def extract_features(url: str) -> dict:
    """Return full feature dictionary for a given URL."""
    feats = {}
    safe_u = safe_url(url)
    if not safe_u:
        return {f: 0 for f in BASE_FEATURES}

    parsed = urlsplit(safe_u)
    domain_info = tld_extract(parsed.netloc)
    domain = f"{domain_info.domain}.{domain_info.suffix}" if domain_info.suffix else domain_info.domain
    subdomain = domain_info.subdomain or ""
    host = parsed.netloc.lower()

    feats.update({
        "url_length": len(safe_u),
        "num_dots": safe_u.count("."),
        "num_hyphens": safe_u.count("-"),
        "num_underscores": safe_u.count("_"),
        "num_slashes": safe_u.count("/"),
        "num_digits": sum(c.isdigit() for c in safe_u),
        "digit_ratio": sum(c.isdigit() for c in safe_u)/(len(safe_u)+1e-5),
        "num_special": sum(safe_u.count(ch) for ch in "?=.$!#%"),
        "num_question": safe_u.count("?"),
        "num_equal": safe_u.count("="),
        "num_dollar": safe_u.count("$"),
        "num_exclamation": safe_u.count("!"),
        "num_hashtag": safe_u.count("#"),
        "num_percent": safe_u.count("%"),
        "repeated_digits_url": int(bool(re.search(r"\d{3,}", safe_u))),
        "domain_length": len(domain),
        "num_hyphens_domain": domain.count("-"),
        "num_special_domain": sum(domain.count(ch) for ch in "$#%_"),
        "has_special_domain": int(sum(domain.count(ch) for ch in "$#%_") > 0),
        "subdomain_count": len(subdomain.split(".")) if subdomain else 0,
        "avg_subdomain_len": np.mean([len(s) for s in subdomain.split(".")]) if subdomain else 0,
        "subdomain_hyphen": int("-" in subdomain),
        "subdomain_repeated_digits": int(bool(re.search(r"\d{2,}", subdomain))),
        "entropy_url": shannon_entropy(safe_u),
        "entropy_domain": shannon_entropy(domain),
        "https": int(parsed.scheme.lower() == "https"),
        "has_ip_host": int(has_ip_address(host)),
        "is_related_to_cse": int(any(kw in safe_u.lower() for kw in CSE_KEYWORDS))
    })

    html = fetch_page_html(safe_u)
    feats.update(extract_html_features(html))

    fav_hash = get_favicon_hash(safe_u)
    cse, dist = compare_with_cse_favicons(fav_hash, CSE_FAVICONS)
    feats["favicon_match_cse"] = 0 if not cse else 1
    feats["favicon_distance"] = dist

    return {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in feats.items()}


BASE_FEATURES = [
    "url_length","num_dots","num_hyphens","num_underscores","num_slashes","num_digits","digit_ratio",
    "num_special","num_question","num_equal","num_dollar","num_exclamation","num_hashtag","num_percent",
    "repeated_digits_url","domain_length","num_hyphens_domain","num_special_domain","has_special_domain",
    "subdomain_count","avg_subdomain_len","subdomain_hyphen","subdomain_repeated_digits","entropy_url",
    "entropy_domain","https","has_ip_host","is_related_to_cse","html_length","num_links",
    "external_links_ratio","num_forms","has_login_form","num_scripts","num_iframes","title_length",
    "meta_keyword_match","favicon_match_cse","favicon_distance"
]

# -------------------------------------
# DATASET BUILDER
# -------------------------------------
def json_safe(obj):
    """Make objects JSON serializable."""
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_)): return bool(obj)
    if isinstance(obj, (dict, list)):
        return {k: json_safe(v) for k, v in obj.items()} if isinstance(obj, dict) else [json_safe(v) for v in obj]
    return obj

def save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(json_safe(cache), f, indent=2)
    except Exception as e:
        print(f"⚠️ Cache save failed: {e}")


def build_dataset(whitelist_xlsx, phishing_dir, output_csv):
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            cache = json.load(open(CACHE_FILE, "r"))
            print(f"🗂️ Loaded existing cache with {len(cache)} entries.")
        except Exception:
            print("⚠️ Cache load failed, starting fresh.")
            cache = {}

    rows = []

    # --- Load BOTH whitelisted files ---
    whitelist_files = [whitelist_xlsx, "data/whitelisted_new.xlsx"]
    whitelist_dfs = []
    wl_total_count = 0

    print("\n📘 Loading whitelisted data...")
    for file in whitelist_files:
        if os.path.exists(file):
            try:
                df = pd.read_excel(file)
                url_col = [c for c in df.columns if "url" in c.lower() or "domain" in c.lower()][0]
                df = df[[url_col]].dropna()
                df.rename(columns={url_col: "url"}, inplace=True)
                whitelist_dfs.append(df)
                wl_total_count += len(df)
                print(f"✅ Loaded {len(df)} URLs from {file}")
            except Exception as e:
                print(f"⚠️ Failed to load {file}: {e}")
        else:
            print(f"⚠️ File not found: {file}")

    # Combine and remove duplicates
    if whitelist_dfs:
        whitelist_df = pd.concat(whitelist_dfs, ignore_index=True).drop_duplicates(subset=["url"])
        wl_urls = whitelist_df["url"].astype(str).tolist()
        print(f"🔹 Total unique whitelisted URLs after merge: {len(wl_urls)} (from {wl_total_count} total entries)")
    else:
        print("❌ No valid whitelisted data found.")
        wl_urls = []

    # --- Process Whitelisted URLs ---
    for u in tqdm(wl_urls, desc="Whitelisted"):
        if u in cache:
            feats = cache[u]
        else:
            feats = extract_features(u)
            cache[u] = feats
        feats["label"] = 0
        rows.append(feats)

    # --- Process Phishing URLs ---
    phishing_files = [f for f in os.listdir(phishing_dir) if f.endswith(".xlsx")]
    ph_total = 0

    print("\n🕷️ Loading phishing data...")
    for f in phishing_files:
        try:
            df = pd.read_excel(os.path.join(phishing_dir, f))
            url_col = [c for c in df.columns if "url" in c.lower() or "domain" in c.lower()][0]
            urls = df[url_col].dropna().astype(str).tolist()
            ph_total += len(urls)
            print(f"✅ Loaded {len(urls)} phishing URLs from {f}")
        except Exception as e:
            print(f"⚠️ Failed to read {f}: {e}")
            continue

        for u in tqdm(urls, desc=f"Phishing - {f}"):
            if u in cache:
                feats = cache[u]
            else:
                feats = extract_features(u)
                cache[u] = feats
            feats["label"] = 1
            rows.append(feats)

            # Checkpoint every 100 processed rows
            if len(rows) % 100 == 0:
                save_cache(cache)

    # --- Save Final Dataset ---
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"\n✅ Dataset saved: {output_csv} | Total samples: {len(df_out)}")

    # --- Print Class Summary ---
    benign_count = len(df_out[df_out["label"] == 0])
    phishing_count = len(df_out[df_out["label"] == 1])

    print("\n📊 Summary:")
    print(f"  🟩 Whitelisted (benign): {benign_count}")
    print(f"  🟥 Phishing: {phishing_count}")
    print(f"  🧩 Combined Total: {len(df_out)}")
    print(f"  💾 Cache entries: {len(cache)}")
    
    save_cache(cache)




# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extended Feature Extractor with Cache")
    ap.add_argument("--whitelist", required=True, help="Path to whitelisted.xlsx")
    ap.add_argument("--phishing_dir", required=True, help="Directory containing phishing .xlsx files")
    ap.add_argument("--output", default="features_extended.csv", help="Output CSV file path")
    args = ap.parse_args()
    build_dataset(args.whitelist, args.phishing_dir, args.output)
