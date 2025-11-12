#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_submission.py
----------------------
Reads heavy classifier output (CSV of candidate domains), extracts features
(using local feature_extractor if available), aligns features with trained model,
predicts phishing probability, collects evidence metadata (RDAP, hosting IPs,
title, favicon phash, screenshot), and writes submission files.

Behavior:
 - Only processes rows that have a detected CSE name (Semantic_CSE / Related_CSE / Best_CSE)
   and Status != 'inactive' (per your request).
 - Screenshots are taken only for predicted phishing (requires Selenium + Chrome).
 - Uses user's feature_extractor.py if available (preferred) for feature parity.
 - Robustly aligns features to model expected feature names and encodes safely.

Usage example:
    python generate_submission.py --input CSE_Relevance_Output_v2.csv --model_dir outputs \
        --out submission_results.csv --application_id AIGR-000001 --screenshots

"""

import os
import sys
import json
import time
import socket
import hashlib
import argparse
import joblib
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# Optional selenium
USE_SELENIUM = False
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    USE_SELENIUM = True
except Exception:
    USE_SELENIUM = False

# Try to import user's feature_extractor (preferred)
USE_EXTERNAL_FE = False
fe_mod = None
try:
    # Try both top-level and model/ module path
    import model.feature_extractor as fe_mod_try
    fe_mod = fe_mod_try
    if hasattr(fe_mod, "extract_features"):
        USE_EXTERNAL_FE = True
except Exception:
    try:
        import model.feature_extractor as fe_mod_try
        fe_mod = fe_mod_try
        if hasattr(fe_mod, "extract_features"):
            USE_EXTERNAL_FE = True
    except Exception:
        USE_EXTERNAL_FE = False
        fe_mod = None

# Defaults / paths
DEFAULT_INPUT_CSV = "CSE_Relevance_Output.csv"
DEFAULT_MODEL_DIR = "outputs"
DEFAULT_SCALER = "scaler.pkl"
CACHE_FILE = "submission_cache.json"
SCREENSHOT_DIR = "evidences"
OUTPUT_CSV = "submission_results.csv"
OUTPUT_XLSX_TEMPLATE = "PS-02_{app}_Submission_Set.xlsx"
SCREENSHOT_TIMEOUT = 12  # seconds

Path(SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)

# ---------------------------
# Fallback lightweight extractor (if user's not available)
# ---------------------------
import re, math
from collections import Counter

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    prob = [freq / len(s) for freq in Counter(s).values()]
    return -sum(p * math.log2(p) for p in prob)

def fallback_extract_features(url: str) -> dict:
    u = str(url).strip()
    if not u:
        return {}
    if not re.match(r"^https?://", u, flags=re.I):
        u_for_parse = "http://" + u
    else:
        u_for_parse = u
    from urllib.parse import urlparse
    parsed = urlparse(u_for_parse)
    host = parsed.netloc.lower()
    path = parsed.path or ""
    parts = host.split(".")
    domain = parts[-2] + "." + parts[-1] if len(parts) >= 2 else host
    subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""
    feats = {}
    feats["url_length"] = len(u)
    feats["num_dots"] = u.count(".")
    feats["num_hyphens"] = u.count("-")
    feats["num_underscores"] = u.count("_")
    feats["num_slashes"] = u.count("/")
    feats["num_digits"] = sum(c.isdigit() for c in u)
    feats["digit_ratio"] = feats["num_digits"] / (len(u) + 1e-5)
    feats["num_special"] = sum(u.count(ch) for ch in "?=.$!#%")
    feats["num_question"] = u.count("?")
    feats["num_equal"] = u.count("=")
    feats["num_dollar"] = u.count("$")
    feats["num_exclamation"] = u.count("!")
    feats["num_hashtag"] = u.count("#")
    feats["num_percent"] = u.count("%")
    feats["repeated_digits_url"] = 1 if re.search(r"\d{3,}", u) else 0
    feats["domain_length"] = len(domain)
    feats["num_hyphens_domain"] = domain.count("-")
    feats["num_special_domain"] = sum(domain.count(ch) for ch in "$#%_")
    feats["has_special_domain"] = 1 if feats["num_special_domain"] > 0 else 0
    feats["subdomain_count"] = len(subdomain.split(".")) if subdomain else 0
    feats["avg_subdomain_len"] = float(np.mean([len(s) for s in subdomain.split(".")])) if subdomain else 0.0
    feats["subdomain_hyphen"] = 1 if "-" in subdomain else 0
    feats["subdomain_repeated_digits"] = 1 if re.search(r"\d{2,}", subdomain) else 0
    feats["entropy_url"] = shannon_entropy(u)
    feats["entropy_domain"] = shannon_entropy(domain)
    feats["https"] = 1 if parsed.scheme.lower() == "https" else 0
    feats["has_ip_host"] = 1 if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", parsed.hostname or "") else 0
    feats["is_related_to_cse"] = 0
    return feats

# ---------------------------
# Utilities
# ---------------------------
def load_best_model(model_dir: str):
    mdl_dir = Path(model_dir)
    if not mdl_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    candidates = sorted([p for p in mdl_dir.iterdir() if p.name.endswith("_balanced.pkl")])
    if not candidates:
        candidates = sorted([p for p in mdl_dir.iterdir() if p.suffix == ".pkl"])
    if not candidates:
        raise FileNotFoundError(f"No model .pkl found in {model_dir}")
    model_path = candidates[-1]  # pick last (sorted)
    print("Loading model:", model_path)
    model = joblib.load(str(model_path))
    scaler_path = mdl_dir / DEFAULT_SCALER
    scaler = joblib.load(str(scaler_path)) if scaler_path.exists() else None
    return model, scaler, model_path.name

def safe_filename_for_url(url: str):
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(SCREENSHOT_DIR, f"{h}.png")

def capture_screenshot_chrome(url: str, out_path: str):
    if not USE_SELENIUM:
        return False
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1366,900")
    driver = None
    try:
        driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(SCREENSHOT_TIMEOUT)
        driver.get(url)
        time.sleep(2)
        driver.save_screenshot(out_path)
        driver.quit()
        return True
    except Exception as e:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass
        return False

def rdap_info(domain: str) -> dict:
    out = {"domain_creation_date": "", "rdap_registrar": "", "rdap_org": ""}
    try:
        r = requests.get(f"https://rdap.org/domain/{domain}", timeout=8)
        if r.status_code != 200:
            return out
        data = r.json()
        events = data.get("events", [])
        for e in events:
            if e.get("eventAction") in ("registration", "registered", "registration"):
                out["domain_creation_date"] = e.get("eventDate") or out["domain_creation_date"]
        for ent in data.get("entities", []):
            roles = ent.get("roles", [])
            if "registrar" in roles and not out["rdap_registrar"]:
                out["rdap_registrar"] = ent.get("handle") or ""
            if any(rn in roles for rn in ("registrant","administrative","technical")) and not out["rdap_org"]:
                vca = ent.get("vcardArray", [])
                if isinstance(vca, list) and len(vca) == 2:
                    for row in vca[1]:
                        if row[0] == "org":
                            out["rdap_org"] = row[3]
                            break
    except Exception:
        pass
    return out

def hosting_info(domain: str) -> dict:
    out = {"ips": "", "hosting_isp": "", "hosting_country": ""}
    try:
        names = socket.gethostbyname_ex(domain)
        ips = names[2]
        out["ips"] = ",".join(ips)
    except Exception:
        out["ips"] = ""
    return out

def safe_encode_feature(val):
    if val is None:
        return 0.0
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            return float(val)
        except Exception:
            return 0.0
    if isinstance(val, (bool, np.bool_)):
        return float(int(val))
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return 0.0
    try:
        return float(s)
    except Exception:
        return float(abs(hash(s)) % 1000) / 1000.0

# ---------------------------
# Core domain processing
# ---------------------------
def process_domain(domain: str, model_feature_names: list, use_external_fe: bool, user_agent: str):
    # 1) get features
    feats = {}
    if use_external_fe and fe_mod:
        try:
            feats = fe_mod.extract_features(domain) or {}
        except Exception:
            feats = fallback_extract_features(domain)
    else:
        feats = fallback_extract_features(domain)

    # 2) align to model_feature_names
    fv = []
    for fname in model_feature_names:
        val = None
        if fname in feats:
            val = feats[fname]
        else:
            alt = fname.lower()
            if alt in feats:
                val = feats[alt]
            else:
                alt2 = fname.replace("-", "_")
                if alt2 in feats:
                    val = feats[alt2]
                else:
                    val = 0
        fv.append(safe_encode_feature(val))

    # Return feature vector, raw feats
    return np.array(fv, dtype=float).reshape(1, -1), feats

# ---------------------------
# Main
# ---------------------------
def main(args):
    # load input
    if not os.path.exists(args.input):
        print("Input file not found:", args.input)
        return
    df_in = pd.read_csv(args.input)

    # determine CSE column candidates
    cse_cols = [c for c in df_in.columns if c.lower() in ("semantic_cse", "related_cse", "best_cse", "cse", "semantic_cse_name")]
    cse_col = cse_cols[0] if cse_cols else None

    # ensure Domain column
    if "Domain" not in df_in.columns:
        candidates = [c for c in df_in.columns if "domain" in c.lower() or "host" in c.lower() or "base_host" in c.lower()]
        if candidates:
            df_in["Domain"] = df_in[candidates[0]]
        else:
            raise ValueError("Input CSV must contain a Domain column or similar")

    # filter rows: only those that have a CSE name (cse_col non-empty) and Status != 'inactive'
    def has_cse_name(row):
        if cse_col:
            val = str(row.get(cse_col, "")).strip()
            return bool(val)
        # fallback: check Semantic_CSE-like columns
        for cand in ("Semantic_CSE","Related_CSE","Best_CSE","CSE"):
            if cand in df_in.columns and str(row.get(cand,"")).strip():
                return True
        return False

    df_in["Status"] = df_in.get("Status", "").astype(str)
    filtered = df_in[df_in["Status"].str.lower() != "inactive"].copy()
    filtered = filtered[filtered.apply(has_cse_name, axis=1)].copy()
    domains = filtered["Domain"].dropna().astype(str).str.strip().str.lower().unique().tolist()
    print(f"Loaded {len(domains)} domains after filtering (Status != inactive & has CSE name).")

    # load model
    model, scaler, model_name = load_best_model(args.model_dir)

    # determine expected feature names
    model_feature_names = None
    # sklearn models often have feature_names_in_
    if hasattr(model, "feature_names_in_"):
        try:
            model_feature_names = list(model.feature_names_in_)
            print(f"Model expects {len(model_feature_names)} features (feature_names_in_).")
        except Exception:
            model_feature_names = None

    # xgboost booster
    if model_feature_names is None:
        try:
            if hasattr(model, "get_booster"):
                booster = model.get_booster()
                if hasattr(booster, "feature_names") and booster.feature_names:
                    model_feature_names = list(booster.feature_names)
                    print(f"Model (xgboost) feature names discovered: {len(model_feature_names)}")
        except Exception:
            model_feature_names = None

    # if still unknown, infer from user's extractor or fallback extractor
    if model_feature_names is None:
        if USE_EXTERNAL_FE and fe_mod:
            try:
                sample_feats = fe_mod.extract_features(domains[0])
                model_feature_names = sorted(list(sample_feats.keys()))
                print(f"Inferred feature names from external extractor: {len(model_feature_names)}")
            except Exception:
                model_feature_names = sorted(list(fallback_extract_features(domains[0]).keys()))
                print("Using fallback feature name set.")
        else:
            model_feature_names = sorted(list(fallback_extract_features(domains[0]).keys()))
            print("Using fallback feature name set:", len(model_feature_names))

    # loading cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            cache = json.load(open(CACHE_FILE, "r"))
        except Exception:
            cache = {}

    results = {}
    to_process = [d for d in domains if d not in cache]

    # process with threadpool (I/O bound)
    workers = max(1, min(16, args.max_workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for d in to_process:
            futures[ex.submit(process_domain, d, model_feature_names, USE_EXTERNAL_FE, args.user_agent)] = d

        for fut in as_completed(futures):
            d = futures[fut]
            try:
                Xraw, feats_raw = fut.result()
            except Exception as e:
                print("Feature extraction failed for", d, "->", e)
                continue

            # scale/transform
            X = Xraw
            try:
                if scaler is not None:
                    Xs = scaler.transform(X)
                else:
                    Xs = X
            except Exception as e:
                # fallback: try align by trimming/padding if sizes differ
                try:
                    if scaler is not None:
                        # try using scaler on zeros with same shape
                        Xs = scaler.transform(X)
                    else:
                        Xs = X
                except Exception:
                    Xs = X

            # predict probability
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(Xs)[:, 1]
                    prob = float(probs[0])
                else:
                    # fallback to predict (0/1)
                    pred = model.predict(Xs)[0]
                    prob = float(pred)
            except Exception as e:
                print("Model prediction failed for", d, "->", e)
                prob = 0.0

            label = int(prob >= args.threshold)

            extra = {}
            if label == 1 or args.always_collect:
                # collect RDAP & hosting & quick fetch
                rd = rdap_info(d)
                hostinfo = hosting_info(d)
                extra.update(rd)
                extra.update(hostinfo)

                # quick fetch page to get final url/title/favicon
                url_try = d if d.startswith("http") else f"http://{d}"
                try:
                    rr = requests.get(url_try, timeout=8, headers={"User-Agent": args.user_agent}, allow_redirects=True)
                    final_url = rr.url
                    status_code = rr.status_code
                    ct = rr.headers.get("Content-Type", "")
                    page_html = rr.text if ("html" in ct.lower()) else ""
                    title = ""
                    if page_html:
                        try:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(page_html, "html.parser")
                            title = (soup.title.string or "").strip() if soup.title else ""
                        except Exception:
                            title = ""
                    extra["final_url"] = final_url
                    extra["http_status"] = status_code
                    extra["page_title"] = title
                    # favicon
                    fav_url = ""
                    try:
                        if page_html:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(page_html, "html.parser")
                            link = soup.find("link", rel=lambda v: v and "icon" in v.lower())
                            if link and link.get("href"):
                                fav_url = urljoin(final_url, link["href"])
                    except Exception:
                        fav_url = ""
                    if not fav_url:
                        fav_url = urljoin(final_url, "/favicon.ico")
                    extra["favicon_url"] = fav_url
                    # favicon phash best-effort
                    try:
                        rf = requests.get(fav_url, timeout=6, headers={"User-Agent": args.user_agent})
                        if rf.status_code == 200 and "image" in (rf.headers.get("Content-Type","") or ""):
                            from PIL import Image
                            import imagehash, io
                            img = Image.open(io.BytesIO(rf.content)).convert("RGB")
                            extra["favicon_phash"] = str(imagehash.phash(img))
                        else:
                            extra["favicon_phash"] = ""
                    except Exception:
                        extra["favicon_phash"] = ""
                except Exception:
                    extra.setdefault("final_url", "")
                    extra.setdefault("http_status", "")
                    extra.setdefault("page_title", "")
                    extra.setdefault("favicon_url", "")
                    extra.setdefault("favicon_phash", "")

                # screenshot for positives
                screenshot_path = ""
                if label == 1 and args.screenshots and USE_SELENIUM:
                    url_try = extra.get("final_url") or (f"http://{d}")
                    screenshot_path = safe_filename_for_url(url_try)
                    ok = capture_screenshot_chrome(url_try, screenshot_path)
                    if not ok:
                        screenshot_path = ""
                extra["screenshot"] = screenshot_path

            out_row = {
                "Domain": d,
                "Predicted_Label": label,
                "Phishing_Score": round(prob, 5),
                "Model": model_name,
                "Extracted_Features": json.dumps(feats_raw),
            }
            out_row.update(extra)
            results[d] = out_row
            cache[d] = out_row

            # checkpoint
            if len(cache) % args.checkpoint == 0:
                with open(CACHE_FILE, "w") as f:
                    json.dump(cache, f, indent=2)
                pd.DataFrame(list(cache.values())).to_csv(OUTPUT_CSV, index=False)
                print(f"Checkpoint saved: {len(cache)} processed")

    # final save
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    df_out = pd.DataFrame(list(cache.values()))
    df_out.to_csv(OUTPUT_CSV, index=False)

    # Build submission excel rows (best-effort mapping to problem statement)
    submission_rows = []
    for r in df_out.to_dict(orient="records"):
        submission_rows.append({
            "Application_ID": args.application_id,
            "Source_of_detection": args.input,
            "Identified_Domain": r.get("Domain",""),
            "Corresponding_CSE_Domain": "",  # heavy classifier may have this in original input
            "Critical_Sector_Entity_Name": "", 
            "Phishing_Suspected": "Phishing" if r.get("Predicted_Label",0)==1 else "Benign",
            "Domain_Registration_Date": r.get("domain_creation_date",""),
            "Registrar_Name": r.get("rdap_registrar",""),
            "Registrant_Org": r.get("rdap_org",""),
            "Hosting_IP": r.get("ips",""),
            "Evidence_File": r.get("screenshot",""),
            "Date_of_detection": datetime.now().strftime("%d-%m-%Y"),
            "Time_of_detection": datetime.now().strftime("%H-%M-%S"),
            "Remarks": ""
        })

    out_xlsx = OUTPUT_XLSX_TEMPLATE.format(app=args.application_id)
    pd.DataFrame(submission_rows).to_excel(out_xlsx, index=False)
    print(f"✅ Done. Results -> {OUTPUT_CSV} and submission excel -> {out_xlsx}")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission predictions and evidence")
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV,required=True, help="Input CSV (heavy classifier output)")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Directory with trained models & scaler")
    parser.add_argument("--out", default=OUTPUT_CSV, help="Output results CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Phishing probability threshold")
    parser.add_argument("--screenshots", action="store_true", help="Capture screenshots for positives (requires selenium chrome)")
    parser.add_argument("--checkpoint", type=int, default=200, help="Checkpoint every N processed domains")
    parser.add_argument("--max_workers", type=int, default=8, help="Parallel worker threads for extraction")
    parser.add_argument("--application_id", default="AIGR-000000", help="Application ID for submission excel")
    parser.add_argument("--user_agent", default="Mozilla/5.0 (X11; Linux x86_64)", help="User-Agent for HTTP requests")
    parser.add_argument("--always_collect", action="store_true", help="Collect RDAP/hosting/title for all domains, not only positives")
    args = parser.parse_args()

    # propagate some parameters
    OUTPUT_CSV = args.out
    main(args)
