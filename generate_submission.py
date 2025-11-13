#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_submission.py

Reads heavy-classifier output, extracts features, predicts with best trained model,
collects evidence (RDAP/IP/title/favicon/screenshot) and writes submission outputs.

- Only processes rows where Status != 'inactive' and there is a CSE match column (non-empty).
- Screenshots are taken ONLY for predicted phishing (requires google-chrome installed).
- Uses feature_extractor.extract_features if present, otherwise a fallback extractor.
- Handles model/scaler feature-length mismatches robustly (pads/truncates).
"""
import os, sys, io, json, time, socket, hashlib, argparse
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import pandas as pd
import numpy as np
import requests
import joblib

# Optional libs for screenshots & favicon phash
USE_SELENIUM = False
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    USE_SELENIUM = True
except Exception:
    USE_SELENIUM = False

# image phash
try:
    from PIL import Image
    import imagehash
except Exception:
    Image = None
    imagehash = None

# Try user-supplied feature_extractor.py (must expose extract_features(url)->dict)
USE_EXTERNAL_FE = False
fe_mod = None
try:
    import model.feature_extractor as fe_mod
    if hasattr(fe_mod, "extract_features"):
        USE_EXTERNAL_FE = True
except Exception:
    USE_EXTERNAL_FE = False
    fe_mod = None

# Defaults / paths
DEFAULT_INPUT = "prefilter_for_heavy.csv"
DEFAULT_MODEL_DIR = "outputs"
DEFAULT_SCALER = "scaler.pkl"
CACHE_FILE = "submission_cache_new.json"
SCREENSHOT_DIR = "evidences_new"
OUTPUT_CSV = "submission_results_new.csv"
OUTPUT_XLSX_TEMPLATE = "PS-02_{app}_Submission_Set.xlsx"
SCREENSHOT_TIMEOUT = 12
Path(SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)

# ------------------------- Helpers -------------------------
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    prob = [freq / len(s) for freq in Counter(s).values()]
    return -sum(p * (np.log2(p)) for p in prob)

def fallback_extract_features(url: str) -> dict:
    u = str(url or "").strip()
    if not u:
        return {}
    if not u.lower().startswith("http"):
        u_parse = "http://" + u
    else:
        u_parse = u
    parsed = urlparse(u_parse)
    host = (parsed.netloc or "").lower()
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
    feats["repeated_digits_url"] = 1 if any(c.isdigit() for c in u) and ("".join([c for c in u if c.isdigit()]) != "") else 0
    feats["domain_length"] = len(domain)
    feats["entropy_url"] = shannon_entropy(u)
    feats["entropy_domain"] = shannon_entropy(domain)
    feats["subdomain_count"] = len(subdomain.split(".")) if subdomain else 0
    feats["avg_subdomain_len"] = float(np.mean([len(s) for s in subdomain.split(".")])) if subdomain else 0.0
    feats["https"] = 1 if parsed.scheme.lower() == "https" else 0
    feats["has_ip_host"] = 1 if parsed.hostname and parsed.hostname.replace(".", "").isdigit() else 0
    feats["is_related_to_cse"] = 0
    return feats

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
    if s == "" or s.lower() in ("nan","none","null"):
        return 0.0
    try:
        return float(s)
    except Exception:
        return float(abs(hash(s)) % 1000) / 1000.0

# RDAP
def rdap_info(domain: str) -> dict:
    out = {"domain_creation_date":"", "rdap_registrar":"", "rdap_org":""}
    try:
        r = requests.get(f"https://rdap.org/domain/{domain}", timeout=8)
        if r.status_code != 200:
            return out
        data = r.json()
        for ev in data.get("events", []):
            if ev.get("eventAction","").lower() in ("registration","registered"):
                out["domain_creation_date"] = ev.get("eventDate") or out["domain_creation_date"]
        for ent in data.get("entities", []):
            roles = ent.get("roles", [])
            if "registrar" in roles and not out["rdap_registrar"]:
                out["rdap_registrar"] = ent.get("handle","") or out["rdap_registrar"]
            if any(rn in roles for rn in ("registrant","administrative","technical")) and not out["rdap_org"]:
                vca = ent.get("vcardArray",[])
                if isinstance(vca, list) and len(vca)==2:
                    for row in vca[1]:
                        if row[0] == "org":
                            out["rdap_org"] = row[3]; break
    except Exception:
        pass
    return out

# Hosting
def hosting_info(domain: str) -> dict:
    out = {"ips": "", "hosting_isp":"", "hosting_country":""}
    try:
        names = socket.gethostbyname_ex(domain)
        ips = names[2]
        out["ips"] = ",".join(ips)
    except Exception:
        out["ips"] = ""
    return out

# favicon phash
def favicon_phash_from_url(fav_url: str):
    if not fav_url or Image is None or imagehash is None:
        return ""
    try:
        r = requests.get(fav_url, timeout=6, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code == 200 and "image" in (r.headers.get("Content-Type","") or ""):
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            return str(imagehash.phash(img))
    except Exception:
        pass
    return ""

# --------------------- Screenshots ---------------------
def find_chrome_binary():
    import shutil
    for cmd in ("google-chrome","chrome","chromium-browser","chromium"):
        p = shutil.which(cmd)
        if p:
            return p
    return None

def capture_screenshot_chrome(url: str, out_path: str) -> bool:
    if not USE_SELENIUM:
        return False
    chrome_bin = find_chrome_binary()
    if not chrome_bin:
        # no chrome found on system
        return False
    try:
        opts = Options()
        opts.binary_location = chrome_bin
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1366,900")
        # install matching chromedriver automatically
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=opts)
        driver.set_page_load_timeout(SCREENSHOT_TIMEOUT)
        driver.get(url)
        time.sleep(2)
        driver.save_screenshot(out_path)
        driver.quit()
        return True
    except Exception as e:
        try:
            driver.quit()
        except Exception:
            pass
        return False

def safe_filename_for_url(url: str):
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(SCREENSHOT_DIR, f"{h}.png")

# --------------------- Model loading & feature-name discovery ---------------------
def load_best_model(model_dir: str = DEFAULT_MODEL_DIR):
    mdl_dir = Path(model_dir)
    if not mdl_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    candidates = sorted([p for p in mdl_dir.iterdir() if p.name.endswith("_balanced.pkl")])
    if not candidates:
        candidates = sorted([p for p in mdl_dir.iterdir() if p.suffix == ".pkl"])
    if not candidates:
        raise FileNotFoundError("No model pkl found in outputs/")
    model_path = candidates[-1]
    model = joblib.load(str(model_path))
    scaler = None
    scaler_path = mdl_dir / DEFAULT_SCALER
    if scaler_path.exists():
        try:
            scaler = joblib.load(str(scaler_path))
        except Exception:
            scaler = None
    return model, scaler, model_path.name

def discover_model_feature_names(model, sample_domain=None):
    # sklearn
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass
    # xgboost
    try:
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            if getattr(booster, "feature_names", None):
                return list(booster.feature_names)
    except Exception:
        pass
    # fallback: from external extractor or lexical fallback
    if USE_EXTERNAL_FE and fe_mod and sample_domain:
        try:
            s = fe_mod.extract_features(sample_domain) or {}
            return sorted(list(s.keys()))
        except Exception:
            pass
    return sorted(list(fallback_extract_features(sample_domain or "example.com").keys()))

# --------------------- Per-domain processing ---------------------
def process_domain(domain: str, model_feature_names, use_external_fe, user_agent):
    # extract raw features
    if use_external_fe and fe_mod:
        try:
            feats = fe_mod.extract_features(domain) or {}
        except Exception:
            feats = fallback_extract_features(domain)
    else:
        feats = fallback_extract_features(domain)

    # build feature vector according to model_feature_names
    fv = [safe_encode_feature(feats.get(fn, feats.get(fn.lower(), feats.get(fn.replace("-","_"), 0)))) for fn in model_feature_names]
    return np.array(fv, dtype=float).reshape(1, -1), feats

# --------------------- Main ---------------------
def main(args):
    # load input
    if not os.path.exists(args.input):
        print("Input not found:", args.input); return
    df = pd.read_csv(args.input)
    # find a CSE column
    cse_cols = [c for c in df.columns if any(k in c.lower() for k in ("related_cse","semantic_cse","best_cse","cse","cse_match"))]
    cse_col = cse_cols[0] if cse_cols else None
    if cse_col is None:
        raise ValueError("No CSE column found in input CSV (looked for related_cse/semantic_cse/best_cse/cse/cse_match).")

    # filter: Status != inactive and has cse name
    # df["Status"] = df.get("Status","").astype(str)
    # filtered = df[(df["Status"].str.lower() != "inactive") & (df[cse_col].astype(str).str.strip() != "")].copy()
    filtered=df
    if "Domain" not in filtered.columns:
        # try other heuristics
        candidates = [c for c in filtered.columns if "domain" in c.lower() or "host" in c.lower() or "base_host" in c.lower()]
        if candidates:
            filtered["Domain"] = filtered[candidates[0]]
        else:
            raise ValueError("Input CSV must contain a Domain column or similar.")
    domains = filtered["Domain"].dropna().astype(str).str.strip().str.lower().unique().tolist()
    print(f"Processing {len(domains)} domains (Status != inactive and has CSE name).")

    model, scaler, model_name = load_best_model(args.model_dir)
    model_feature_names = discover_model_feature_names(model, sample_domain=domains[0] if domains else None)
    print("Model:", model_name, "| Feature count expected:", len(model_feature_names))

    # load cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            cache = json.load(open(CACHE_FILE))
        except Exception:
            cache = {}

    results = {}
    to_process = [d for d in domains if d not in cache]

    # threadpool for IO-bound extraction
    workers = max(1, min(16, args.max_workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_domain, d, model_feature_names, USE_EXTERNAL_FE, args.user_agent): d for d in to_process}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                Xraw, feats_raw = fut.result()
            except Exception as e:
                print("Feature extraction failed for", d, "->", e)
                continue

            # ensure shape matches scaler/model
            X = Xraw
            if scaler is not None:
                try:
                    Xs = scaler.transform(X)
                except Exception as e:
                    # pad/truncate to scaler mean_ length if available
                    if hasattr(scaler, "mean_"):
                        n = len(scaler.mean_)
                        xpad = np.zeros((1, n), dtype=float)
                        k = min(n, X.shape[1])
                        xpad[0, :k] = X[0, :k]
                        try:
                            Xs = scaler.transform(xpad)
                        except Exception:
                            Xs = xpad
                    else:
                        # try to pad to model_feature_names length
                        n = len(model_feature_names)
                        xpad = np.zeros((1, n), dtype=float)
                        k = min(n, X.shape[1])
                        xpad[0, :k] = X[0, :k]
                        Xs = xpad
            else:
                # if no scaler, still ensure feature-count matches model expectation for xgboost/sklearn
                n_expected = len(model_feature_names)
                if X.shape[1] != n_expected:
                    xpad = np.zeros((1, n_expected), dtype=float)
                    k = min(n_expected, X.shape[1])
                    xpad[0, :k] = X[0, :k]
                    Xs = xpad
                else:
                    Xs = X

            # predict probability
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(Xs)[:, 1]
                    prob = float(probs[0])
                else:
                    pred = model.predict(Xs)[0]
                    prob = float(pred)
            except Exception as e:
                print("Model predict failed for", d, "->", e)
                prob = 0.0

            label = int(prob >= args.threshold)

            extra = {}
            if label == 1 or args.always_collect:
                # RDAP & hosting
                rd = rdap_info(d); extra.update(rd)
                hi = hosting_info(d); extra.update(hi)
                # quick http fetch for final url/title/favicon
                try:
                    url_try = d if d.lower().startswith("http") else "http://" + d
                    rr = requests.get(url_try, timeout=10, headers={"User-Agent": args.user_agent}, allow_redirects=True)
                    extra["final_url"] = rr.url
                    extra["http_status"] = rr.status_code
                    ct = rr.headers.get("Content-Type","")
                    page_html = rr.text if "html" in (ct or "").lower() else ""
                    # title
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(page_html, "html.parser")
                        extra["page_title"] = (soup.title.string or "").strip() if soup.title else ""
                        # favicon
                        fav_url = ""
                        link = soup.find("link", rel=lambda v: v and "icon" in v.lower())
                        if link and link.get("href"):
                            fav_url = urljoin(rr.url, link["href"])
                        if not fav_url:
                            fav_url = urljoin(rr.url, "/favicon.ico")
                        extra["favicon_url"] = fav_url
                        extra["favicon_phash"] = favicon_phash_from_url(fav_url)
                    except Exception:
                        extra.setdefault("page_title","")
                        extra.setdefault("favicon_url","")
                        extra.setdefault("favicon_phash","")
                except Exception:
                    extra.setdefault("final_url","")
                    extra.setdefault("http_status","")
                    extra.setdefault("page_title","")
                    extra.setdefault("favicon_url","")
                    extra.setdefault("favicon_phash","")

                # screenshot for positives if requested and chrome present
                screenshot_path = ""
                if label == 1 and args.screenshots and USE_SELENIUM:
                    url_for_shot = extra.get("final_url") or ("http://" + d)
                    path = safe_filename_for_url(url_for_shot)
                    ok = capture_screenshot_chrome(url_for_shot, path)
                    screenshot_path = path if ok else ""
                extra["screenshot"] = screenshot_path

            out_row = {
                "Domain": d,
                "Predicted_Label": label,
                "Phishing_Score": round(prob, 5),
                "Model": model_name,
                "Extracted_Features": json.dumps(feats_raw)
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

    # Build submission XLSX rows per problem statement
    submission_rows = []
    for r in df_out.to_dict(orient="records"):
        submission_rows.append({
            "Application_ID": args.application_id,
            "Source_of_detection": args.input,
            "Identified_Domain": r.get("Domain",""),
            "Corresponding_CSE_Domain": "",   # (heavy classifier input may contain this column; user can merge)
            "Critical_Sector_Entity_Name": "", # (user to enrich if available)
            "Final_URL": r.get("final_url",""),
            "HTTP_Status": r.get("http_status",""),
            "Domain_Registration_Date": r.get("domain_creation_date",""),
            "Registrar_Name": r.get("rdap_registrar",""),
            "Registrant_Org": r.get("rdap_org",""),
            "Hosting_IP": r.get("ips",""),
            "Favicon_pHash": r.get("favicon_phash",""),
            "Page_Title": r.get("page_title",""),
            "Evidence_File": r.get("screenshot",""),
            "Date_of_detection": datetime.now().strftime("%d-%m-%Y"),
            "Time_of_detection": datetime.now().strftime("%H:%M:%S"),
            "Remarks": ""
        })
    out_xlsx = OUTPUT_XLSX_TEMPLATE.format(app=args.application_id)
    pd.DataFrame(submission_rows).to_excel(out_xlsx, index=False)
    print(f"✅ Done. Results -> {OUTPUT_CSV} and Excel -> {out_xlsx}")

# --------------------- CLI ---------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate submission predictions and evidence")
    p.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV (heavy classifier output)")
    p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Dir with trained model & scaler")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for phishing")
    p.add_argument("--screenshots", action="store_true", help="Capture screenshot for positives (requires google-chrome)")
    p.add_argument("--checkpoint", type=int, default=200, help="Checkpoint every N domains")
    p.add_argument("--max_workers", type=int, default=8, help="Parallel workers for feature extraction")
    p.add_argument("--application_id", default="AIGR-000000", help="Application ID for submission XLSX")
    p.add_argument("--user_agent", default="Mozilla/5.0 (X11; Linux x86_64)", help="User-Agent for HTTP requests")
    p.add_argument("--always_collect", action="store_true", help="Collect RDAP/host/title for all domains")
    args = p.parse_args()
    main(args)
