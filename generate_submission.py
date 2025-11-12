#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_submission.py
----------------------
Reads prefilter_for_heavy.csv (Domain, CSE_Match), extracts features using
user feature_extractor (or fallback lexical), applies trained phishing model,
collects evidence (RDAP, hosting IPs, favicon, screenshots), and generates
submission CSV + Excel per problem statement.

Usage:
  python generate_submission.py --input prefilter_for_heavy.csv \
      --model_dir outputs --out submission_results.csv \
      --application_id AIGR-000001 --screenshots
"""

import os, sys, json, time, socket, hashlib, argparse, joblib, requests
import pandas as pd, numpy as np
from pathlib import Path
from urllib.parse import urljoin
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try selenium (optional)
USE_SELENIUM = False
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    USE_SELENIUM = True
except Exception:
    USE_SELENIUM = False

# Try user feature extractor
USE_EXTERNAL_FE = False
fe_mod = None
try:
    import model.feature_extractor as fe_mod_try
    fe_mod = fe_mod_try
    if hasattr(fe_mod, "extract_features"):
        USE_EXTERNAL_FE = True
except Exception:
    USE_EXTERNAL_FE = False

# ---------------- Config ----------------
DEFAULT_INPUT = "prefilter_for_heavy.csv"
DEFAULT_MODEL_DIR = "outputs"
DEFAULT_SCALER = "scaler.pkl"
OUTPUT_CSV = "submission_results.csv"
OUTPUT_XLSX_TEMPLATE = "PS-02_{app}_Submission_Set.xlsx"
CACHE_FILE = "submission_cache.json"
SCREENSHOT_DIR = "evidences"
SCREENSHOT_TIMEOUT = 12

Path(SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)

# ---------------- Helpers ----------------
import re, math
from collections import Counter

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    prob = [freq / len(s) for freq in Counter(s).values()]
    return -sum(p * math.log2(p) for p in prob)

def fallback_extract_features(url: str) -> dict:
    """Fallback lexical-only extractor"""
    u = str(url).strip()
    if not u: return {}
    if not re.match(r"^https?://", u, flags=re.I):
        u = "http://" + u
    from urllib.parse import urlparse
    parsed = urlparse(u)
    host = parsed.netloc.lower()
    path = parsed.path or ""
    parts = host.split(".")
    domain = parts[-2] + "." + parts[-1] if len(parts) >= 2 else host
    subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""
    feats = {
        "url_length": len(u),
        "num_dots": u.count("."),
        "num_hyphens": u.count("-"),
        "num_slashes": u.count("/"),
        "num_digits": sum(c.isdigit() for c in u),
        "digit_ratio": sum(c.isdigit() for c in u)/(len(u)+1e-5),
        "entropy_url": shannon_entropy(u),
        "entropy_domain": shannon_entropy(domain),
        "subdomain_count": len(subdomain.split(".")) if subdomain else 0,
        "domain_length": len(domain),
        "https": 1 if parsed.scheme.lower() == "https" else 0,
    }
    return feats

def safe_encode_feature(val):
    if val is None: return 0.0
    try: return float(val)
    except Exception:
        s = str(val).strip()
        if s == "" or s.lower() in ("nan", "none", "null"): return 0.0
        try: return float(s)
        except Exception: return float(abs(hash(s)) % 1000) / 1000.0

def load_best_model(model_dir):
    mdl_dir = Path(model_dir)
    if not mdl_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    candidates = sorted([p for p in mdl_dir.iterdir() if p.name.endswith("_balanced.pkl")])
    if not candidates:
        candidates = sorted([p for p in mdl_dir.iterdir() if p.suffix == ".pkl"])
    if not candidates:
        raise FileNotFoundError("No model found in outputs/")
    model_path = candidates[-1]
    print("✅ Loading model:", model_path)
    model = joblib.load(str(model_path))
    scaler_path = mdl_dir / DEFAULT_SCALER
    scaler = joblib.load(str(scaler_path)) if scaler_path.exists() else None
    return model, scaler, model_path.name

def safe_filename_for_url(url):
    return os.path.join(SCREENSHOT_DIR, hashlib.md5(url.encode()).hexdigest() + ".png")

def capture_screenshot(url, out_path):
    if not USE_SELENIUM: return False
    try:
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1366,900")
        driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(SCREENSHOT_TIMEOUT)
        driver.get(url)
        time.sleep(2)
        driver.save_screenshot(out_path)
        driver.quit()
        return True
    except Exception:
        try: driver.quit()
        except Exception: pass
        return False

def rdap_info(domain):
    out = {"domain_creation_date": "", "rdap_registrar": "", "rdap_org": ""}
    try:
        r = requests.get(f"https://rdap.org/domain/{domain}", timeout=8)
        if r.status_code != 200: return out
        data = r.json()
        for e in data.get("events", []):
            if e.get("eventAction") in ("registration","registered"):
                out["domain_creation_date"] = e.get("eventDate")
        for ent in data.get("entities", []):
            roles = ent.get("roles", [])
            if "registrar" in roles and not out["rdap_registrar"]:
                out["rdap_registrar"] = ent.get("handle","")
            if any(r in roles for r in ("registrant","technical","administrative")) and not out["rdap_org"]:
                vca = ent.get("vcardArray", [])
                if isinstance(vca, list) and len(vca) == 2:
                    for row in vca[1]:
                        if row[0] == "org": out["rdap_org"] = row[3]
    except Exception:
        pass
    return out

def hosting_info(domain):
    out = {"ips": ""}
    try:
        out["ips"] = ",".join(socket.gethostbyname_ex(domain)[2])
    except Exception: out["ips"] = ""
    return out

# ---------------- Core ----------------
def process_domain(domain, model_feature_names, use_external_fe):
    if use_external_fe and fe_mod:
        try:
            feats = fe_mod.extract_features(domain) or {}
        except Exception:
            feats = fallback_extract_features(domain)
    else:
        feats = fallback_extract_features(domain)
    fv = [safe_encode_feature(feats.get(f, 0)) for f in model_feature_names]
    return np.array(fv).reshape(1, -1), feats

def main(args):
    # Read input
    df = pd.read_csv(args.input)
    if "Domain" not in df.columns or "CSE_Match" not in df.columns:
        raise ValueError("Input must have columns 'Domain' and 'CSE_Match'")
    df = df[df["CSE_Match"].notna() & (df["CSE_Match"].astype(str).str.strip() != "")]
    domains = df["Domain"].dropna().astype(str).str.strip().str.lower().unique().tolist()
    print(f"Loaded {len(domains)} domains with valid CSE_Match.")

    # Load model
    model, scaler, model_name = load_best_model(args.model_dir)
    if hasattr(model, "feature_names_in_"):
        model_feature_names = list(model.feature_names_in_)
    else:
        model_feature_names = sorted(fallback_extract_features(domains[0]).keys())

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(process_domain, d, model_feature_names, USE_EXTERNAL_FE): d for d in domains}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                X, feats = fut.result()
                if scaler is not None:
                    X = scaler.transform(X)
                prob = model.predict_proba(X)[:, 1][0] if hasattr(model, "predict_proba") else model.predict(X)[0]
                label = int(prob >= args.threshold)
            except Exception as e:
                print(f"⚠️ Prediction failed for {d}: {e}")
                continue

            row = {
                "Domain": d,
                "Predicted_Label": label,
                "Phishing_Score": round(float(prob), 5),
                "Model": model_name,
                "Extracted_Features": json.dumps(feats),
            }

            if label == 1:  # phishing
                row.update(rdap_info(d))
                row.update(hosting_info(d))
                try:
                    url_try = d if d.startswith("http") else f"http://{d}"
                    r = requests.get(url_try, timeout=8, headers={"User-Agent": args.user_agent})
                    row["http_status"] = r.status_code
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(r.text, "html.parser")
                    row["page_title"] = soup.title.string if soup.title else ""
                    row["final_url"] = r.url
                    fav_url = urljoin(r.url, "/favicon.ico")
                    row["favicon_url"] = fav_url
                except Exception:
                    row.update({"http_status": "", "page_title": "", "final_url": "", "favicon_url": ""})

                screenshot_path = ""
                if args.screenshots and USE_SELENIUM:
                    screenshot_path = safe_filename_for_url(d)
                    capture_screenshot(f"http://{d}", screenshot_path)
                row["screenshot"] = screenshot_path

            results.append(row)

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.out, index=False)
    print(f"✅ Saved results -> {args.out}")

    # Create submission Excel
    submission = []
    for r in df_out.to_dict(orient="records"):
        submission.append({
            "Application_ID": args.application_id,
            "Source_of_detection": args.input,
            "Identified_Domain": r["Domain"],
            "Corresponding_CSE_Domain": df[df["Domain"] == r["Domain"]]["CSE_Match"].values[0] if r["Domain"] in df["Domain"].values else "",
            "Critical_Sector_Entity_Name": "",
            "Phishing_Suspected": "Phishing" if r["Predicted_Label"] == 1 else "Benign",
            "Domain_Registration_Date": r.get("domain_creation_date", ""),
            "Registrar_Name": r.get("rdap_registrar", ""),
            "Registrant_Org": r.get("rdap_org", ""),
            "Hosting_IP": r.get("ips", ""),
            "Evidence_File": r.get("screenshot", ""),
            "Date_of_detection": datetime.now().strftime("%d-%m-%Y"),
            "Time_of_detection": datetime.now().strftime("%H-%M-%S"),
            "Remarks": ""
        })
    out_xlsx = OUTPUT_XLSX_TEMPLATE.format(app=args.application_id)
    pd.DataFrame(submission).to_excel(out_xlsx, index=False)
    print(f"✅ Submission Excel -> {out_xlsx}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate phishing detection submission & evidence")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV (Domain, CSE_Match)")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Directory with trained models")
    parser.add_argument("--out", default=OUTPUT_CSV, help="Output results CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Phishing score threshold")
    parser.add_argument("--max_workers", type=int, default=8, help="Parallel threads")
    parser.add_argument("--application_id", default="AIGR-000000", help="Application ID for submission Excel")
    parser.add_argument("--screenshots", action="store_true", help="Take screenshots for phishing (requires Selenium)")
    parser.add_argument("--user_agent", default="Mozilla/5.0", help="User-Agent header for requests")
    args = parser.parse_args()
    main(args)
