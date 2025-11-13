#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_submission.py
----------------------
Improved, robust submission generator for phishing detection.

✅ Reads: CSE_Relevance_Output_v2.csv or similar (must have Domain + CSE columns)
✅ Uses: trained model from /outputs
✅ Extracts: features using feature_extractor.py (if available)
✅ Predicts: phishing probability
✅ Collects: RDAP, IP, title, favicon, screenshot (only for phishing)
✅ Auto-handles: Chrome + ChromeDriver (downloads if missing)
✅ Compatible with Linux, macOS, Windows

Usage:
  python generate_submission.py --input CSE_Relevance_Output.csv \
      --model_dir outputs --application_id AIGR-000001 --screenshots
"""

import os, sys, re, io, time, json, math, socket, hashlib, argparse, joblib, requests
import pandas as pd, numpy as np
from pathlib import Path
from urllib.parse import urljoin
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# Optional imports
USE_EXTERNAL_FE, fe_mod = False, None
try:
    import feature_extractor as fe_mod
    if hasattr(fe_mod, "extract_features"):
        USE_EXTERNAL_FE = True
except Exception:
    USE_EXTERNAL_FE = False

# Paths and defaults
DEFAULT_MODEL_DIR = "outputs"
DEFAULT_SCALER = "scaler.pkl"
CACHE_FILE = "submission_cache.json"
SCREENSHOT_DIR = "evidences"
OUTPUT_CSV = "submission_results.csv"
OUTPUT_XLSX_TEMPLATE = "PS-02_{app}_Submission_Set.xlsx"
SCREENSHOT_TIMEOUT = 12

Path(SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Screenshot utilities
# ---------------------------------------------------------------------
def safe_filename_for_url(url: str):
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(SCREENSHOT_DIR, f"{h}.png")

def capture_screenshot_auto(url: str, out_path: str) -> str:
    """Fully robust Chrome screenshot handler."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        import shutil
        import subprocess

        # Detect Chrome or Chromium
        chrome_path = None
        for cmd in ["google-chrome", "chrome", "chromium-browser", "chromium"]:
            if shutil.which(cmd):
                chrome_path = shutil.which(cmd)
                break

        if not chrome_path:
            print("❌ No Chrome/Chromium found on system. Screenshot disabled.")
            return ""

        opts = Options()
        opts.binary_location = chrome_path

        # Essential flags for Linux VM
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--disable-software-rasterizer")
        opts.add_argument("--disable-extensions")
        opts.add_argument("--ignore-certificate-errors")
        opts.add_argument("--window-size=1366,900")

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=opts
        )

        driver.set_page_load_timeout(12)
        driver.get(url)
        time.sleep(2)
        driver.save_screenshot(out_path)
        driver.quit()

        print(f"📸 Screenshot saved: {out_path}")
        return out_path

    except Exception as e:
        print(f"❌ Screenshot failed for {url} → {e}")
        return ""


def take_screenshot_if_needed(domain: str, label: int, final_url: str = "") -> str:
    if label != 1:
        return ""
    url_try = final_url or (f"http://{domain}")
    out_path = safe_filename_for_url(url_try)
    result = capture_screenshot_auto(url_try, out_path)
    return result if result else "Screenshot unavailable"

# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    prob = [freq / len(s) for freq in Counter(s).values()]
    return -sum(p * math.log2(p) for p in prob)

def fallback_extract_features(url: str) -> dict:
    """Lightweight lexical fallback extractor."""
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
    feats["num_digits"] = sum(c.isdigit() for c in u)
    feats["digit_ratio"] = feats["num_digits"] / (len(u) + 1e-5)
    feats["entropy_url"] = shannon_entropy(u)
    feats["entropy_domain"] = shannon_entropy(domain)
    feats["subdomain_count"] = len(subdomain.split(".")) if subdomain else 0
    feats["has_ip_host"] = 1 if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", parsed.hostname or "") else 0
    feats["https"] = 1 if parsed.scheme.lower() == "https" else 0
    return feats

def rdap_info(domain: str) -> dict:
    out = {"domain_creation_date": "", "rdap_registrar": "", "rdap_org": ""}
    try:
        r = requests.get(f"https://rdap.org/domain/{domain}", timeout=8)
        if r.status_code != 200:
            return out
        data = r.json()
        for e in data.get("events", []):
            if e.get("eventAction") in ("registration", "registered"):
                out["domain_creation_date"] = e.get("eventDate")
        for ent in data.get("entities", []):
            roles = ent.get("roles", [])
            if "registrar" in roles:
                out["rdap_registrar"] = ent.get("handle", "")
            if any(rn in roles for rn in ("registrant","administrative","technical")):
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
    out = {"ips": ""}
    try:
        names = socket.gethostbyname_ex(domain)
        out["ips"] = ",".join(names[2])
    except Exception:
        out["ips"] = ""
    return out

def safe_encode_feature(val):
    if val is None:
        return 0.0
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    if isinstance(val, (bool, np.bool_)):
        return float(int(val))
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return 0.0
    try:
        return float(s)
    except Exception:
        return float(abs(hash(s)) % 1000) / 1000.0

def load_best_model(model_dir: str):
    mdl_dir = Path(model_dir)
    candidates = sorted([p for p in mdl_dir.iterdir() if p.name.endswith("_balanced.pkl")])
    if not candidates:
        candidates = sorted([p for p in mdl_dir.iterdir() if p.suffix == ".pkl"])
    model_path = candidates[-1]
    print("Loading model:", model_path)
    model = joblib.load(str(model_path))
    scaler_path = mdl_dir / DEFAULT_SCALER
    scaler = joblib.load(str(scaler_path)) if scaler_path.exists() else None
    return model, scaler, model_path.name

# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------
def process_domain(domain, model_feature_names, use_external_fe):
    feats = {}
    if use_external_fe and fe_mod:
        try:
            feats = fe_mod.extract_features(domain) or {}
        except Exception:
            feats = fallback_extract_features(domain)
    else:
        feats = fallback_extract_features(domain)

    fv = []
    for fname in model_feature_names:
        val = feats.get(fname, feats.get(fname.lower(), feats.get(fname.replace("-", "_"), 0)))
        fv.append(safe_encode_feature(val))
    return np.array(fv).reshape(1, -1), feats

def main(args):
    df = pd.read_csv(args.input)
    cse_col = [c for c in df.columns if "cse" in c.lower()]
    if not cse_col:
        raise ValueError("No CSE column found in input file!")
    cse_col = cse_col[0]
    df = df[df[cse_col].notna()].copy()
    df["Domain"] = df["Domain"].astype(str)
    domains = df["Domain"].str.strip().str.lower().unique().tolist()
    print(f"Loaded {len(domains)} domains with CSE matches.")

    model, scaler, model_name = load_best_model(args.model_dir)
    model_feature_names = getattr(model, "feature_names_in_", sorted(list(fallback_extract_features(domains[0]).keys())))

    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            cache = json.load(open(CACHE_FILE))
        except Exception:
            cache = {}

    results = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(process_domain, d, model_feature_names, USE_EXTERNAL_FE): d for d in domains}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                Xraw, feats_raw = fut.result()
                Xs = scaler.transform(Xraw) if scaler is not None else Xraw
                prob = float(model.predict_proba(Xs)[:, 1][0]) if hasattr(model, "predict_proba") else float(model.predict(Xs)[0])
                label = int(prob >= args.threshold)
            except Exception as e:
                print(f"⚠️ Failed {d}: {e}")
                continue

            extra = {}
            if label == 1:
                rd = rdap_info(d)
                extra.update(rd)
                extra.update(hosting_info(d))
                extra["screenshot"] = take_screenshot_if_needed(d, label)
            out_row = {"Domain": d, "Predicted_Label": label, "Phishing_Score": prob, "Model": model_name}
            out_row.update(extra)
            results[d] = out_row
            cache[d] = out_row
            if len(results) % args.checkpoint == 0:
                pd.DataFrame(list(results.values())).to_csv(OUTPUT_CSV, index=False)

    pd.DataFrame(list(results.values())).to_csv(OUTPUT_CSV, index=False)
    sub_rows = []
    for r in results.values():
        sub_rows.append({
            "Application_ID": args.application_id,
            "Source_of_detection": args.input,
            "Identified_Domain": r["Domain"],
            "Phishing_Suspected": "Phishing" if r["Predicted_Label"] else "Benign",
            "Hosting_IP": r.get("ips", ""),
            "Evidence_File": r.get("screenshot", ""),
            "Date_of_detection": datetime.now().strftime("%d-%m-%Y"),
            "Time_of_detection": datetime.now().strftime("%H-%M-%S"),
        })
    out_xlsx = OUTPUT_XLSX_TEMPLATE.format(app=args.application_id)
    pd.DataFrame(sub_rows).to_excel(out_xlsx, index=False)
    print(f"✅ Done. Results -> {OUTPUT_CSV} and Excel -> {out_xlsx}")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission predictions and evidence")
    parser.add_argument("--input", required=True, help="Input CSV with Domain and CSE column")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--application_id", default="AIGR-000000")
    parser.add_argument("--checkpoint", type=int, default=200)
    args = parser.parse_args()
    main(args)
