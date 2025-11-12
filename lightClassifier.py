#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prefilter_cseaware.py – Improved CSE-aware Prefilter
----------------------------------------------------
Fast lexical + CSE resemblance scorer for massive domain lists.

✅ Input:
    CSV with a column containing 'domain' or 'url'
✅ Output:
    prefilter_scored.csv with:
      Domain, base_host, CSE_Score (0..1), risk_bucket, reason, matched_CSE

✅ Features:
  - Lexical suspiciousness (length, dots, hyphens, digits, entropy)
  - Suspicious TLD detection
  - CSE resemblance via substring + fuzzy distance
  - Optional semantic similarity (commented for speed)

Usage:
  python prefilter_cseaware.py --input combined_domains.csv --out prefilter_scored.csv
"""

import argparse
import os
import re
import time
import math
import pandas as pd
import numpy as np
from pathlib import Path
import tldextract
from difflib import SequenceMatcher
from sklearn.preprocessing import MinMaxScaler

# ---------- CONFIG ----------
OUT_DEFAULT = "prefilter_scored.csv"
SUSP_TLDS = {".xyz", ".top", ".tk", ".ga", ".cf", ".ml", ".gq", ".cam", ".buzz", ".click", ".work", ".monster", ".icu"}

# Critical Sector Entities (CSEs)
CSE_KEYWORDS = {
    "SBI": ["sbi", "sbicard", "onlinesbi", "sbiepay", "sbilife"],
    "ICICI": ["icici", "icicibank", "iciciprulife", "icicidirect", "icicilombard"],
    "HDFC": ["hdfc", "hdfcbank", "hdfclife", "hdfcergo"],
    "PNB": ["pnb", "pnbindia", "netpnb"],
    "BoB": ["bankofbaroda", "bobibanking", "bob"],
    "NIC": ["nic", "gov.in", "kavach"],
    "IRCTC": ["irctc"],
    "Airtel": ["airtel"],
    "IOCL": ["iocl", "indianoil"],
}

# ---------- HELPERS ----------
def detect_domain_col(df):
    for c in df.columns:
        if "domain" in c.lower() or "url" in c.lower():
            return c
    return df.columns[0]

def normalize_host(s):
    s = (str(s) or "").strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = s.split("/")[0]
    return s

def entropy(s):
    if not s:
        return 0.0
    vals = {}
    for ch in s:
        vals[ch] = vals.get(ch, 0) + 1
    ent = 0.0
    L = len(s)
    for v in vals.values():
        p = v / L
        ent -= p * math.log2(p)
    return ent

def levenshtein_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

def lexical_features(hosts):
    n = len(hosts)
    features = {
        "len_host": np.zeros(n),
        "num_dots": np.zeros(n),
        "num_hyphen": np.zeros(n),
        "num_digits": np.zeros(n),
        "digit_ratio": np.zeros(n),
        "num_special": np.zeros(n),
        "subdomain_count": np.zeros(n),
        "avg_sub_len": np.zeros(n),
        "susp_tld": np.zeros(n),
        "entropy_host": np.zeros(n),
    }

    for i, h in enumerate(hosts):
        if not h or h == "nan":
            continue
        features["len_host"][i] = len(h)
        features["num_dots"][i] = h.count(".")
        features["num_hyphen"][i] = h.count("-")
        digits = sum(c.isdigit() for c in h)
        features["num_digits"][i] = digits
        features["digit_ratio"][i] = digits / (len(h) + 1e-6)
        features["num_special"][i] = sum(h.count(ch) for ch in "_$#%!")
        parts = h.split(".")
        features["subdomain_count"][i] = max(0, len(parts) - 2)
        lens = [len(p) for p in parts[:-2]] if len(parts) > 2 else []
        features["avg_sub_len"][i] = float(np.mean(lens)) if lens else 0.0
        ext = tldextract.extract(h)
        tld = ("." + ext.suffix) if ext.suffix else ""
        features["susp_tld"][i] = 1 if tld in SUSP_TLDS else 0
        features["entropy_host"][i] = entropy(h)
    return features

def cse_resemblance_features(host):
    best_match = None
    best_score = 0.0
    for cse, kws in CSE_KEYWORDS.items():
        for kw in kws:
            if kw in host:
                # exact or substring match
                score = min(1.0, 0.6 + len(kw)/len(host))
            else:
                score = levenshtein_ratio(host, kw)
            if score > best_score:
                best_score = score
                best_match = cse
    return best_match, best_score

# ---------- MAIN ----------
def main(input_csv, out_csv, min_cse_score=0.45):
    df = pd.read_csv(input_csv)
    domain_col = detect_domain_col(df)
    hosts = df[domain_col].astype(str).map(normalize_host).tolist()
    print(f"📥 Loaded {len(hosts)} domains")

    existing = set()
    if Path(out_csv).exists():
        try:
            prev = pd.read_csv(out_csv)
            existing = set(prev["Domain"].astype(str).str.lower())
        except Exception:
            pass

    feats = lexical_features(hosts)
    lex_mat = np.column_stack([
        feats["len_host"], feats["num_dots"], feats["num_hyphen"], feats["num_digits"],
        feats["digit_ratio"], feats["num_special"], feats["avg_sub_len"],
        feats["susp_tld"], feats["entropy_host"]
    ])

    scaler = MinMaxScaler()
    lex_scaled = scaler.fit_transform(lex_mat)
    lex_score = np.mean(lex_scaled, axis=1)  # combined lexical suspiciousness

    rows = []
    t0 = time.time()
    for i, h in enumerate(hosts):
        if not h or h in existing:
            continue

        matched_cse, cse_sim = cse_resemblance_features(h)

        # weighted combination of lexical suspiciousness + cse resemblance
        score = 0.4 * lex_score[i] + 0.6 * cse_sim

        if score >= 0.75:
            bucket = "high"
        elif score >= 0.45:
            bucket = "medium"
        else:
            bucket = "low"

        rows.append({
            "Domain": h,
            "base_host": h,
            "CSE_Score": round(score, 4),
            "CSE_Match": matched_cse or "",
            "Lexical_Score": round(float(lex_score[i]), 4),
            "CSE_Similarity": round(float(cse_sim), 4),
            "risk_bucket": bucket,
            "reason": "lexical+cse",
        })

        if len(rows) >= 50000:
            pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)
            print(f"💾 checkpoint {i}/{len(hosts)} processed...")
            rows = []

    if rows:
        pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)

    print(f"✅ Done. Wrote to {out_csv}")
    print(f"⌛ Time elapsed: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV file")
    ap.add_argument("--out", default=OUT_DEFAULT, help="Output CSV file")
    ap.add_argument("--min-score", type=float, default=0.45, help="Min score to pass (for downstream use)")
    args = ap.parse_args()
    main(args.input, args.out, args.min_score)
