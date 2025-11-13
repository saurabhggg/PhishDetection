import pandas as pd
df = pd.read_csv("prefilter_scored.csv")
filtered = df[df["CSE_Score"] >= 0.5]  # or 0.5 for stricter cutoff
filtered.to_csv("prefilter_for_heavy.csv", index=False)
print("✅ Filtered:", len(filtered))
