import pandas as pd

df = pd.read_csv("CSE_Relevance_Output.csv")

filtered = df[

    df["Relation_Score"] >= 0.2             # moderately or strongly related
]
filtered.to_csv("CSE_Relevance_Filtered.csv", index=False)

print("✅ Filtered domains:", len(filtered))
