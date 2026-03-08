"""Build source_table.csv from temp source names."""

import os
import sys

import pandas as pd

# --- Snakemake or standalone ---
if "snakemake" in dir():
    source_names_file = snakemake.input["source_names"]
    output_file = snakemake.output["source_table"]
else:
    source_names_file = "source_names.csv.gz"
    output_file = "source_table.csv"


print("Building source_table.csv...")
df = pd.read_csv(source_names_file, dtype=str)
df["source_id"] = df["source_id"].astype(int)
df["openalex_source_id"] = df["openalex_source_id"].astype(int)

# Deduplicate (should already be unique from pass2)
df = df.drop_duplicates(subset="source_id").sort_values("source_id").reset_index(drop=True)

df.to_csv(output_file, index=False)
print(f"  Sources: {len(df):,}")
print(f"  Saved {output_file}")
