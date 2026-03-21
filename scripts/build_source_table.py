"""Build source_table.csv from temp source names."""

import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import setup_logging

log = setup_logging(__name__)

# --- Snakemake or standalone ---
if "snakemake" in dir():
    source_names_file = snakemake.input["source_names"]
    output_file = snakemake.output["source_table"]
else:
    source_names_file = "source_names.csv.gz"
    output_file = "source_table.csv"


log.info("Building source_table.csv...")
df = pl.read_csv(source_names_file, schema_overrides={"source_id": pl.Int64, "openalex_source_id": pl.Int64})

# Deduplicate (should already be unique from pass2)
df = df.unique(subset="source_id").sort("source_id")

df.write_csv(output_file)
log.info(f"  Sources: {len(df):,}")
log.info(f"  Saved {output_file}")
