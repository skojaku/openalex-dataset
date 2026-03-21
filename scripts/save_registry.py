"""Save openalex_registry.npz with ID mappings and sync metadata."""

import os
import sys
from datetime import datetime

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import setup_logging

log = setup_logging(__name__)

# --- Snakemake or standalone ---
if "snakemake" in dir():
    paper_index_file = snakemake.input["paper_index"]
    paper_table_file = snakemake.input["paper_table"]
    author_table_file = snakemake.input["author_table"]
    output_file = snakemake.output["registry"]
else:
    paper_index_file = "paper_index.npz"
    paper_table_file = "paper_table.csv"
    author_table_file = "author_table.csv"
    output_file = "openalex_registry.npz"


log.info("Saving OpenAlex registry...")
idx = np.load(paper_index_file)
oa_ids_sorted = idx["oa_ids_sorted"]
n_papers = int(idx["n_papers"][0])

# Load author table to get author count and OA author IDs
author_df = pl.read_csv(author_table_file, columns=["author_id", "openalex_author_id"])
n_authors = len(author_df)
oa_author_ids = author_df.get_column("openalex_author_id").to_numpy().astype(np.int64)
author_ids = author_df.get_column("author_id").to_numpy().astype(np.int64)

sync_date = datetime.now().strftime("%Y-%m-%d")

np.savez(
    output_file,
    # Paper mappings: paper_id = position in oa_ids_sorted
    oa_ids_sorted=oa_ids_sorted,
    # Author mappings
    oa_author_ids=oa_author_ids,
    author_ids=author_ids,
    # Metadata
    n_papers=np.array([n_papers]),
    n_authors=np.array([n_authors]),
    sync_date=np.array([sync_date]),
)
log.info(f"  Papers: {n_papers:,}")
log.info(f"  Authors: {n_authors:,}")
log.info(f"  Sync date: {sync_date}")
log.info(f"  Saved {output_file}")
