"""Pass 1: Scan OpenAlex works snapshot and build paper index.

Streams all works, extracts (openalex_numeric_id, year) for each,
sorts by (year, oa_id), assigns sequential paper_id = 0..N-1.

Output: paper_index.npz with arrays:
  - oa_ids_sorted: OpenAlex numeric IDs sorted by oa_id (for binary search lookup)
  - paper_ids_for_sorted: corresponding paper_ids for the sorted oa_ids
  - oa_ids_by_paper: OpenAlex numeric IDs in paper_id order
  - years: publication year in paper_id order
  - n_papers: total number of papers
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import parse_openalex_id, stream_works

# --- Snakemake or standalone ---
if "snakemake" in dir():
    snapshot_dir = snakemake.params["snapshot_dir"]
    output_file = snakemake.output["paper_index"]
else:
    snapshot_dir = "/data/raw/openalex"
    output_file = "paper_index.npz"


# --- Stream works and collect (oa_id, year) ---
INIT_SIZE = 100_000_000  # pre-allocate for ~250M works
oa_ids = np.empty(INIT_SIZE, dtype=np.int64)
years = np.empty(INIT_SIZE, dtype=np.int16)
count = 0

print("Pass 1: Scanning works to build paper index...")
for work in stream_works(snapshot_dir):
    oa_id = parse_openalex_id(work.get("id"))
    if oa_id < 0:
        continue
    year = work.get("publication_year")
    if year is None:
        year = 0

    # Grow arrays if needed
    if count >= len(oa_ids):
        new_size = int(len(oa_ids) * 1.5)
        oa_ids = np.resize(oa_ids, new_size)
        years = np.resize(years, new_size)

    oa_ids[count] = oa_id
    years[count] = year
    count += 1

    if count % 10_000_000 == 0:
        print(f"  Scanned {count:,} works...")

# Trim to actual size
oa_ids = oa_ids[:count]
years = years[:count]
print(f"Pass 1 complete: {count:,} works found.")

if count == 0:
    raise RuntimeError(
        f"No works found in {snapshot_dir}. "
        f"Expected JSONL files at {snapshot_dir}/data/works/updated_date=*/*.gz. "
        f"Download with: aws s3 sync 's3://openalex' '{snapshot_dir}' --no-sign-request"
    )

# --- Sort by (year, oa_id) and assign paper_ids ---
sort_order = np.lexsort((oa_ids, years))
oa_ids_by_paper = oa_ids[sort_order]
years_by_paper = years[sort_order]
# paper_id is simply the position in sorted order: 0, 1, 2, ...

# --- Build lookup structure: oa_id -> paper_id ---
# Sort by oa_id for binary search
lookup_order = np.argsort(oa_ids_by_paper)
oa_ids_sorted = oa_ids_by_paper[lookup_order]
paper_ids_for_sorted = lookup_order.astype(np.int64)

# --- Save ---
os.makedirs(os.path.dirname(output_file), exist_ok=True)
np.savez(
    output_file,
    oa_ids_sorted=oa_ids_sorted,
    paper_ids_for_sorted=paper_ids_for_sorted,
    oa_ids_by_paper=oa_ids_by_paper,
    years=years_by_paper.astype(np.int16),
    n_papers=np.array([count]),
)
print(f"Saved paper index to {output_file}")
print(f"  Papers: {count:,}")
print(f"  Year range: {years_by_paper.min()} - {years_by_paper.max()}")
