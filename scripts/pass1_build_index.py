"""Pass 1: Scan OpenAlex works snapshot and build paper index.

Streams all works, collects OpenAlex numeric IDs, sorts them, and assigns
sequential paper_id = 0..N-1 (in oa_id sorted order for binary search).

Output: paper_index.npz with arrays:
  - oa_ids_sorted: OpenAlex numeric IDs sorted (paper_id = position in this array)
  - n_papers: total number of papers
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import parse_openalex_id, setup_logging, stream_works

log = setup_logging(__name__)

KEEP_TYPES = {"article", "book-chapter", "preprint", "review", "letter", "book-section"}

# --- Snakemake or standalone ---
if "snakemake" in dir():
    snapshot_dir = snakemake.params["snapshot_dir"]
    output_file = snakemake.output["paper_index"]
else:
    snapshot_dir = "/data/raw/openalex"
    output_file = "paper_index.npz"


# --- Stream works and collect oa_ids ---
INIT_SIZE = 100_000_000  # pre-allocate for ~250M works
oa_ids = np.empty(INIT_SIZE, dtype=np.int64)
count = 0

log.info("Pass 1: Scanning works to build paper index...")
for work in stream_works(snapshot_dir):
    oa_id = parse_openalex_id(work.get("id"))
    if oa_id < 0:
        continue

    work_type = work.get("type", "")
    if work_type not in KEEP_TYPES:
        continue

    # Grow array if needed
    if count >= len(oa_ids):
        new_size = int(len(oa_ids) * 1.5)
        oa_ids = np.resize(oa_ids, new_size)

    oa_ids[count] = oa_id
    count += 1

    if count % 10_000_000 == 0:
        log.info(f"  Scanned {count:,} works...")

# Trim to actual size
oa_ids = oa_ids[:count]
log.info(f"Pass 1 complete: {count:,} works found.")

if count == 0:
    raise RuntimeError(
        f"No works found in {snapshot_dir}. "
        f"Expected JSONL files at {snapshot_dir}/data/works/updated_date=*/*.gz. "
        f"Download with: aws s3 sync 's3://openalex' '{snapshot_dir}' --no-sign-request"
    )

# --- Sort by oa_id and assign paper_ids ---
# paper_id = position in sorted array (0, 1, 2, ...)
oa_ids_sorted = np.sort(oa_ids)

# --- Save ---
os.makedirs(os.path.dirname(output_file), exist_ok=True)
np.savez(
    output_file,
    oa_ids_sorted=oa_ids_sorted,
    n_papers=np.array([count]),
)
log.info(f"Saved paper index to {output_file}")
log.info(f"  Papers: {count:,}")
