"""Build abstracts.parquet from temp abstracts CSV.

Processes in chunks to avoid OOM on large datasets (~hundreds of millions of rows).
Uses PyArrow writer to append row groups incrementally.
"""

import csv
import gzip
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import setup_logging

log = setup_logging(__name__)

# --- Snakemake or standalone ---
if "snakemake" in dir():
    abstracts_file = snakemake.input["abstracts"]
    output_file = snakemake.output["abstracts_parquet"]
else:
    abstracts_file = "abstracts.csv.gz"
    output_file = "abstracts.parquet"


CHUNK_SIZE = 1_000_000  # rows per chunk (kept small to avoid Arrow offset overflow)

schema = pa.schema([
    ("paper_id", pa.int64()),
    ("abstract", pa.large_string()),
])

log.info("Building abstracts.parquet (chunked)...")
total_rows = 0

writer = pq.ParquetWriter(output_file, schema, compression="snappy")

paper_ids = []
abstracts = []

with gzip.open(abstracts_file, "rt", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header

    for row in reader:
        paper_ids.append(int(row[0]))
        abstracts.append(row[1] if len(row) > 1 else "")

        if len(paper_ids) >= CHUNK_SIZE:
            table = pa.table({
                "paper_id": pa.array(paper_ids, type=pa.int64()),
                "abstract": pa.array(abstracts, type=pa.large_string()),
            })
            # Sort chunk by paper_id
            table = table.sort_by("paper_id")
            writer.write_table(table)
            total_rows += len(paper_ids)
            log.info(f"  Written {total_rows:,} rows...")
            paper_ids.clear()
            abstracts.clear()

    # Flush remaining
    if paper_ids:
        table = pa.table({
            "paper_id": pa.array(paper_ids, type=pa.int64()),
            "abstract": pa.array(abstracts, type=pa.large_string()),
        })
        table = table.sort_by("paper_id")
        writer.write_table(table)
        total_rows += len(paper_ids)

writer.close()
log.info(f"  Total abstracts: {total_rows:,}")
log.info(f"  Saved {output_file}")
