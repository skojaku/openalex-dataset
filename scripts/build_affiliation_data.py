"""Build institution_table.csv and affiliation_table.parquet from temp files."""

import os
import sys

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import BinaryTripleReader, setup_logging

log = setup_logging(__name__)

# --- Snakemake or standalone ---
if "snakemake" in dir():
    aff_edges_file = snakemake.input["aff_edges"]
    inst_names_file = snakemake.input["inst_names"]
    output_institution_table = snakemake.output["institution_table"]
    output_affiliation_table = snakemake.output["affiliation_table"]
else:
    aff_edges_file = "affiliation_edges.bin"
    inst_names_file = "institution_names.csv.gz"
    output_institution_table = "institution_table.csv"
    output_affiliation_table = "affiliation_table.parquet"


log.info("Building affiliation data...")

# --- Institution table ---
inst_df = pl.read_csv(
    inst_names_file,
    schema_overrides={
        "display_name": pl.String,
        "ror": pl.String,
        "country_code": pl.String,
        "type": pl.String,
    },
)
# Deduplicate by institution_id (should already be unique from pass2)
inst_df = inst_df.unique(subset="institution_id").sort("institution_id")
log.info(f"  Institutions: {len(inst_df):,}")

inst_df.write_csv(output_institution_table)
log.info(f"  Saved {output_institution_table}")

# --- Affiliation table (paper_id, author_id, institution_id) ---
reader = BinaryTripleReader(aff_edges_file)
log.info(f"  Affiliation edges: {reader.n_triples:,}")

schema = pa.schema([
    ("paper_id", pa.int32()),
    ("author_id", pa.int32()),
    ("institution_id", pa.int32()),
])
writer = pq.ParquetWriter(output_affiliation_table, schema)
for paper_ids, author_ids, institution_ids in reader.iter_chunks(chunk_size=50_000_000):
    batch = pa.record_batch(
        [
            pa.array(paper_ids, pa.int32()),
            pa.array(author_ids, pa.int32()),
            pa.array(institution_ids, pa.int32()),
        ],
        schema=schema,
    )
    writer.write_batch(batch)
writer.close()
log.info(f"  Saved {output_affiliation_table}")
