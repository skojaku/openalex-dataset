"""Build category_table.csv and paper_category_table.csv from temp topic data."""

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import setup_logging

log = setup_logging(__name__)

# --- Snakemake or standalone ---
if "snakemake" in dir():
    topics_file = snakemake.input["topics"]
    paper_index_file = snakemake.input["paper_index"]
    output_category_table = snakemake.output["category_table"]
    output_paper_category_table = snakemake.output["paper_category_table"]
else:
    topics_file = "paper_topics.csv.gz"
    paper_index_file = "paper_index.npz"
    output_category_table = "category_table.csv"
    output_paper_category_table = "paper_category_table.csv"


log.info("Building category data...")
df = pl.read_csv(topics_file)

# --- Build category table ---
# Extract unique fields (main classes)
fields = (
    df.select("field_id", "field_name")
    .unique(subset="field_id")
    .filter(pl.col("field_id") >= 0)
    .sort("field_id")
)

# Extract unique subfields (sub classes)
subfields = (
    df.select("subfield_id", "subfield_name")
    .unique(subset="subfield_id")
    .filter(pl.col("subfield_id") >= 0)
    .sort("subfield_id")
)

# Assign sequential class_ids
field_ids = fields.get_column("field_id").to_list()
field_names = fields.get_column("field_name").to_list()
subfield_ids = subfields.get_column("subfield_id").to_list()
subfield_names = subfields.get_column("subfield_name").to_list()

field_id_to_class_id = {fid: i for i, fid in enumerate(field_ids)}
offset = len(field_ids)
subfield_id_to_class_id = {sid: offset + i for i, sid in enumerate(subfield_ids)}

cat_df = pl.DataFrame({
    "class_id": list(range(len(field_ids) + len(subfield_ids))),
    "type": ["main"] * len(field_ids) + ["sub"] * len(subfield_ids),
    "title": field_names + subfield_names,
    "openalex_id": field_ids + subfield_ids,
})

cat_df.write_csv(output_category_table)
log.info(f"  Categories: {len(cat_df)} ({len(field_ids)} fields, {len(subfield_ids)} subfields)")
log.info(f"  Saved {output_category_table}")

# --- Build paper-category table ---
# Map field_id and subfield_id to class_id
field_map_df = pl.DataFrame({
    "field_id": list(field_id_to_class_id.keys()),
    "main_class_id": list(field_id_to_class_id.values()),
})
subfield_map_df = pl.DataFrame({
    "subfield_id": list(subfield_id_to_class_id.keys()),
    "sub_class_id": list(subfield_id_to_class_id.values()),
})

paper_cat_df = (
    df.join(field_map_df, on="field_id", how="left")
    .join(subfield_map_df, on="subfield_id", how="left")
    .filter(pl.col("main_class_id").is_not_null())
    .select(
        pl.col("paper_id"),
        pl.col("main_class_id").cast(pl.Int64),
        pl.col("sub_class_id").fill_null(-1).cast(pl.Int64),
        pl.col("sequence"),
    )
    .sort("paper_id", "sequence")
)

paper_cat_df.write_csv(output_paper_category_table)
log.info(f"  Paper-category assignments: {len(paper_cat_df):,}")
log.info(f"  Saved {output_paper_category_table}")
