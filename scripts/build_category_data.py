"""Build category_table.csv and paper_category_table.csv from temp topic data."""

import os
import sys

import numpy as np
import pandas as pd

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


print("Building category data...")
df = pd.read_csv(topics_file)

# --- Build category table ---
# Extract unique fields (main classes)
fields = df[["field_id", "field_name"]].drop_duplicates(subset="field_id")
fields = fields[fields["field_id"] >= 0].sort_values("field_id").reset_index(drop=True)

# Extract unique subfields (sub classes)
subfields = df[["subfield_id", "subfield_name"]].drop_duplicates(subset="subfield_id")
subfields = subfields[subfields["subfield_id"] >= 0].sort_values("subfield_id").reset_index(drop=True)

# Assign sequential class_ids
cat_rows = []
field_id_to_class_id = {}
for i, row in enumerate(fields.itertuples()):
    class_id = i
    field_id_to_class_id[row.field_id] = class_id
    cat_rows.append({
        "class_id": class_id,
        "type": "main",
        "title": row.field_name,
        "openalex_id": int(row.field_id),
    })

subfield_id_to_class_id = {}
offset = len(fields)
for i, row in enumerate(subfields.itertuples()):
    class_id = offset + i
    subfield_id_to_class_id[row.subfield_id] = class_id
    cat_rows.append({
        "class_id": class_id,
        "type": "sub",
        "title": row.subfield_name,
        "openalex_id": int(row.subfield_id),
    })

cat_df = pd.DataFrame(cat_rows)
cat_df.to_csv(output_category_table, index=False)
print(f"  Categories: {len(cat_df)} ({len(fields)} fields, {len(subfields)} subfields)")
print(f"  Saved {output_category_table}")

# --- Build paper-category table ---
paper_cat_rows = []
for row in df.itertuples():
    main_class_id = field_id_to_class_id.get(row.field_id, -1)
    sub_class_id = subfield_id_to_class_id.get(row.subfield_id, -1)
    if main_class_id >= 0:
        paper_cat_rows.append({
            "paper_id": int(row.paper_id),
            "main_class_id": main_class_id,
            "sub_class_id": sub_class_id,
            "sequence": int(row.sequence),
        })

paper_cat_df = pd.DataFrame(paper_cat_rows)
paper_cat_df = paper_cat_df.sort_values(["paper_id", "sequence"]).reset_index(drop=True)
paper_cat_df.to_csv(output_paper_category_table, index=False)
print(f"  Paper-category assignments: {len(paper_cat_df):,}")
print(f"  Saved {output_paper_category_table}")
