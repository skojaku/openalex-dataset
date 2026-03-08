"""Build paper_table.csv from temp metadata and citation network.

Adds is_connected flag (1 = in largest weakly connected component, 0 = not).
"""

import os
import sys

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.csgraph import connected_components

# --- Snakemake or standalone ---
if "snakemake" in dir():
    metadata_file = snakemake.input["metadata"]
    paper_index_file = snakemake.input["paper_index"]
    citation_net_file = snakemake.input["citation_net"]
    output_file = snakemake.output["paper_table"]
else:
    metadata_file = "paper_metadata.csv.gz"
    paper_index_file = "paper_index.npz"
    citation_net_file = "citation_net.npz"
    output_file = "paper_table.csv"


print("Building paper_table.csv...")
df = pl.read_csv(
    metadata_file,
    schema_overrides={"doi": pl.String, "language": pl.String, "type": pl.String, "source_id": pl.Int64},
)
df = df.sort("paper_id")

# Verify completeness against index
idx = np.load(paper_index_file)
n_papers = int(idx["n_papers"][0])
print(f"  Papers in metadata: {len(df):,}")
print(f"  Papers in index: {n_papers:,}")

# --- Compute largest connected component ---
print("Computing largest weakly connected component...")
net = sparse.load_npz(citation_net_file)
n_components, labels = connected_components(net, directed=True, connection="weak")
component_sizes = np.bincount(labels)
largest_component = np.argmax(component_sizes)
lcc_size = component_sizes[largest_component]

print(f"  Components: {n_components:,}")
print(f"  Largest component: {lcc_size:,} ({lcc_size / n_papers:.4f})")

# Add is_connected flag: 1 if in LCC, 0 otherwise
is_connected = (labels == largest_component).astype(np.int8)
# Map to dataframe by paper_id
paper_ids = df.get_column("paper_id").to_numpy()
df = df.with_columns(
    pl.Series("is_connected", is_connected[paper_ids])
)

df.write_csv(output_file)
print(f"  Saved {output_file}")
