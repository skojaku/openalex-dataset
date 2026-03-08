"""Build author_table.csv and paper_author_net.npz from temp files."""

import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import BinaryEdgeReader

# --- Snakemake or standalone ---
if "snakemake" in dir():
    auth_edges_file = snakemake.input["auth_edges"]
    auth_names_file = snakemake.input["auth_names"]
    paper_index_file = snakemake.input["paper_index"]
    output_author_table = snakemake.output["author_table"]
    output_net = snakemake.output["paper_author_net"]
else:
    auth_edges_file = "authorship_edges.bin"
    auth_names_file = "author_names.csv.gz"
    paper_index_file = "paper_index.npz"
    output_author_table = "author_table.csv"
    output_net = "paper_author_net.npz"


print("Building author data...")
idx = np.load(paper_index_file)
n_papers = int(idx["n_papers"][0])

# --- Author table ---
author_df = pd.read_csv(auth_names_file, dtype={"orcid": str, "name": str})
# Deduplicate by author_id (should already be unique from pass2)
author_df = author_df.drop_duplicates(subset="author_id").sort_values("author_id")
author_df = author_df.reset_index(drop=True)
n_authors = len(author_df)
print(f"  Authors: {n_authors:,}")

author_df.to_csv(output_author_table, index=False)
print(f"  Saved {output_author_table}")

# --- Paper-author network ---
reader = BinaryEdgeReader(auth_edges_file)
print(f"  Authorship edges: {reader.n_edges:,}")

paper_ids, author_ids = reader.read_all()

net = sparse.csr_matrix(
    (np.ones(len(paper_ids), dtype=np.int8), (paper_ids, author_ids)),
    shape=(n_papers, n_authors),
)
net.data[:] = 1

sparse.save_npz(output_net, net)
print(f"  Shape: {net.shape}")
print(f"  Non-zero entries: {net.nnz:,}")
print(f"  Saved {output_net}")
