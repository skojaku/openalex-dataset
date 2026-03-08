"""Build citation_net.npz from binary edge file."""

import os
import sys

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import BinaryEdgeReader

# --- Snakemake or standalone ---
if "snakemake" in dir():
    cit_edges_file = snakemake.input["cit_edges"]
    paper_index_file = snakemake.input["paper_index"]
    output_file = snakemake.output["citation_net"]
else:
    cit_edges_file = "citation_edges.bin"
    paper_index_file = "paper_index.npz"
    output_file = "citation_net.npz"


print("Building citation_net.npz...")
idx = np.load(paper_index_file)
n_papers = int(idx["n_papers"][0])

reader = BinaryEdgeReader(cit_edges_file)
print(f"  Total edges in file: {reader.n_edges:,}")

srcs, dsts = reader.read_all()

# Remove self-loops
mask = srcs != dsts
srcs = srcs[mask]
dsts = dsts[mask]

# Build sparse matrix
net = sparse.csr_matrix(
    (np.ones(len(srcs), dtype=np.int8), (srcs, dsts)),
    shape=(n_papers, n_papers),
)
# Eliminate duplicates (some works may cite the same ref multiple times)
net.data[:] = 1

sparse.save_npz(output_file, net)
print(f"  Shape: {net.shape}")
print(f"  Non-zero entries: {net.nnz:,}")
print(f"  Saved {output_file}")
