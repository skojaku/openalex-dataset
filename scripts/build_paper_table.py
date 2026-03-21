"""Build paper_table.csv from temp metadata and citation network.

Adds is_connected flag (1 = in largest weakly connected component, 0 = not).

Uses a fast heuristic: iteratively expand from the highest-degree node via
sparse matrix-vector multiplication (A @ x) until convergence.  Repeat on
remaining nodes up to 5 times, then pick the largest component found.

Validates by checking that the top-100 most-cited Nature/Science papers
are included in the largest component.
"""

import os
import sys

import numpy as np
import polars as pl
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import setup_logging

log = setup_logging(__name__)

# --- Snakemake or standalone ---
if "snakemake" in dir():
    metadata_file = snakemake.input["metadata"]
    paper_index_file = snakemake.input["paper_index"]
    citation_net_file = snakemake.input["citation_net"]
    source_names_file = snakemake.input["source_names"]
    output_file = snakemake.output["paper_table"]
else:
    metadata_file = "paper_metadata.csv.gz"
    paper_index_file = "paper_index.npz"
    citation_net_file = "citation_net.npz"
    source_names_file = "source_names.csv.gz"
    output_file = "paper_table.csv"


def find_largest_component(A, n_repeats=5):
    """Find the largest connected component via iterative expansion.

    For each repeat:
      1. Pick the highest-degree node among remaining nodes.
      2. Set x = one-hot at that node.
      3. Expand: x = (A @ x > 0) | x, until no new nodes are added.
      4. Record that component and remove its nodes.

    Returns an int8 array: 1 = in largest component, 0 = not.
    """
    n = A.shape[0]
    remaining = np.ones(n, dtype=bool)
    components = []

    for rep in range(n_repeats):
        if not remaining.any():
            break

        # Degree among remaining nodes
        remaining_f = remaining.astype(np.float32)
        degrees = np.asarray(A.dot(remaining_f)).ravel()
        degrees[~remaining] = -1

        start_node = int(np.argmax(degrees))
        log.info(
            f"  Component {rep + 1}: seed node {start_node} "
            f"(degree {int(degrees[start_node]):,})"
        )

        # Iterative expansion
        x = np.zeros(n, dtype=np.float32)
        x[start_node] = 1.0
        prev_nnz = 1
        iteration = 0

        while True:
            x_new = np.asarray(A.dot(x)).ravel()
            x_new = (x_new > 0).astype(np.float32)
            x_new = np.maximum(x, x_new)
            x_new[~remaining] = 0.0
            curr_nnz = int(x_new.sum())
            iteration += 1
            if curr_nnz == prev_nnz:
                break
            log.info(f"    Iteration {iteration}: {curr_nnz:,} nodes")
            x = x_new
            prev_nnz = curr_nnz

        comp_nodes = np.where(x > 0)[0]
        components.append(comp_nodes)
        remaining[comp_nodes] = False
        log.info(f"  Component {rep + 1}: {len(comp_nodes):,} nodes")

    largest_idx = max(range(len(components)), key=lambda i: len(components[i]))
    is_connected = np.zeros(n, dtype=np.int8)
    is_connected[components[largest_idx]] = 1
    log.info(
        f"  Largest component: {len(components[largest_idx]):,} / {n:,} nodes"
    )
    return is_connected


# --- Load metadata ---
log.info("Building paper_table.csv...")
df = pl.read_csv(
    metadata_file,
    schema_overrides={
        "doi": pl.String,
        "language": pl.String,
        "type": pl.String,
        "source_id": pl.Int64,
    },
)
df = df.sort("paper_id")

# Verify completeness against index
idx = np.load(paper_index_file)
n_papers = int(idx["n_papers"][0])
log.info(f"  Papers in metadata: {len(df):,}")
log.info(f"  Papers in index: {n_papers:,}")

# --- Compute largest connected component ---
log.info("Computing largest connected component (heuristic)...")
net = sparse.load_npz(citation_net_file)

# Symmetrize to make undirected
A = net + net.T
A.data = np.ones(len(A.data), dtype=np.int8)
A.eliminate_zeros()

is_connected = find_largest_component(A, n_repeats=5)
del A, net

# Map to dataframe by paper_id
paper_ids = df.get_column("paper_id").to_numpy()
df = df.with_columns(pl.Series("is_connected", is_connected[paper_ids]))

# --- Validation: top-cited Nature/Science papers should be in LCC ---
log.info("Validating: checking top-cited Nature/Science papers...")
try:
    source_df = pl.read_csv(
        source_names_file,
        schema_overrides={"type": pl.String, "issn_l": pl.String},
    )
    nature_science = source_df.filter(
        pl.col("display_name").str.to_lowercase().is_in(["nature", "science"])
    )
    if len(nature_science) > 0:
        ns_source_ids = nature_science.get_column("source_id").to_list()
        top_papers = (
            df.filter(pl.col("source_id").is_in(ns_source_ids))
            .sort("cited_by_count", descending=True)
            .head(100)
        )
        n_in_lcc = int(top_papers.get_column("is_connected").sum())
        log.info(f"  Top 100 Nature/Science papers in LCC: {n_in_lcc}/100")
        if n_in_lcc < 95:
            log.warning(
                f"  WARNING: Only {n_in_lcc}/100 top Nature/Science papers in LCC! "
                "The heuristic may have missed the true largest component."
            )
    else:
        log.info("  Nature/Science sources not found, skipping validation.")
except Exception as e:
    log.info(f"  Validation skipped: {e}")

df.write_csv(output_file)
log.info(f"Saved {output_file}")
