"""Filter all outputs to the largest connected component (LCC).

Reads unfiltered outputs, keeps only papers with is_connected == 1,
reassigns all IDs to be contiguous (0..N-1), and writes filtered outputs.
"""

import os
import sys
from datetime import datetime

# Limit threads for BLAS/OpenMP before importing numpy/scipy
N_THREADS = str(os.environ.get("OMP_NUM_THREADS", "10"))
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = N_THREADS

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import setup_logging

log = setup_logging(__name__)

# --- Snakemake or standalone ---
if "snakemake" in dir():
    in_paper_table = snakemake.input["paper_table"]
    in_citation_net = snakemake.input["citation_net"]
    in_author_table = snakemake.input["author_table"]
    in_paper_author_net = snakemake.input["paper_author_net"]
    in_category_table = snakemake.input["category_table"]
    in_paper_category_table = snakemake.input["paper_category_table"]
    in_abstracts = snakemake.input["abstracts"]
    in_source_table = snakemake.input["source_table"]
    out_paper_table = snakemake.output["paper_table"]
    out_citation_net = snakemake.output["citation_net"]
    out_author_table = snakemake.output["author_table"]
    out_paper_author_net = snakemake.output["paper_author_net"]
    out_category_table = snakemake.output["category_table"]
    out_paper_category_table = snakemake.output["paper_category_table"]
    out_abstracts = snakemake.output["abstracts"]
    out_source_table = snakemake.output["source_table"]
    out_registry = snakemake.output["registry"]
else:
    base_in = "/data/datasets/openalex/preprocessed_unfiltered"
    base_out = "/data/datasets/openalex/preprocessed"
    in_paper_table = f"{base_in}/paper_table.csv"
    in_citation_net = f"{base_in}/citation_net.npz"
    in_author_table = f"{base_in}/author_table.csv"
    in_paper_author_net = f"{base_in}/paper_author_net.npz"
    in_category_table = f"{base_in}/category_table.csv"
    in_paper_category_table = f"{base_in}/paper_category_table.csv"
    in_abstracts = f"{base_in}/abstracts.parquet"
    in_source_table = f"{base_in}/source_table.csv"
    out_paper_table = f"{base_out}/paper_table.csv"
    out_citation_net = f"{base_out}/citation_net.npz"
    out_author_table = f"{base_out}/author_table.csv"
    out_paper_author_net = f"{base_out}/paper_author_net.npz"
    out_category_table = f"{base_out}/category_table.csv"
    out_paper_category_table = f"{base_out}/paper_category_table.csv"
    out_abstracts = f"{base_out}/abstracts.parquet"
    out_source_table = f"{base_out}/source_table.csv"
    out_registry = f"{base_out}/openalex_registry.npz"


os.makedirs(os.path.dirname(out_paper_table), exist_ok=True)

# =========================================================================
# 1. Paper table — filter and build ID mapping
# =========================================================================
log.info("Filtering to largest connected component...")

paper_df = pl.read_csv(
    in_paper_table,
    schema_overrides={
        "doi": pl.String, "language": pl.String,
        "type": pl.String, "source_id": pl.Int64,
    },
)
n_total = len(paper_df)

lcc_df = paper_df.filter(pl.col("is_connected") == 1).drop("is_connected")
n_lcc = len(lcc_df)
log.info(f"  Papers: {n_lcc:,} / {n_total:,} kept")

# old_paper_id -> new_paper_id (0..N-1, sorted by old paper_id)
old_paper_ids = lcc_df.get_column("paper_id").to_numpy()
old_to_new_paper = np.full(n_total, -1, dtype=np.int64)
new_paper_ids = np.arange(n_lcc, dtype=np.int64)
old_to_new_paper[old_paper_ids] = new_paper_ids

lcc_df = lcc_df.with_columns(pl.Series("paper_id", new_paper_ids))
lcc_df.write_csv(out_paper_table)
log.info(f"  Saved {out_paper_table}")

# =========================================================================
# 2. Citation network — submatrix with new IDs
# =========================================================================
log.info("Filtering citation network...")
net = sparse.load_npz(in_citation_net)
# Extract submatrix for LCC papers
net_filtered = net[old_paper_ids][:, old_paper_ids]
sparse.save_npz(out_citation_net, net_filtered.tocsr())
log.info(f"  Edges: {net_filtered.nnz:,} (was {net.nnz:,})")
log.info(f"  Saved {out_citation_net}")
del net, net_filtered

# =========================================================================
# 3. Paper-author network — filter, remap author IDs
# =========================================================================
log.info("Filtering paper-author network...")
pa_net = sparse.load_npz(in_paper_author_net)
# Keep rows for LCC papers
pa_filtered = pa_net[old_paper_ids]

# Find authors that still have at least one paper
author_mask = np.array(pa_filtered.sum(axis=0)).ravel() > 0
old_author_ids = np.where(author_mask)[0]
n_authors_old = pa_net.shape[1]
n_authors_new = len(old_author_ids)
log.info(f"  Authors: {n_authors_new:,} (was {n_authors_old:,})")

# Subselect columns and save
pa_filtered = pa_filtered[:, old_author_ids]
sparse.save_npz(out_paper_author_net, pa_filtered.tocsr())
log.info(f"  Saved {out_paper_author_net}")
del pa_net, pa_filtered

# old_author_id -> new_author_id
old_to_new_author = np.full(n_authors_old, -1, dtype=np.int64)
old_to_new_author[old_author_ids] = np.arange(n_authors_new, dtype=np.int64)

# =========================================================================
# 4. Author table — filter and remap
# =========================================================================
log.info("Filtering author table...")
author_df = pl.read_csv(
    in_author_table,
    schema_overrides={"orcid": pl.String, "name": pl.String},
)
author_df = author_df.filter(
    pl.col("author_id").is_in(old_author_ids.tolist())
)
# Remap author_id
old_aids = author_df.get_column("author_id").to_numpy()
author_df = author_df.with_columns(
    pl.Series("author_id", old_to_new_author[old_aids])
).sort("author_id")
author_df.write_csv(out_author_table)
log.info(f"  Authors: {len(author_df):,}")
log.info(f"  Saved {out_author_table}")

# =========================================================================
# 5. Paper-category table — filter and remap paper_id
# =========================================================================
log.info("Filtering paper-category table...")
pcat_df = pl.read_csv(in_paper_category_table)
# Keep only LCC papers (old_to_new_paper >= 0)
old_pids = pcat_df.get_column("paper_id").to_numpy()
keep_mask = old_to_new_paper[old_pids] >= 0
pcat_df = pcat_df.filter(pl.Series(keep_mask))
# Remap paper_id
old_pids = pcat_df.get_column("paper_id").to_numpy()
pcat_df = pcat_df.with_columns(
    pl.Series("paper_id", old_to_new_paper[old_pids])
).sort("paper_id", "sequence")
pcat_df.write_csv(out_paper_category_table)
log.info(f"  Paper-category rows: {len(pcat_df):,}")
log.info(f"  Saved {out_paper_category_table}")

# =========================================================================
# 6. Category table — keep only categories that appear
# =========================================================================
log.info("Filtering category table...")
cat_df = pl.read_csv(in_category_table)
used_main = set(pcat_df.get_column("main_class_id").to_list())
used_sub = set(pcat_df.get_column("sub_class_id").to_list()) - {-1}
used_class_ids = used_main | used_sub
cat_df = cat_df.filter(pl.col("class_id").is_in(list(used_class_ids)))
cat_df.write_csv(out_category_table)
log.info(f"  Categories: {len(cat_df):,}")
log.info(f"  Saved {out_category_table}")

# =========================================================================
# 7. Abstracts — filter and remap paper_id
# =========================================================================
log.info("Filtering abstracts...")
abs_df = pl.read_parquet(in_abstracts)
old_abs_pids = abs_df.get_column("paper_id").to_numpy()
abs_keep = old_to_new_paper[old_abs_pids] >= 0
abs_df = abs_df.filter(pl.Series(abs_keep))
old_abs_pids = abs_df.get_column("paper_id").to_numpy()
abs_df = abs_df.with_columns(
    pl.Series("paper_id", old_to_new_paper[old_abs_pids])
).sort("paper_id")
abs_df.write_parquet(out_abstracts)
log.info(f"  Abstracts: {len(abs_df):,}")
log.info(f"  Saved {out_abstracts}")

# =========================================================================
# 8. Source table — keep sources that appear in LCC papers
# =========================================================================
log.info("Filtering source table...")
source_df = pl.read_csv(
    in_source_table,
    schema_overrides={"issn_l": pl.String, "type": pl.String},
)
used_sources = set(lcc_df.get_column("source_id").to_list()) - {-1}
source_df = source_df.filter(pl.col("source_id").is_in(list(used_sources)))
source_df.write_csv(out_source_table)
log.info(f"  Sources: {len(source_df):,}")
log.info(f"  Saved {out_source_table}")

# =========================================================================
# 9. Registry — rebuild with new IDs
# =========================================================================
log.info("Building filtered registry...")
# Paper: openalex_id sorted for binary search lookup
oa_ids = lcc_df.get_column("openalex_id").to_numpy().astype(np.int64)
sort_order = np.argsort(oa_ids)
oa_ids_sorted = oa_ids[sort_order]

# Author: openalex_author_id
oa_author_ids = author_df.get_column("openalex_author_id").to_numpy().astype(np.int64)
new_author_ids = author_df.get_column("author_id").to_numpy().astype(np.int64)

sync_date = datetime.now().strftime("%Y-%m-%d")

np.savez(
    out_registry,
    oa_ids_sorted=oa_ids_sorted,
    oa_author_ids=oa_author_ids,
    author_ids=new_author_ids,
    n_papers=np.array([n_lcc]),
    n_authors=np.array([n_authors_new]),
    sync_date=np.array([sync_date]),
)
log.info(f"  Papers: {n_lcc:,}")
log.info(f"  Authors: {n_authors_new:,}")
log.info(f"  Saved {out_registry}")

log.info("LCC filtering complete.")
