"""Filter a full preprocessed OpenAlex dataset to papers by Binghamton authors.

Selects every paper written by an author who was ever affiliated with
Binghamton University (OpenAlex institution I123946342), keeps the largest
weakly connected component of the citation network among those papers,
re-indexes all IDs to contiguous ranges, and writes the same files and schemas
as filter_to_lcc.py.

Usage: filter_to_binghamton.py <in_dir> <out_dir> <author_ids.txt>
  author_ids.txt: one numeric OpenAlex author id per line (A123 -> 123)
"""

import os
import sys
from datetime import datetime

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import setup_logging

log = setup_logging(__name__)

if "snakemake" in dir():
    base_in = snakemake.params["in_dir"]
    base_out = snakemake.params["out_dir"]
    in_author_ids = snakemake.input["author_ids"]
else:
    base_in, base_out, in_author_ids = sys.argv[1:4]

os.makedirs(base_out, exist_ok=True)
out = lambda f: f"{base_out}/{f}"
inp = lambda f: f"{base_in}/{f}"

# =========================================================================
# 1. Select authors and their papers
# =========================================================================
oa_author_ids = np.loadtxt(in_author_ids, dtype=np.int64)
author_df = pl.read_csv(
    inp("author_table.csv"), schema_overrides={"orcid": pl.String, "name": pl.String}
)
bing_authors = author_df.filter(pl.col("openalex_author_id").is_in(oa_author_ids.tolist()))
log.info(f"Binghamton authors: {len(oa_author_ids):,} in API, {len(bing_authors):,} in author_table")
bing_aids = bing_authors.get_column("author_id").to_numpy()

pa_net = sparse.load_npz(inp("paper_author_net.npz")).tocsc()
n_total, n_authors_old = pa_net.shape
sel = np.zeros(n_total, dtype=bool)
for a in bing_aids:
    sel[pa_net.indices[pa_net.indptr[a]:pa_net.indptr[a + 1]]] = True
pa_net = pa_net.tocsr()
sel_ids = np.where(sel)[0]
log.info(f"Papers by Binghamton authors: {len(sel_ids):,}")

# =========================================================================
# 2. Citation submatrix and largest weakly connected component
# =========================================================================
net = sparse.load_npz(inp("citation_net.npz"))
assert net.shape[0] == n_total, "citation_net / paper_author_net size mismatch"
sub = net[sel_ids][:, sel_ids].tocsr()
del net
n_comp, labels = connected_components(sub, directed=True, connection="weak")
lcc_label = np.bincount(labels).argmax()
keep_local = np.where(labels == lcc_label)[0]
old_paper_ids = sel_ids[keep_local]
n_lcc = len(old_paper_ids)
log.info(f"Components: {n_comp:,}; LCC papers: {n_lcc:,} / {len(sel_ids):,}")

old_to_new_paper = np.full(n_total, -1, dtype=np.int64)
old_to_new_paper[old_paper_ids] = np.arange(n_lcc, dtype=np.int64)

net_f = sub[keep_local][:, keep_local].tocsr()
sparse.save_npz(out("citation_net.npz"), net_f)
log.info(f"Citation edges: {net_f.nnz:,}")
del sub, net_f

# =========================================================================
# 3. Paper table
# =========================================================================
paper_df = pl.read_csv(
    inp("paper_table.csv"),
    schema_overrides={"doi": pl.String, "language": pl.String, "type": pl.String, "source_id": pl.Int64},
)
assert len(paper_df) == n_total and (paper_df.get_column("paper_id").to_numpy() == np.arange(n_total)).all()
lcc_df = paper_df[old_paper_ids].with_columns(pl.Series("paper_id", np.arange(n_lcc, dtype=np.int64)))
lcc_df.write_csv(out("paper_table.csv"))
del paper_df

# =========================================================================
# 4. Paper-author network and author table (all co-authors of kept papers)
# =========================================================================
pa_f = pa_net[old_paper_ids]
del pa_net
old_author_ids = np.where(np.asarray(pa_f.sum(axis=0)).ravel() > 0)[0]
pa_f = pa_f[:, old_author_ids].tocsr()
sparse.save_npz(out("paper_author_net.npz"), pa_f)
old_to_new_author = np.full(n_authors_old, -1, dtype=np.int64)
old_to_new_author[old_author_ids] = np.arange(len(old_author_ids), dtype=np.int64)

author_f = author_df.filter(pl.col("author_id").is_in(old_author_ids.tolist()))
author_f = author_f.with_columns(
    pl.Series("author_id", old_to_new_author[author_f.get_column("author_id").to_numpy()])
).sort("author_id")
author_f.write_csv(out("author_table.csv"))
log.info(f"Authors: {len(author_f):,}")

# =========================================================================
# 5. Categories
# =========================================================================
pcat = pl.read_csv(inp("paper_category_table.csv"))
pcat = pcat.filter(pl.Series(old_to_new_paper[pcat.get_column("paper_id").to_numpy()] >= 0))
pcat = pcat.with_columns(
    pl.Series("paper_id", old_to_new_paper[pcat.get_column("paper_id").to_numpy()])
).sort("paper_id", "sequence")
pcat.write_csv(out("paper_category_table.csv"))

cat = pl.read_csv(inp("category_table.csv"))
used = set(pcat.get_column("main_class_id").to_list()) | (set(pcat.get_column("sub_class_id").to_list()) - {-1})
cat.filter(pl.col("class_id").is_in(list(used))).write_csv(out("category_table.csv"))

# =========================================================================
# 6. Abstracts (streamed; source is large)
# =========================================================================
abs_df = (
    pl.scan_parquet(inp("abstracts.parquet"))
    .filter(pl.col("paper_id").is_in(old_paper_ids.tolist()))
    .collect()
)
abs_df = abs_df.with_columns(
    pl.Series("paper_id", old_to_new_paper[abs_df.get_column("paper_id").to_numpy()])
).sort("paper_id")
abs_df.write_parquet(out("abstracts.parquet"))
log.info(f"Abstracts: {len(abs_df):,}")

# =========================================================================
# 7. Sources
# =========================================================================
source_df = pl.read_csv(inp("source_table.csv"), schema_overrides={"issn_l": pl.String, "type": pl.String})
used_src = set(lcc_df.get_column("source_id").to_list()) - {-1}
source_df.filter(pl.col("source_id").is_in(list(used_src))).write_csv(out("source_table.csv"))

# =========================================================================
# 8. Registry
# =========================================================================
oa_ids = lcc_df.get_column("openalex_id").to_numpy().astype(np.int64)
np.savez(
    out("openalex_registry.npz"),
    oa_ids_sorted=oa_ids[np.argsort(oa_ids)],
    oa_author_ids=author_f.get_column("openalex_author_id").to_numpy().astype(np.int64),
    author_ids=author_f.get_column("author_id").to_numpy().astype(np.int64),
    n_papers=np.array([n_lcc]),
    n_authors=np.array([len(author_f)]),
    sync_date=np.array([datetime.now().strftime("%Y-%m-%d")]),
)
log.info("Done.")
