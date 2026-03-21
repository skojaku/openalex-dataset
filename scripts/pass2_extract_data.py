"""Pass 2: Stream OpenAlex works and extract all data into temp files.

Loads the paper_index from Pass 1, then streams all works again.
For each work, looks up the paper_id and writes:
  - paper_metadata.csv: paper metadata (one row per paper, includes source_id)
  - citation_edges.bin: (paper_id, ref_paper_id) int32 pairs
  - authorship_edges.bin: (paper_id, author_id) int32 pairs
  - author_names.csv: (author_id, oa_author_id, name, orcid)
  - paper_topics.csv: (paper_id, field_id, field_name, subfield_id, subfield_name, sequence)
  - abstracts.csv: (paper_id, abstract)
  - source_names.csv: (source_id, openalex_source_id, display_name, issn_l, type)
"""

import csv
import gzip
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from openalex_utils import (
    BinaryEdgeWriter,
    compute_frac_year,
    lookup_ids_batch,
    parse_openalex_id,
    reconstruct_abstract,
    setup_logging,
    stream_works,
)

log = setup_logging(__name__)

KEEP_TYPES = {"article", "book-chapter", "preprint", "review", "letter", "book-section"}

# --- Snakemake or standalone ---
if "snakemake" in dir():
    snapshot_dir = snakemake.params["snapshot_dir"]
    paper_index_file = snakemake.input["paper_index"]
    metadata_file = snakemake.output["metadata"]
    cit_edges_file = snakemake.output["cit_edges"]
    auth_edges_file = snakemake.output["auth_edges"]
    auth_names_file = snakemake.output["auth_names"]
    topics_file = snakemake.output["topics"]
    abstracts_file = snakemake.output["abstracts"]
    source_names_file = snakemake.output["source_names"]
else:
    snapshot_dir = "/data/raw/openalex"
    paper_index_file = "paper_index.npz"
    metadata_file = "paper_metadata.csv.gz"
    cit_edges_file = "citation_edges.bin"
    auth_edges_file = "authorship_edges.bin"
    auth_names_file = "author_names.csv.gz"
    topics_file = "paper_topics.csv.gz"
    abstracts_file = "abstracts.csv.gz"
    source_names_file = "source_names.csv.gz"


# --- Load paper index ---
log.info("Loading paper index...")
idx = np.load(paper_index_file)
oa_ids_sorted = idx["oa_ids_sorted"]
n_papers = int(idx["n_papers"][0])
log.info(f"  {n_papers:,} papers in index.")


def lookup_single(oa_id):
    """Look up a single OpenAlex ID -> paper_id. Returns -1 if not found."""
    pos = np.searchsorted(oa_ids_sorted, oa_id)
    if pos < len(oa_ids_sorted) and oa_ids_sorted[pos] == oa_id:
        return int(pos)
    return -1


# --- Open output files ---
os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

def _open_csv_writer(filepath):
    """Open a gzip-compressed CSV writer. Returns (file_handle, csv_writer)."""
    fh = gzip.open(filepath, "wt", encoding="utf-8", newline="")
    return fh, csv.writer(fh)


meta_fh, meta_writer = _open_csv_writer(metadata_file)
meta_writer.writerow([
    "paper_id", "openalex_id", "title", "year", "doi", "type",
    "publication_date", "frac_year", "cited_by_count", "language", "source_id",
])

cit_writer = BinaryEdgeWriter(cit_edges_file)
auth_writer = BinaryEdgeWriter(auth_edges_file)

auth_fh, auth_csv = _open_csv_writer(auth_names_file)
auth_csv.writerow(["author_id", "openalex_author_id", "name", "orcid"])

topics_fh, topics_writer = _open_csv_writer(topics_file)
topics_writer.writerow([
    "paper_id", "field_id", "field_name", "subfield_id", "subfield_name", "sequence",
])

abs_fh, abs_writer = _open_csv_writer(abstracts_file)
abs_writer.writerow(["paper_id", "abstract"])

source_fh, source_csv = _open_csv_writer(source_names_file)
source_csv.writerow(["source_id", "openalex_source_id", "display_name", "issn_l", "type"])


# --- Author ID assignment ---
# Maps openalex_author_id -> sequential author_id
author_id_map = {}
next_author_id = 0

# --- Source ID assignment ---
# Maps openalex_source_id -> sequential source_id
source_id_map = {}
next_source_id = 0


# --- Stream works and extract data ---
count = 0
n_cit_edges = 0
n_auth_edges = 0
n_topics = 0
n_abstracts = 0

# Batch buffer for citation edge lookups
CIT_BATCH_SIZE = 10000
cit_batch_paper_ids = []
cit_batch_ref_oa_ids = []

log.info("Pass 2: Extracting data from works...")
for work in stream_works(snapshot_dir):
    oa_id = parse_openalex_id(work.get("id"))
    if oa_id < 0:
        continue
    paper_id = lookup_single(oa_id)
    if paper_id < 0:
        continue

    # --- Metadata ---
    year = work.get("publication_year", 0)
    pub_date = work.get("publication_date", "")
    frac_year = compute_frac_year(pub_date, year if year else 0)
    doi = work.get("doi", "")
    title = work.get("title", "")
    work_type = work.get("type", "")
    if work_type not in KEEP_TYPES:
        continue
    cited_by_count = work.get("cited_by_count", 0)
    language = work.get("language", "")

    # --- Source (journal/venue) ---
    source_id = -1
    primary_loc = work.get("primary_location") or {}
    source_info = primary_loc.get("source") or {}
    if source_info:
        oa_source_id = parse_openalex_id(source_info.get("id"))
        if oa_source_id >= 0:
            if oa_source_id not in source_id_map:
                sid = next_source_id
                source_id_map[oa_source_id] = sid
                next_source_id += 1
                source_csv.writerow([
                    sid,
                    oa_source_id,
                    source_info.get("display_name", ""),
                    source_info.get("issn_l", "") or "",
                    source_info.get("type", "") or "",
                ])
            source_id = source_id_map[oa_source_id]

    meta_writer.writerow([
        paper_id, oa_id, title, year, doi, work_type,
        pub_date, frac_year, cited_by_count, language, source_id,
    ])

    # --- Citations ---
    ref_works = work.get("referenced_works", [])
    if ref_works:
        for ref_url in ref_works:
            ref_oa_id = parse_openalex_id(ref_url)
            if ref_oa_id < 0:
                continue
            cit_batch_paper_ids.append(paper_id)
            cit_batch_ref_oa_ids.append(ref_oa_id)

        # Flush citation batch if large enough
        if len(cit_batch_paper_ids) >= CIT_BATCH_SIZE:
            ref_paper_ids = lookup_ids_batch(
                cit_batch_ref_oa_ids, oa_ids_sorted
            )
            mask = ref_paper_ids >= 0
            if mask.any():
                srcs = np.array(cit_batch_paper_ids, dtype=np.int32)[mask]
                dsts = ref_paper_ids[mask].astype(np.int32)
                cit_writer.write_batch(srcs, dsts)
                n_cit_edges += int(mask.sum())
            cit_batch_paper_ids.clear()
            cit_batch_ref_oa_ids.clear()

    # --- Authorships ---
    authorships = work.get("authorships", [])
    for authorship in authorships:
        author_info = authorship.get("author", {})
        if not author_info:
            continue
        oa_author_id = parse_openalex_id(author_info.get("id"))
        if oa_author_id < 0:
            continue

        if oa_author_id not in author_id_map:
            aid = next_author_id
            author_id_map[oa_author_id] = aid
            next_author_id += 1
            # Write author info
            name = author_info.get("display_name", "")
            orcid = author_info.get("orcid", "")
            auth_csv.writerow([aid, oa_author_id, name, orcid or ""])
        else:
            aid = author_id_map[oa_author_id]

        auth_writer.write(paper_id, aid)
        n_auth_edges += 1

    # --- Topics (field / subfield) ---
    topics = work.get("topics", [])
    for seq, topic in enumerate(topics):
        # Each topic has a field (top-level) and subfield
        field = topic.get("field", {})
        subfield = topic.get("subfield", {})
        if not field:
            continue
        field_id = parse_openalex_id(field.get("id"))
        field_name = field.get("display_name", "")
        subfield_id = parse_openalex_id(subfield.get("id")) if subfield else -1
        subfield_name = subfield.get("display_name", "") if subfield else ""
        topics_writer.writerow([
            paper_id, field_id, field_name, subfield_id, subfield_name, seq,
        ])
        n_topics += 1

    # --- Abstract ---
    abstract_inv = work.get("abstract_inverted_index")
    if abstract_inv:
        abstract_text = reconstruct_abstract(abstract_inv)
        if abstract_text:
            abs_writer.writerow([paper_id, abstract_text])
            n_abstracts += 1

    count += 1
    if count % 10_000_000 == 0:
        log.info(f"  Processed {count:,} works...")

# --- Flush remaining citation batch ---
if cit_batch_paper_ids:
    ref_paper_ids = lookup_ids_batch(
        cit_batch_ref_oa_ids, oa_ids_sorted
    )
    mask = ref_paper_ids >= 0
    if mask.any():
        srcs = np.array(cit_batch_paper_ids, dtype=np.int32)[mask]
        dsts = ref_paper_ids[mask].astype(np.int32)
        cit_writer.write_batch(srcs, dsts)
        n_cit_edges += int(mask.sum())

# --- Close files ---
meta_fh.close()
cit_writer.close()
auth_writer.close()
auth_fh.close()
topics_fh.close()
abs_fh.close()
source_fh.close()

log.info(f"Pass 2 complete:")
log.info(f"  Papers processed: {count:,}")
log.info(f"  Citation edges: {n_cit_edges:,}")
log.info(f"  Authorship edges: {n_auth_edges:,}")
log.info(f"  Authors: {next_author_id:,}")
log.info(f"  Sources: {next_source_id:,}")
log.info(f"  Topic assignments: {n_topics:,}")
log.info(f"  Abstracts: {n_abstracts:,}")
