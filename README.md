# openalex-dataset

Snakemake pipeline that downloads an [OpenAlex](https://openalex.org/) snapshot from S3 and preprocesses it into structured tables.

## Outputs

The pipeline produces two sets of outputs:

- **`preprocessed_unfiltered/`** — all papers matching the allowed types, with an `is_connected` flag
- **`preprocessed/`** — filtered to the largest weakly connected component (LCC) of the citation network, with contiguous re-indexed IDs

| File | Description |
|------|-------------|
| `paper_table.csv` | Paper metadata (paper_id, openalex_id, title, year, doi, type, source_id, is_connected, ...) |
| `citation_net.npz` | Citation network (scipy sparse CSR matrix) |
| `author_table.csv` | Author metadata (author_id, openalex_author_id, name, orcid) |
| `paper_author_net.npz` | Paper–author bipartite network (scipy sparse CSR) |
| `category_table.csv` | Field/subfield categories |
| `paper_category_table.csv` | Paper–category assignments |
| `abstracts.parquet` | Paper abstracts (Parquet) |
| `source_table.csv` | Journal/venue metadata |
| `openalex_registry.npz` | OpenAlex ID ↔ paper_id mappings |

In the **unfiltered** output, `is_connected` in `paper_table.csv` indicates whether a paper belongs to the LCC (1 = yes, 0 = no). In the **filtered** output, all papers have `is_connected = 1` and IDs are re-mapped to be contiguous starting from 0.

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create environment and install dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install numpy scipy polars pyarrow snakemake
```

### 3. Install AWS CLI (for downloading the snapshot)

```bash
uv pip install awscli
```

### 4. Configure paths

Edit `config.yaml`:

```yaml
raw_dir: "/data/datasets/openalex/raw"        # Where to download/find the snapshot
output_dir: "/data/datasets/openalex/preprocessed"  # Where to write outputs
```

The pipeline will also create a sibling `preprocessed_unfiltered/` directory alongside `output_dir`.

## Usage

```bash
# Dry run
snakemake -n

# Run full pipeline
snakemake --cores all

# Download snapshot only
snakemake download_openalex

# Skip download if snapshot already exists
touch /data/datasets/openalex/raw/.sync_complete
snakemake --cores all
```

## Pipeline

```
S3 snapshot
  → pass1_build_index         Scan all works, assign paper_ids (sorted by OpenAlex ID)
  → pass2_extract_data        Stream works, extract 7 temp files
  → build_citation_net    ┐
  → build_paper_table     │
  → build_author_data     ├── Convert temp files to final outputs (parallel)
  → build_category_data   │
  → build_abstracts       │
  → build_source_table    ┘
  → save_unfiltered_registry   Save ID mappings for unfiltered data
  → filter_to_lcc              Filter to LCC, re-index all IDs, write final outputs
```

**Pass 1** assigns `paper_id = position in OpenAlex-ID–sorted order`, filtering to allowed work types (article, book-chapter, preprint, review, letter, book-section). **Pass 2** uses binary search on the sorted index for O(log N) citation reference resolution.

**filter_to_lcc** keeps only papers in the largest weakly connected component, then re-maps paper, author, and category IDs to contiguous ranges and rebuilds all output files.
