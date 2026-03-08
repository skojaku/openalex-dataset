# openalex-dataset

Snakemake pipeline that downloads an [OpenAlex](https://openalex.org/) snapshot from S3 and preprocesses it into structured tables.

## Outputs

| File | Description |
|------|-------------|
| `paper_table.csv` | Paper metadata (paper_id, openalex_id, title, year, doi, type, source_id, is_connected, ...) |
| `citation_net.npz` | Citation network (scipy sparse CSR matrix) |
| `author_table.csv` | Author metadata (author_id, openalex_author_id, name, orcid) |
| `paper_author_net.npz` | Paper-author bipartite network (scipy sparse CSR) |
| `category_table.csv` | Field/subfield categories |
| `paper_category_table.csv` | Paper-category assignments |
| `abstracts.parquet` | Paper abstracts (PyArrow parquet) |
| `source_table.csv` | Journal/venue metadata |
| `openalex_registry.npz` | OpenAlex ID <-> paper_id mappings |

The `is_connected` column in `paper_table.csv` indicates whether a paper belongs to the largest weakly connected component of the citation network (1 = yes, 0 = no).

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
  -> pass1_build_index    (scan all works, assign paper_ids)
  -> pass2_extract_data   (stream works, extract 7 temp files)
  -> build_*              (convert temp files to final outputs, in parallel)
  -> save_registry        (save ID mappings)
```

Pass 1 assigns `paper_id = position in oa_id-sorted order`. Pass 2 uses binary search on the sorted index for citation reference resolution.
