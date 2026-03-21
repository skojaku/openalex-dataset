"""
OpenAlex Dataset Pipeline

Downloads an OpenAlex snapshot from S3 and preprocesses it into structured
tables (papers, authors, citations, categories, abstracts, sources).

Two output directories:
  - preprocessed_unfiltered/: all papers (with is_connected flag)
  - preprocessed/: filtered to largest connected component only

Usage:
    snakemake --cores all          # Run full pipeline
    snakemake -n                   # Dry run
"""

from os.path import join as j

configfile: "config.yaml"

# =============================================================================
# Directories
# =============================================================================

RAW_DIR = config.get("raw_dir", "/data/datasets/openalex/raw")
OUTPUT_DIR = config.get("output_dir", "/data/datasets/openalex/preprocessed")
UNFILTERED_DIR = OUTPUT_DIR + "_unfiltered"
TEMP_DIR = j(UNFILTERED_DIR, "temp")

# =============================================================================
# Sentinel
# =============================================================================

SYNC_SENTINEL = j(RAW_DIR, ".sync_complete")

# =============================================================================
# Temp intermediate files (auto-deleted after use)
# =============================================================================

PAPER_INDEX = j(TEMP_DIR, "paper_index.npz")
TEMP_METADATA = j(TEMP_DIR, "paper_metadata.csv.gz")
TEMP_CIT_EDGES = j(TEMP_DIR, "citation_edges.bin")
TEMP_AUTH_EDGES = j(TEMP_DIR, "authorship_edges.bin")
TEMP_AUTH_NAMES = j(TEMP_DIR, "author_names.csv.gz")
TEMP_TOPICS = j(TEMP_DIR, "paper_topics.csv.gz")
TEMP_ABSTRACTS = j(TEMP_DIR, "abstracts.csv.gz")
TEMP_SOURCE_NAMES = j(TEMP_DIR, "source_names.csv.gz")

# =============================================================================
# Unfiltered outputs
# =============================================================================

UF_PAPER_TABLE = j(UNFILTERED_DIR, "paper_table.csv")
UF_CITATION_NET = j(UNFILTERED_DIR, "citation_net.npz")
UF_AUTHOR_TABLE = j(UNFILTERED_DIR, "author_table.csv")
UF_PAPER_AUTHOR_NET = j(UNFILTERED_DIR, "paper_author_net.npz")
UF_CATEGORY_TABLE = j(UNFILTERED_DIR, "category_table.csv")
UF_PAPER_CATEGORY_TABLE = j(UNFILTERED_DIR, "paper_category_table.csv")
UF_ABSTRACTS = j(UNFILTERED_DIR, "abstracts.parquet")
UF_SOURCE_TABLE = j(UNFILTERED_DIR, "source_table.csv")
UF_REGISTRY = j(UNFILTERED_DIR, "openalex_registry.npz")

# =============================================================================
# Final filtered outputs
# =============================================================================

PAPER_TABLE = j(OUTPUT_DIR, "paper_table.csv")
CITATION_NET = j(OUTPUT_DIR, "citation_net.npz")
AUTHOR_TABLE = j(OUTPUT_DIR, "author_table.csv")
PAPER_AUTHOR_NET = j(OUTPUT_DIR, "paper_author_net.npz")
CATEGORY_TABLE = j(OUTPUT_DIR, "category_table.csv")
PAPER_CATEGORY_TABLE = j(OUTPUT_DIR, "paper_category_table.csv")
ABSTRACTS = j(OUTPUT_DIR, "abstracts.parquet")
SOURCE_TABLE = j(OUTPUT_DIR, "source_table.csv")
REGISTRY = j(OUTPUT_DIR, "openalex_registry.npz")

# =============================================================================
# Rules
# =============================================================================

rule all:
    input:
        PAPER_TABLE,
        CITATION_NET,
        AUTHOR_TABLE,
        PAPER_AUTHOR_NET,
        CATEGORY_TABLE,
        PAPER_CATEGORY_TABLE,
        ABSTRACTS,
        SOURCE_TABLE,
        REGISTRY,


rule download_openalex:
    output:
        sentinel = SYNC_SENTINEL,
    params:
        snapshot_dir = RAW_DIR,
    shell:
        """
        aws s3 sync "s3://openalex" "{params.snapshot_dir}" --no-sign-request
        touch "{output.sentinel}"
        """


rule pass1_build_index:
    input:
        sentinel = SYNC_SENTINEL,
    params:
        snapshot_dir = RAW_DIR,
    output:
        paper_index = temp(PAPER_INDEX),
    script:
        "scripts/pass1_build_index.py"


rule pass2_extract_data:
    input:
        sentinel = SYNC_SENTINEL,
        paper_index = PAPER_INDEX,
    params:
        snapshot_dir = RAW_DIR,
    output:
        metadata = temp(TEMP_METADATA),
        cit_edges = temp(TEMP_CIT_EDGES),
        auth_edges = temp(TEMP_AUTH_EDGES),
        auth_names = temp(TEMP_AUTH_NAMES),
        topics = temp(TEMP_TOPICS),
        abstracts = temp(TEMP_ABSTRACTS),
        source_names = temp(TEMP_SOURCE_NAMES),
    script:
        "scripts/pass2_extract_data.py"


rule build_citation_net:
    input:
        cit_edges = TEMP_CIT_EDGES,
        paper_index = PAPER_INDEX,
    output:
        citation_net = UF_CITATION_NET,
    script:
        "scripts/build_citation_net.py"


rule build_paper_table:
    input:
        metadata = TEMP_METADATA,
        paper_index = PAPER_INDEX,
        citation_net = UF_CITATION_NET,
        source_names = TEMP_SOURCE_NAMES,
    output:
        paper_table = UF_PAPER_TABLE,
    script:
        "scripts/build_paper_table.py"


rule build_author_data:
    input:
        auth_edges = TEMP_AUTH_EDGES,
        auth_names = TEMP_AUTH_NAMES,
        paper_index = PAPER_INDEX,
    output:
        author_table = UF_AUTHOR_TABLE,
        paper_author_net = UF_PAPER_AUTHOR_NET,
    script:
        "scripts/build_author_data.py"


rule build_category_data:
    input:
        topics = TEMP_TOPICS,
        paper_index = PAPER_INDEX,
    output:
        category_table = UF_CATEGORY_TABLE,
        paper_category_table = UF_PAPER_CATEGORY_TABLE,
    script:
        "scripts/build_category_data.py"


rule build_abstracts:
    input:
        abstracts = TEMP_ABSTRACTS,
    output:
        abstracts_parquet = UF_ABSTRACTS,
    script:
        "scripts/build_abstracts.py"


rule build_source_table:
    input:
        source_names = TEMP_SOURCE_NAMES,
    output:
        source_table = UF_SOURCE_TABLE,
    script:
        "scripts/build_source_table.py"


rule save_unfiltered_registry:
    input:
        paper_index = PAPER_INDEX,
        paper_table = UF_PAPER_TABLE,
        author_table = UF_AUTHOR_TABLE,
    output:
        registry = UF_REGISTRY,
    script:
        "scripts/save_registry.py"


rule filter_to_lcc:
    threads: 10
    resources:
        mem_mb = 200000,
    input:
        paper_table = UF_PAPER_TABLE,
        citation_net = UF_CITATION_NET,
        author_table = UF_AUTHOR_TABLE,
        paper_author_net = UF_PAPER_AUTHOR_NET,
        category_table = UF_CATEGORY_TABLE,
        paper_category_table = UF_PAPER_CATEGORY_TABLE,
        abstracts = UF_ABSTRACTS,
        source_table = UF_SOURCE_TABLE,
    output:
        paper_table = PAPER_TABLE,
        citation_net = CITATION_NET,
        author_table = AUTHOR_TABLE,
        paper_author_net = PAPER_AUTHOR_NET,
        category_table = CATEGORY_TABLE,
        paper_category_table = PAPER_CATEGORY_TABLE,
        abstracts = ABSTRACTS,
        source_table = SOURCE_TABLE,
        registry = REGISTRY,
    script:
        "scripts/filter_to_lcc.py"
