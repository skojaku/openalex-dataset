"""Shared utilities for OpenAlex preprocessing pipeline."""

import gzip
import json
import os
import struct
from glob import glob
from pathlib import Path

import numpy as np


def parse_openalex_id(url):
    """Extract numeric ID from OpenAlex URL.

    "https://openalex.org/W2741809807" -> 2741809807
    """
    if url is None:
        return -1
    s = url.rsplit("/", 1)[-1]
    # Strip the entity prefix (W, A, S, I, C, T, etc.)
    return int(s[1:]) if s[0].isalpha() else int(s)


def reconstruct_abstract(inverted_index):
    """Reconstruct abstract text from OpenAlex inverted index format.

    {"word": [0, 5], "other": [1]} -> "word other ... word ..."
    """
    if not inverted_index:
        return ""
    # Find total length
    max_pos = -1
    for positions in inverted_index.values():
        for pos in positions:
            if pos > max_pos:
                max_pos = pos
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)


def compute_frac_year(pub_date, year):
    """Compute fractional year from publication date string.

    "2018-02-13", 2018 -> 2018.0833...
    """
    if pub_date and isinstance(pub_date, str) and len(pub_date) >= 7:
        try:
            month = int(pub_date[5:7])
            return year + (month - 1) / 12.0
        except (ValueError, IndexError):
            pass
    return float(year)


def stream_works(snapshot_dir, partitions=None):
    """Yield parsed JSON objects from OpenAlex works snapshot.

    Reads data/works/updated_date=YYYY-MM-DD/*.gz files.
    If FILES_PER_ENTITY env var is set, limits files per partition (for testing).
    """
    works_dir = os.path.join(snapshot_dir, "data", "works")
    if partitions is not None:
        partition_dirs = [os.path.join(works_dir, p) for p in partitions]
    else:
        partition_dirs = sorted(glob(os.path.join(works_dir, "updated_date=*")))

    files_per_entity = int(os.environ.get("FILES_PER_ENTITY", 0))

    for pdir in partition_dirs:
        if not os.path.isdir(pdir):
            continue
        gz_files = sorted(glob(os.path.join(pdir, "*.gz")))
        if files_per_entity > 0:
            gz_files = gz_files[:files_per_entity]
        for gz_file in gz_files:
            with gzip.open(gz_file, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


def lookup_ids_batch(query_ids, sorted_oa_ids):
    """Vectorized ID mapping using binary search.

    Given query OpenAlex IDs, look them up in sorted_oa_ids and return
    the position (= paper_id). Returns -1 for IDs not found.
    """
    query_ids = np.asarray(query_ids, dtype=np.int64)
    indices = np.searchsorted(sorted_oa_ids, query_ids)
    # Clamp indices to valid range for comparison
    valid = indices < len(sorted_oa_ids)
    indices_clamped = np.where(valid, indices, 0)
    matches = valid & (sorted_oa_ids[indices_clamped] == query_ids)
    result = np.full(len(query_ids), -1, dtype=np.int64)
    result[matches] = indices_clamped[matches]
    return result


class BinaryEdgeWriter:
    """Write (int32, int32) edge pairs to a binary file. 8 bytes per edge."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._file = open(filepath, "wb")
        self._packer = struct.Struct("<ii")  # two little-endian int32

    def write(self, src, dst):
        self._file.write(self._packer.pack(src, dst))

    def write_batch(self, srcs, dsts):
        srcs = np.asarray(srcs, dtype=np.int32)
        dsts = np.asarray(dsts, dtype=np.int32)
        data = np.column_stack([srcs, dsts])
        self._file.write(data.tobytes())

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BinaryEdgeReader:
    """Read (int32, int32) edge pairs from a binary file in chunks."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._filesize = os.path.getsize(filepath)
        self.n_edges = self._filesize // 8

    def iter_chunks(self, chunk_size=1_000_000):
        """Yield (src_array, dst_array) chunks."""
        with open(self.filepath, "rb") as f:
            remaining = self.n_edges
            while remaining > 0:
                n = min(chunk_size, remaining)
                data = f.read(n * 8)
                arr = np.frombuffer(data, dtype=np.int32).reshape(-1, 2)
                yield arr[:, 0], arr[:, 1]
                remaining -= n

    def read_all(self):
        """Read all edges at once. Returns (src_array, dst_array)."""
        with open(self.filepath, "rb") as f:
            data = f.read()
        if len(data) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        arr = np.frombuffer(data, dtype=np.int32).reshape(-1, 2)
        return arr[:, 0].copy(), arr[:, 1].copy()
