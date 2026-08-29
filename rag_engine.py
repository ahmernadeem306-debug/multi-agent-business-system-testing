"""
rag_engine.py
=============
Production-grade Retrieval-Augmented Generation (RAG) pipeline for the
BizAgent platform's policy knowledge base.

Replaces the flat-file `sop_policy.txt` approach with a real PDF ->
vector-store pipeline:

    ./policies/*.pdf   --(pypdf)-->   raw page text
                        --(RecursiveCharacterTextSplitter)-->   chunks
                        --(embeddings)-->   vectors
                        --(Chroma, persisted)-->   ./chroma_db/

NO HARDCODED POLICY STRINGS
----------------------------
This module contains no embedded policy text of any kind. Every chunk
returned by `query_knowledge_base()` is extracted verbatim, at runtime,
from whatever PDF files an operator drops into ./policies/. If that
folder is empty, the knowledge base is empty -- there is no fallback
sample text baked into this file.

Usage
-----
    python rag_engine.py
        -> scans ./policies/, (re)builds ./chroma_db/ from scratch

    from rag_engine import query_knowledge_base
    hits = query_knowledge_base("What is the escalation policy for reorders?")
    for hit in hits:
        print(hit["source_file"], hit["page"], hit["score"])
        print(hit["content"])
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --------------------------------------------------------------------------- 
# Configuration (pipeline parameters, NOT policy content)
# --------------------------------------------------------------------------- 

BASE_DIR = Path(__file__).resolve().parent
POLICIES_DIR = BASE_DIR / "policies"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "bizagent_sop_policies"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

DEFAULT_TOP_K = 4
# Chroma's cosine "distance" -- lower is more similar. Hits above this
# distance are considered too weak to be a genuine policy match.
MAX_RELEVANT_DISTANCE = 0.8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("bizagent.rag_engine")


# --------------------------------------------------------------------------- 
# Embedding model
# --------------------------------------------------------------------------- 

def get_embedding_function() -> Embeddings:
    """
    Return the embedding model used for both indexing and querying.

    Isolated in its own function so the embedding backend can be swapped
    (e.g. for an OpenAI or Bedrock embedding model in a cloud deployment)
    without touching any ingestion or retrieval logic below.

    Defaults to FastEmbed (ONNX-based, runs fully locally, no API key
    required) so the knowledge base works out of the box in an
    air-gapped or offline operational environment.
    """
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# --------------------------------------------------------------------------- 
# PDF discovery + extraction
# --------------------------------------------------------------------------- 

def discover_pdf_files(policies_dir: Path = POLICIES_DIR) -> list[Path]:
    """Return every .pdf file found under `policies_dir` (recursive)."""
    if not policies_dir.exists():
        logger.warning("Policies directory '%s' does not exist yet.", policies_dir)
        return []
    pdf_files = sorted(policies_dir.rglob("*.pdf"))
    logger.info("Discovered %d PDF file(s) in '%s'.", len(pdf_files), policies_dir)
    return pdf_files


def extract_documents_from_pdf(pdf_path: Path) -> list[Document]:
    """
    Extract text page-by-page from a single PDF and return one
    LangChain `Document` per non-empty page, tagged with source
    filename and page number metadata for traceable citations.
    """
    documents: list[Document] = []
    try:
        reader = PdfReader(str(pdf_path))
    except (PdfReadError, OSError) as exc:
        logger.error("Could not open '%s': %s -- skipping this file.", pdf_path.name, exc)
        return documents

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pypdf can raise various parser errors on malformed pages
            logger.warning(
                "Failed to extract text from '%s' page %d: %s -- skipping page.",
                pdf_path.name, page_number, exc,
            )
            continue

        page_text = page_text.strip()
        if not page_text:
            continue

        documents.append(
            Document(
                page_content=page_text,
                metadata={
                    "source_file": pdf_path.name,
                    "page": page_number,
                },
            )
        )

    logger.info("Extracted %d non-empty page(s) from '%s'.", len(documents), pdf_path.name)
    return documents


def load_all_policy_documents(policies_dir: Path = POLICIES_DIR) -> list[Document]:
    """Extract page-level Documents from every PDF in `policies_dir`."""
    all_documents: list[Document] = []
    for pdf_path in discover_pdf_files(policies_dir):
        all_documents.extend(extract_documents_from_pdf(pdf_path))
    return all_documents


# --------------------------------------------------------------------------- 
# Chunking
# --------------------------------------------------------------------------- 

def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split page-level Documents into overlapping chunks sized for
    embedding, preserving each chunk's source_file/page metadata so
    every retrieval result can be traced back to an exact PDF location.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d page(s) into %d chunk(s).", len(documents), len(chunks))
    return chunks


# --------------------------------------------------------------------------- 
# Vector store (build + load)
# --------------------------------------------------------------------------- 

def _get_chroma_store(
    persist_directory: Path = CHROMA_PERSIST_DIR,
    embedding_function: Optional[Embeddings] = None,
) -> Chroma:
    """Open (or lazily create) the persisted Chroma collection on disk."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function or get_embedding_function(),
        persist_directory=str(persist_directory),
    )


def build_vector_store(
    policies_dir: Path = POLICIES_DIR,
    persist_directory: Path = CHROMA_PERSIST_DIR,
    force_rebuild: bool = True,
) -> int:
    """
    Full ingestion pipeline: discover PDFs -> extract -> chunk -> embed
    -> persist to ChromaDB.

    Parameters
    ----------
    force_rebuild : bool
        If True (default), any existing persisted collection at
        `persist_directory` is wiped first. This keeps re-runs
        idempotent -- editing or removing a PDF and re-running this
        function won't leave stale or duplicated chunks behind.

    Returns
    -------
    int
        The number of chunks written to the vector store.
    """
    if force_rebuild and persist_directory.exists():
        logger.info("force_rebuild=True -- clearing existing store at '%s'.", persist_directory)
        shutil.rmtree(persist_directory)

    raw_documents = load_all_policy_documents(policies_dir)
    if not raw_documents:
        logger.warning(
            "No extractable text found in '%s'. The knowledge base will be empty "
            "until PDF policy documents are added there.",
            policies_dir,
        )
        # Still stand up an empty, valid Chroma collection so downstream
        # code can query it (and get an empty, well-formed result) rather
        # than crashing on a missing persist directory.
        persist_directory.mkdir(parents=True, exist_ok=True)
        _get_chroma_store(persist_directory)
        return 0

    chunks = chunk_documents(raw_documents)
    embedding_function = get_embedding_function()

    logger.info("Embedding and persisting %d chunk(s) to '%s'...", len(chunks), persist_directory)
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
    )
    logger.info("Vector store build complete: %d chunk(s) indexed.", len(chunks))
    return len(chunks)


# --------------------------------------------------------------------------- 
# Retrieval interface
# --------------------------------------------------------------------------- 

@dataclass(frozen=True)
class PolicyMatch:
    content: str
    source_file: str
    page: int
    score: float  # similarity score in [0, 1]; higher = more relevant


def query_knowledge_base(
    user_query: str,
    k: int = DEFAULT_TOP_K,
    persist_directory: Path = CHROMA_PERSIST_DIR,
    max_distance: float = MAX_RELEVANT_DISTANCE,
) -> list[dict]:
    """
    Search the persisted ChromaDB knowledge base for the passage(s)
    most relevant to `user_query` and return exact, verbatim excerpts
    from the source PDFs -- never paraphrased or generated text.

    Parameters
    ----------
    user_query : str
        Natural-language question or lookup term.
    k : int
        Maximum number of chunks to return.
    persist_directory : Path
        Location of the persisted Chroma collection (defaults to
        ./chroma_db, matching build_vector_store()).
    max_distance : float
        Chroma L2/cosine distance cutoff; results weaker than this are
        dropped rather than returned as a low-confidence false match.

    Returns
    -------
    list[dict]
        Each dict has keys: 'content', 'source_file', 'page', 'score'.
        Empty list if the knowledge base has no relevant match (or has
        not been built yet).
    """
    if not user_query or not user_query.strip():
        raise ValueError("user_query must be a non-empty string.")

    if not persist_directory.exists():
        logger.warning(
            "No vector store found at '%s'. Run `python rag_engine.py` first to "
            "index the PDFs in '%s'.",
            persist_directory, POLICIES_DIR,
        )
        return []

    store = _get_chroma_store(persist_directory)

    try:
        results = store.similarity_search_with_score(user_query, k=k)
    except Exception:
        logger.exception("Similarity search failed against the ChromaDB store.")
        raise

    matches: list[PolicyMatch] = []
    for document, distance in results:
        if distance > max_distance:
            continue
        similarity = max(0.0, 1.0 - distance)
        matches.append(
            PolicyMatch(
                content=document.page_content,
                source_file=document.metadata.get("source_file", "unknown"),
                page=document.metadata.get("page", -1),
                score=round(similarity, 4),
            )
        )

    logger.info(
        "Query %r returned %d relevant chunk(s) (of %d retrieved).",
        user_query, len(matches), len(results),
    )
    return [match.__dict__ for match in matches]


# --------------------------------------------------------------------------- 
# Isolated initialization entry point
# --------------------------------------------------------------------------- 

if __name__ == "__main__":
    logger.info("Initializing BizAgent policy knowledge base...")
    logger.info("Source folder : %s", POLICIES_DIR)
    logger.info("Vector store  : %s", CHROMA_PERSIST_DIR)

    POLICIES_DIR.mkdir(parents=True, exist_ok=True)

    chunk_count = build_vector_store(
        policies_dir=POLICIES_DIR,
        persist_directory=CHROMA_PERSIST_DIR,
        force_rebuild=True,
    )

    if chunk_count == 0:
        logger.warning(
            "No PDFs were indexed. Add policy documents to '%s' and re-run this script.",
            POLICIES_DIR,
        )
    else:
        logger.info("Knowledge base ready: %d chunk(s) indexed from '%s'.", chunk_count, POLICIES_DIR)
        sample = query_knowledge_base("policy", k=1)
        if sample:
            logger.info(
                "Sanity check retrieval succeeded (matched '%s', page %d).",
                sample[0]["source_file"], sample[0]["page"],
            )
