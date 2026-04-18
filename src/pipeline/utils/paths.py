from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
CHUNKS = DATA / "chunks"
EMBEDDINGS = DATA / "embeddings"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
METRICS = OUTPUTS / "metrics"
LOGS = ROOT / "logs"


def current_pdf() -> Path | None:
    """Return the first PDF found in data/raw/, or None if the dir is empty.
    The pipeline processes whichever PDF is present — upload replaces it."""
    if not RAW.exists():
        return None
    pdfs = sorted(RAW.glob("*.pdf"))
    return pdfs[0] if pdfs else None
