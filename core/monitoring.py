"""Evidently AI wrapper for drift detection."""


def detect_embedding_drift(reference_embeddings, current_embeddings) -> dict:
    """
    Detect concept drift in face embeddings.

    Compares distribution of embeddings from reference (training)
    vs current (production) data.

    Returns:
        dict with drift detected flag and statistics.
    """
    raise NotImplementedError


def generate_drift_report(reference_embeddings, current_embeddings, output_path: str) -> None:
    """Generate HTML drift report using Evidently AI."""
    raise NotImplementedError
