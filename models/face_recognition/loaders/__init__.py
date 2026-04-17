"""Factory for face recognition model loaders.

Each loader follows the ModelLoader protocol (loaders/base.py): owns its full
per-model preprocessing chain so benchmark comparisons stay apple-to-apple.
"""

from .base import ModelLoader


def get_loader(model_name: str) -> ModelLoader:
    if model_name == "arcface_buffalo_l":
        from .arcface import buffalo_l
        return buffalo_l()
    if model_name == "arcface_antelopev2":
        from .arcface import antelopev2
        return antelopev2()
    if model_name == "adaface_ir50":
        from .adaface import ir50
        return ir50()
    if model_name == "facenet512":
        from .facenet import facenet512
        return facenet512()
    raise ValueError(
        f"Unknown model: {model_name!r}. "
        "Valid: arcface_buffalo_l, arcface_antelopev2, adaface_ir50, facenet512"
    )


__all__ = ["ModelLoader", "get_loader"]
