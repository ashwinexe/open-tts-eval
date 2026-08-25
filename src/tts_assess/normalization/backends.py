from __future__ import annotations

import importlib
from collections.abc import Callable

from tts_assess.config import NormalizationConfig
from tts_assess.normalization.english import normalize_english

Normalizer = Callable[[str], str]


def build_normalizer(config: NormalizationConfig) -> Normalizer:
    if config.backend == "english-basic":
        return lambda text: normalize_english(text, expand_numbers=config.expand_numbers)
    if config.backend == "nemo":
        return _build_nemo_normalizer()
    if config.backend == "plugin":
        if not config.plugin:
            raise ValueError("normalization.plugin is required when backend is 'plugin'")
        return _load_plugin(config.plugin)
    raise ValueError(f"Unsupported normalization backend: {config.backend}")


def _build_nemo_normalizer() -> Normalizer:
    try:
        from nemo_text_processing.text_normalization.normalize import Normalizer as NemoNormalizer
    except ImportError as exc:
        raise RuntimeError(
            "NeMo text normalization is not installed. Install nemo_text_processing or use "
            "`normalization.backend: english-basic`."
        ) from exc
    normalizer = NemoNormalizer(input_case="cased", lang="en")
    return lambda text: normalize_english(normalizer.normalize(text, verbose=False))


def _load_plugin(spec: str) -> Normalizer:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError("normalization.plugin must use 'module:function' format")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"normalization plugin is not callable: {spec}")
    return fn
