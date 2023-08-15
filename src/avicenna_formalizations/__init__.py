from importlib import resources
from pathlib import Path


def get_pattern_file_path() -> Path:
    with resources.path('avicenna_formalizations', 'patterns.toml') as p:
        return Path(p)


def get_islearn_pattern_file() -> Path:
    with resources.path('avicenna_formalizations', 'patterns_islearn.toml') as p:
        return Path(p)
