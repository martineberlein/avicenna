import os
from pathlib import Path


def get_pattern_file_path() -> Path:
    return Path(__file__).parent.resolve() / 'patterns.toml'

def get_islearn_pattern_file() -> Path:
    return Path(__file__).parent.resolve() / 'patterns_islearn.py'
