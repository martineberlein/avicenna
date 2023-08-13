import pkg_resources
from pathlib import Path

def get_pattern_file_path() -> Path:
    return Path(pkg_resources.resource_filename('avicenna_formalizations', 'patterns.toml'))

def get_islearn_pattern_file() -> Path:
    return Path(pkg_resources.resource_filename('avicenna_formalizations', 'patterns_islearn.toml'))
