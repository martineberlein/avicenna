"""
The learning module provides the necessary tools to learn new failure inducing candidates.
"""

import importlib.resources as pkg_resources


def get_pattern_file_path():
    return pkg_resources.path("avicenna.resources", "patterns.toml")


def get_islearn_pattern_file_path():
    return pkg_resources.path("avicenna.resources", "patterns_islearn.toml")
