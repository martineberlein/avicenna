from typing import Dict, List, Set, Iterable, cast
import toml
import logging

from islearn.language import AbstractISLaUnparser, parse_abstract_isla
import isla.language as language

from avicenna.learner import get_pattern_file_path

logger = logging.getLogger("learner")


class PatternRepository:
    def __init__(self, data: Dict[str, List[Dict[str, str]]]):
        self.groups: Dict[str, Dict[str, language.Formula]] = {
            group_name: {entry["name"]: parse_abstract_isla(entry["constraint"]) for entry in elements}
            for group_name, elements in data.items()
        }

    @classmethod
    def from_file(cls, file_path: str = None) -> 'PatternRepository':
        file_name = file_path if file_path else get_pattern_file_path()
        try:
            with open(file_name, "r") as f:
                contents = f.read()
        except FileNotFoundError:
            logger.warning(f"Could not find pattern file at {file_name}.")
            return cls(dict())

        data: Dict[str, List[Dict[str, str]]] = cast(Dict[str, List[Dict[str, str]]], toml.loads(contents))
        return cls(data)

    @classmethod
    def from_data(cls, data: Dict[str, List[Dict[str, str]]]) -> 'PatternRepository':
        return cls(data)

    def __getitem__(self, item: str) -> Set[language.Formula]:
        for group in self.groups.values():
            if item in group:
                return {group[item]}
        logger.warning(f"Could not find pattern for query {item}.")
        return set()

    def __contains__(self, item: str) -> bool:
        return any(item in group for group in self.groups.values())

    def __len__(self) -> int:
        return sum(len(group) for group in self.groups.values())

    def get_all(self, but: Iterable[str] = tuple()) -> Set[language.Formula]:
        exclude = {formula for pattern in but for formula in self[pattern]}
        all_patterns = {formula for group in self.groups.values() for formula in group.values()}
        return all_patterns - exclude

    def __str__(self) -> str:
        result = []
        for group, patterns in self.groups.items():
            for name, constraint in patterns.items():
                unparsed_constraint = AbstractISLaUnparser(constraint).unparse()
                result.append(f"[[{group}]]\n\nname = \"{name}\"\nconstraint = '''\n{unparsed_constraint}\n'''\n")
        return '\n'.join(result).strip()
