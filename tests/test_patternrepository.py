import unittest
from pathlib import Path

from islearn.language import Formula
from avicenna.learner.repository import PatternRepository
from avicenna.learner import get_pattern_file_path, get_islearn_pattern_file_path


class TestPatternRepository(unittest.TestCase):

    def test_pattern_file_path(self):
        path = get_pattern_file_path()
        self.assertTrue(path.exists())

    def test_islearn_pattern_file_path(self):
        path = get_islearn_pattern_file_path()
        self.assertTrue(path.exists())

    def test_repo(self):
        # Test the PatternRepository class
        repo = PatternRepository.from_file()
        self.assertEqual(len(repo), 11)

    def test_repo_get_all(self):
        repo = PatternRepository.from_file()
        all_patterns = repo.get_all()
        self.assertEqual(len(all_patterns), 11)
        self.assertTrue(all(isinstance(pattern, Formula) for pattern in all_patterns))


if __name__ == '__main__':
    unittest.main()
