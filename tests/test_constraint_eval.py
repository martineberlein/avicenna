import unittest
from lark import Lark

from isla.isla_predicates import IN_TREE_PREDICATE
from isla.language import parse_isla
from debugging_benchmark.calculator.calculator import calculator_grammar

from avicenna.learning.evaluation.constraint_eval_new_indexer import (
    ConstraintTransformer, constraint_grammar, evaluate, evaluate_candidates)
from avicenna.data import Input
from avicenna.learning.table import Candidate
from avicenna.learning.evaluation.constraint_eval_new_indexer import create_constraint_ast, FastEvaluationNotSupported


class TestConstraintParser(unittest.TestCase):

    def setUp(self):
        self.constraint_texts_calculator = [
            '''exists <function> elem in start:
                   (= elem "sqrt")''',
            '''exists <maybe_minus> elem in start: (= elem "-")''',
            '''forall <number> elem_0 in start: (<= (str.to.int elem_0) (str.to.int "-1"))''',
            '''forall <number> container in start: exists <maybe_minus> elem in container: (= elem "-")'''
        ]
        self.parser = Lark(constraint_grammar, start='constraint')
        self.positive_input_texts = ["sqrt(-1)", "sqrt(-900)", "sqrt(-1000000)", "sqrt(-100)"]
        self.negative_input_texts = ["sqrt(1)", "sqrt(900)", "sqrt(1000000)", "cos(-10)", "sin(9)"]

    def test_valid_constraints(self):
        valid_constraints = [
            '''exists <function> elem in start:
                   (= elem "sqrt")''',
            '''forall <digit> elem_0 in start:
                (= (str.to.int elem_0) (str.to.int "1"))''',
            '''forall <number> elem_0 in start:
                (<= (str.to.int elem_0) (str.to.int "-1"))''',
            '''exists <payload> container in start:
                    exists <payload-length> length_field in start:
                        (<= (str.len container) (str.to.int length_field))'''
        ]
        for constraint in valid_constraints:
            with self.subTest(constraint=constraint):
                parse_tree = self.parser.parse(constraint)
                transformer = ConstraintTransformer()
                ast = transformer.transform(parse_tree)
                self.assertIsNotNone(ast, "AST should not be None after transformation")

    def test_invalid_constraints(self):
        invalid_constraints = [
            '''exists <function> elem in start
                   (= elem "sqrt")''',
            '''forall <invalid_syntax> container''',  # Example of malformed syntax
            '''exists <number> elem in start: (= elem)''',  # Missing parts of expression
            '''exists <operator> elem_xy in start: (inside(elem_xy, start))''',  # Invalid semantic type
        ]
        for constraint in invalid_constraints:
            with self.subTest(constraint=constraint):
                with self.assertRaises(Exception, msg=f"Expected parsing failure for: {constraint}"):
                    self.parser.parse(constraint)

    def test_parsing_text_constraints_to_ast(self):
        transformer = ConstraintTransformer()
        for constraint_text in self.constraint_texts_calculator:
            parse_tree = self.parser.parse(constraint_text)
            ast = transformer.transform(parse_tree)
            self.assertIsNotNone(ast, "AST should not be None after transformation")

    def test_constraint_eval_calculator_positive(self):
        positive_inputs = {
            Input.from_str(calculator_grammar, inp) for inp in self.positive_input_texts
        }
        constraints = [Candidate.from_str(constraint) for constraint in self.constraint_texts_calculator]

        for inp in positive_inputs:
            result = evaluate_candidates(constraints, inp)
            self.assertTrue(result, f"Expected True for input: {inp}")

    def test_constraint_eval_calculator_negative(self):
        negative_inputs = {
            Input.from_str(calculator_grammar, inp) for inp in self.negative_input_texts
        }
        constraints = [Candidate.from_str(constraint) for constraint in self.constraint_texts_calculator]

        for inp in negative_inputs:
            result = evaluate_candidates(constraints, inp)
            self.assertFalse(result, f"Expected False for input: {inp}")

    def test_candidate_concatenation(self):
        formula1 = '''exists <maybe_minus> elem in start: (= elem "-")'''
        formula2 = """exists <function> elem_0 in start:
            (= elem_0 "sqrt")
        """
        candidate1 = Candidate.from_str(formula1)
        candidate2 = Candidate.from_str(formula2)
        candidate3 = candidate1 & candidate2

        self.assertIsNotNone(candidate1.eval_ast)
        self.assertTrue(candidate1.use_fast_eval)
        self.assertIsNotNone(candidate2.eval_ast)
        self.assertTrue(candidate2.use_fast_eval)
        self.assertIsNotNone(candidate3.eval_ast)
        self.assertTrue(candidate3.use_fast_eval)

        positive_inputs = {Input.from_str(calculator_grammar, inp) for inp in self.positive_input_texts}
        negative_inputs = {Input.from_str(calculator_grammar, inp) for inp in self.negative_input_texts}

        for inp in positive_inputs:
            result = evaluate(candidate3, inp)
            self.assertTrue(result, f"Expected True for concatenated constraint: {inp}")

        for inp in negative_inputs:
            result = evaluate(candidate3, inp)
            self.assertFalse(result, f"Expected False for concatenated constraint: {inp}")

    def test_candidate_creation_with_invalid_constraint(self):
        formula = '''exists <operator> elem_xy in start:
            (inside(elem_xy, start))'''  # not supported
        formula = parse_isla(formula, structural_predicates={IN_TREE_PREDICATE})
        candidate1 = Candidate(formula=formula, use_fast_eval=True)

        self.assertFalse(candidate1.use_fast_eval)
        self.assertIsNone(candidate1.eval_ast)

    def test_unsupported_constraint(self):
        formula = '''exists <operator> elem_xy in start:
            (inside(elem_xy, start))'''  # not supported
        formula = parse_isla(formula, structural_predicates={IN_TREE_PREDICATE})

        with self.assertRaises(FastEvaluationNotSupported, msg="Expected exception for unsupported constraint"):
            create_constraint_ast(formula)


if __name__ == "__main__":
    unittest.main()

