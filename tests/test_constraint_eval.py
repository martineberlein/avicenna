import unittest
from lark import Lark

from grammar_graph import gg
from isla import language
from debugging_benchmark.calculator.calculator import calculator_grammar

from avicenna.learning.evaluation.constraint_eval_new_indexer import (ConstraintTransformer, constraint_grammar,
                                                                      Index, evaluate_constraints_batch,
                                                                      evaluate, evaluate_candidates)
from avicenna.data import Input
from avicenna.learning.table import Candidate


class TestConstraintParser(unittest.TestCase):

    def setUp(self):
        self.constraint_texts_calculator = [
            '''exists <function> elem in start:
                   (= elem "sqrt")''',
            '''exists <maybe_minus> elem in start: (= elem "-")''',
            # '''forall <number> elem_0 in start: (< (str.to.int elem_0) (str.to.int "2"))''',
            '''forall <number> elem_0 in start: (<= (str.to.int elem_0) (str.to.int "-1"))''',
            '''forall <number> container in start: exists <maybe_minus> elem in container: (= elem "-")'''
        ]

        self.parser = Lark(constraint_grammar, start='constraint')

    def test_valid_constraints(self):
        valid_constraints = [
            '''exists <function> elem in start:
                   (= elem "sqrt")''',
            '''forall <digit> elem_0 in start:
                (= (str.to.int elem_0) (str.to.int "1"))''',
            '''forall <number> elem_0 in start:
                (<= (str.to.int elem_0) (str.to.int "-1"))''',
            '''exists <container> elem_0 in start:
                (<= (str.len elem_0) (str.to.int "-1"))''',
            # Add more valid constraints
            '''exists <payload> container in start:
                    exists <payload-length> length_field in start:
                        (<= (str.len container) (str.to.int length_field))''',
            '''forall <arith_expr> container in start:
                    exists <function> elem in container:
                        (= (str.len elem) (str.to.int 4))''',
            '''forall <x> elem_1 in start:
                    exists <y> elem_2 in start:
                        (> (str.to.int elem_1) (str.to.int elem_2))'''

        ]
        for constraint in valid_constraints:
            with self.subTest(constraint=constraint):
                try:
                    parse_tree = self.parser.parse(constraint)
                    transformer = ConstraintTransformer()
                    ast = transformer.transform(parse_tree)
                except Exception as e:
                    self.fail(f"Constraint failed to parse or transform: {e}")

    def test_invalid_constraints(self):
        invalid_constraints = [
            '''exists <function> elem in start
                   (= elem "sqrt")''',
            # Add more invalid constraints
        ]
        for constraint in invalid_constraints:
            with self.subTest(constraint=constraint):
                with self.assertRaises(Exception):
                    self.parser.parse(constraint)

    def test_parsing_text_constraints_to_ast(self):
        transformer = ConstraintTransformer()
        for constraint_text in self.constraint_texts_calculator:
            parse_tree = self.parser.parse(constraint_text)
            ast = transformer.transform(parse_tree)
            print(ast)

    def test_constraint_eval_calculator(self):
        positive_input_texts = [
            "sqrt(-1)", "sqrt(-900)", "sqrt(-1000000)", "sqrt(-100)"
        ]
        positive_inputs = {
            Input.from_str(calculator_grammar, inp) for inp in positive_input_texts
        }
        negative_input_texts = [
            "sqrt(1)", "sqrt(900)", "sqrt(1000000)", "sqrt(100)", "cos(-10)", "sin(9)",
        ]
        negative_inputs = {
            Input.from_str(calculator_grammar, inp) for inp in negative_input_texts
        }

        constraints = [
            Candidate.from_str(constraint_text) for constraint_text in self.constraint_texts_calculator
        ]

        for inp in positive_inputs:
            result = evaluate_candidates(constraints, inp)
            print(str(inp), result)
            self.assertTrue(result)

        for inp in negative_inputs:
            result = evaluate_candidates(constraints, inp)
            print(str(inp), result)
            self.assertFalse(result)

    def test_create_constraint_ast(self):
        formula1 = """exists <function> elem_0 in start:
            (= elem_0 "cos")
        """
        formula2 = """exists <function> elem_0 in start:
            (= elem_0 "sqrt")
        """

        candidate1 = Candidate.from_str(formula1)
        candidate2 = Candidate.from_str(formula2)

        input_texts = [
            "sqrt(-1)", "cos(-900)", "sqrt(-1000000)", "cos(-100)"
        ]
        test_inputs = {
            Input.from_str(calculator_grammar, inp) for inp in input_texts
        }

        for inp in test_inputs:
            result = evaluate(candidate2, inp)
            print(str(inp), result)

    def test_candidate_evaluation(self):
        formula = """exists <function> elem_0 in start:
            (= elem_0 "sqrt")
        """
        candidate1 = Candidate.from_str(formula)
        input_texts = [
            "sqrt(-1)", "cos(-900)", "sqrt(-1000000)", "cos(-100)"
        ]
        test_inputs = {
            Input.from_str(calculator_grammar, inp) for inp in input_texts
        }
        graph = gg.GrammarGraph.from_grammar(calculator_grammar)

        candidate1.evaluate(test_inputs, graph)
        for inp, result in candidate1.comb.items():
            print(str(inp), result)

    def test_candidate_concatenation(self):
        formula1 = '''exists <maybe_minus> elem in start: (= elem "-")'''
        formula2 = """exists <function> elem_0 in start:
            (= elem_0 "sqrt")
        """
        candidate1 = Candidate.from_str(formula1)
        candidate2 = Candidate.from_str(formula2)
        candidate3 = candidate1 & candidate2
        print(candidate3.eval_ast)

        positive_input_texts = [
            "sqrt(-1)", "sqrt(-900)", "sqrt(-1000000)", "sqrt(-100)"
        ]
        positive_inputs = {
            Input.from_str(calculator_grammar, inp) for inp in positive_input_texts
        }
        negative_input_texts = [
            "sqrt(1)", "sqrt(900)", "sqrt(1000000)", "sqrt(100)", "cos(-10)", "sin(9)",
        ]
        negative_inputs = {
            Input.from_str(calculator_grammar, inp) for inp in negative_input_texts
        }

        for inp in positive_inputs:
            result = evaluate(candidate3, inp)
            print(str(inp), result)
            self.assertTrue(result)

        for inp in negative_inputs:
            result = evaluate(candidate3, inp)
            print(str(inp), result)
            self.assertFalse(result)

    def test_expression_candidate(self):
        formulas = [
            # """exists <operator> elem in start: (= elem "/")""",
            """forall <maybe_minus> container in start: exists <arith_expr> length_field in start: (<= (str.len 
            container) (str.to.int length_field))"""
        ]
        candidates = [Candidate.from_str(formula) for formula in formulas]
        candidate = candidates[0] # & candidates[1]

        positive_input_texts = []
        # positive_input_texts = [
        #     "1 / 0", "5 / (3 - 3)", "(2 + 3) / 5", "7 / (2 * 0)", "9 / (0 / 3)"
        # ]
        from debugging_framework.fuzzingbook.fuzzer import GrammarFuzzer
        from debugging_benchmark.expression.expression import expression_grammar
        from evaluation.evaluate_expression import divide_by_zero_grammar

        fuzzer = GrammarFuzzer(divide_by_zero_grammar)
        for _ in range(100):
            inp = fuzzer.fuzz()
            positive_input_texts.append(inp)

        positive_inputs = {
            Input.from_str(divide_by_zero_grammar, inp) for inp in positive_input_texts
        }

        for inp in positive_inputs:
            result = evaluate(candidate, inp)
            print(str(inp), result)


