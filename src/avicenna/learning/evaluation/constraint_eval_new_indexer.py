from typing import List, Tuple, Dict
from lark import Lark, Transformer, Token
from isla.language import DerivationTree, Path, Formula, ISLaUnparser
from functools import lru_cache

from .index import Index
from .transformer import constraint_grammar
from avicenna.data import Input


class FastEvaluationNotSupported(Exception):
    pass


def create_constraint_ast(formula: Formula) -> Dict:
    constraint_text = ISLaUnparser(formula).unparse()
    try:
        parser = Lark(constraint_grammar)
        parse_tree = parser.parse(constraint_text)

        transformer = ConstraintTransformer()
        ast = transformer.transform(parse_tree)
        return ast
    except Exception as e:
        raise FastEvaluationNotSupported(f"Error creating constraint AST: {e}")


def evaluate(candidate, inp: Input) -> bool:
    context = {'start': inp.tree}
    indexer = inp.index
    try:
        results = evaluate_constraints_batch(candidate.eval_ast, context, indexer)
    except Exception as e:
        print(f"Error evaluating constraint: {e}")
        raise e
    return all(results)


def evaluate_candidates(candidates: list, inp: Input) -> bool:
    results = []
    for cand in candidates:
        results.append(evaluate(cand, inp))
    return all(results)


class ConstraintTransformer(Transformer):
    def constraint(self, items):
        return items[0]

    def quantifier(self, items):
        quantifier_token = items[0].value
        nonterminal = items[1]
        variable_name = items[2].value
        context = items[3]
        condition = items[4]
        return {
            'type': 'quantifier',
            'quantifier': quantifier_token,
            'variable': variable_name,
            'nonterminal': nonterminal,
            'context': context,
            'condition': condition,
        }

    def nonterminal(self, items):
        return items[0].value

    def context(self, items):
        return items[0]

    def condition(self, items):
        return items[0]

    def comparison(self, items):
        left = items[1]
        operator = items[0]
        right = items[2]
        return {
            'type': 'comparison',
            'left': left,
            'operator': operator,
            'right': right,
        }

    def expr(self, items):
        return items[0]

    def function_call(self, items):
        function_name = items[0].value
        argument = items[1]
        return {
            'type': 'function_call',
            'function': function_name,
            'argument': argument,
        }

    def function_name(self, items):
        return items[0].value

    def operator(self, items):
        return items[0]

    def variable(self, items):
        return {'type': 'variable', 'name': items[0].value}

    def string(self, items):
        return {'type': 'string', 'value': items[0].value.strip('"')}

    def number(self, items):
        return {'type': 'number', 'value': int(items[0].value)}


def evaluate_constraint(ast_node, context, indexer: Index):
    node_type = ast_node['type']
    if node_type == 'quantifier':
        quantifier = ast_node['quantifier']
        variable_name = ast_node['variable']
        nonterminal = ast_node['nonterminal']
        in_context = ast_node['context']
        condition = ast_node['condition']

        # Determine the nodes to search based on the context
        if in_context == 'start':
            nodes = [context['start']]
        elif in_context in context:
            # 'in_context' is a variable in the context
            nodes = [context[in_context]]
        else:
            # 'in_context' is a non-terminal; find nodes matching it in the entire tree
            nodes = indexer.get_nodes(context['start'], in_context)
            if not nodes:
                return False  # No nodes found for the non-terminal

        results = []

        for node in nodes:
            # Find all nodes matching the non-terminal within the current node
            matching_nodes = indexer.get_nodes(node, nonterminal)

            if quantifier == 'forall':
                if not matching_nodes:
                    results.append(False)
                else:
                    result = all(
                        evaluate_constraint(
                            condition,
                            {**context, variable_name: mn},
                            indexer
                        )
                        for mn in matching_nodes
                    )
                    results.append(result)
            elif quantifier == 'exists':
                if not matching_nodes:
                    results.append(False)
                else:
                    result = any(
                        evaluate_constraint(
                            condition,
                            {**context, variable_name: mn},
                            indexer
                        )
                        for mn in matching_nodes
                    )
                    results.append(result)

        if quantifier == 'forall':
            return all(results)
        elif quantifier == 'exists':
            return any(results)

    elif node_type == 'comparison':
        left = evaluate_expression(ast_node['left'], context)
        right = evaluate_expression(ast_node['right'], context)
        operator = ast_node['operator']
        return compare_values(left, operator, right)
    else:
        raise NotImplementedError(f"Unknown constraint type: {node_type}")


def evaluate_expression(expr_node, context):
    if isinstance(expr_node, dict):
        expr_type = expr_node.get('type')
        if expr_type == 'function_call':
            function = expr_node['function']
            argument = evaluate_expression(expr_node['argument'], context)
            if function == 'str.len':
                return len(argument)
            elif function == 'str.to.int':
                try:
                    return int(float(argument))
                except ValueError:
                    return None
            else:
                raise NotImplementedError(f"Unknown function: {function}")
        elif expr_type == 'variable':
            variable_name = expr_node['name']
            node = context.get(variable_name)
            if node is not None:
                value = get_node_value(node)
                # if value == '':
                #     # Handle non-terminal nodes by using their value directly
                #     value = node.value.strip('<>')  # Remove angle brackets if present
                #print(f"Variable '{variable_name}' value: {value}")
                return value
            else:
                raise ValueError(f"Variable '{variable_name}' not found in context.")
        elif expr_type == 'string':
            return expr_node['value']
        elif expr_type == 'number':
            return expr_node['value']
        else:
            raise NotImplementedError(f"Unknown expression type: {expr_type}")
    else:
        return expr_node


def evaluate_constraints_batch(ast_nodes: list, context, indexes):
    results = []
    for ast_node in ast_nodes:
        result = evaluate_constraint(ast_node, context, indexes)
        results.append(result)
    return results


def get_node_value(node: DerivationTree) -> str:
    return node.to_string(show_open_leaves=False)
    # if node.children:
    #     # Concatenate the values of all terminal descendants
    #     values = []
    #     def collect_values(n):
    #         if not n.children:
    #             values.append(n.value)
    #         else:
    #             for child in n.children:
    #                 collect_values(child)
    #     collect_values(node)
    #     return ''.join(values)
    # else:
    #     return node.value


def compare_values(left, operator, right):
    op = operator.value if isinstance(operator, Token) else operator
    if type(left) is not type(right) or left is None or right is None:
        return False
    assert type(left) is type(right), f"Left value type is not equal to right value type: {type(left)} != {type(right)}"
    # print(type(left), left, op, right, type(right))
    try:
        if op == '=':
            return left == right
        elif op == '<':
            return left < right
        elif op == '>':
            return left > right
        elif op == '<=':
            return left <= right
        elif op == '>=':
            return left >= right
        else:
            raise ValueError(f"Unknown operator: {op}")
    except (TypeError, ValueError):
        # Handle errors in comparison
        return False


if __name__ == "__main__":
    # constraint_text = '''
    # exists <function> elem in start:
    #     (elem = "sqrt")
    # '''
    #
    # constraint_text = '''
    # forall <number> elem in start:
    #     ((str.to.int elem) < 1)
    # '''

    from time import time
    # from debugging_benchmark.calculator.calculator import calculator_grammar as grammar
    from debugging_benchmark.middle.middle import middle_grammar as grammar
    from isla.fuzzer import GrammarFuzzer

    fuzzer = GrammarFuzzer(grammar)
    test_inputs = []
    for _ in range(10000):
        test_inputs.append(fuzzer.fuzz_tree())

    # constraints_texts = {
    #     'sqrt-constraint1': 'exists <function> elem in start: ( = elem "sqrt")',
    #     'number-constraint2': 'forall <digit> elem in start: ( = (str.to.int elem) 1)',
    #     'number-1-constraint2': 'forall <one_nine> elem in start: ( = (str.to.int elem) 1)',
    # }

    constraints_texts = {
        #'number-constraint2': 'forall <y> elem in start: ( <= (str.to.int elem) (str.to.int "10"))',
        'number-constraint1': 'forall <x> container in start: exists <digit> elem_0 in container: ( = (str.to.int elem_0) (str.to.int "9"))',
        # 'number-1-constraint2': 'forall <one_nine> elem in start: ( = (str.to.int elem) 1)',
    }

    parser = Lark(constraint_grammar)
    transformer = ConstraintTransformer()

    constraints_asts = {}
    for name, text in constraints_texts.items():
        parse_tree = parser.parse(text)
        ast = transformer.transform(parse_tree)
        constraints_asts[name] = ast

    constraint_text = '''
    forall <x> container in start:
        exists <digit> elem_0 in container:
            (= (str.to.int elem_0) (str.to.int "9"))
    '''

    parse_tree = parser.parse(constraint_text)
    ast = transformer.transform(parse_tree)

    syntax_tree_input1 = DerivationTree("<start>", [
        DerivationTree("<x>", [
            DerivationTree("<integer>", [
                DerivationTree("<digit>", [DerivationTree("9")])
            ])
        ]),
        DerivationTree("<y>", [
            DerivationTree("<integer>", [
                DerivationTree("<digit>", [DerivationTree("6")])
            ])
        ]),
        DerivationTree("<z>", [
            DerivationTree("<integer>", [
                DerivationTree("<digit>", [DerivationTree("2"), DerivationTree("1")])
            ])
        ])
    ])

    syntax_tree_input2 = DerivationTree("<start>", [
        DerivationTree("<x>", [
            DerivationTree("<integer>", [
                DerivationTree("<digit>", [DerivationTree("3")])
            ])
        ]),
        DerivationTree("<y>", [
            DerivationTree("<integer>", [
                DerivationTree("<digit>", [DerivationTree("9")])
            ])
        ]),
        DerivationTree("<z>", [
            DerivationTree("<integer>", [
                DerivationTree("<digit>", [DerivationTree("9")])
            ])
        ])
    ])

    # print(parse_tree.pretty())

    start = time()
    for inp in test_inputs:
        indexer = Index(inp)
        results = evaluate_constraints_batch(list(constraints_asts.values()), {'start': inp}, indexer)
        print(str(inp), all(results))

    new_time = time() - start
    print("New time:", new_time)
    # # For Input 1
    # indexer_input1 = Index(syntax_tree_input1)
    #
    # # For Input 2
    # indexer_input2 = Index(syntax_tree_input2)
    #
    # context = {'start': syntax_tree_input1}
    # result = evaluate_constraint(ast, context, indexer_input1)
    # print("Constraint satisfied for Input 1:", result)
    #
    #
    # from time import time
    #
    # start = time()
    # for inp in test_inputs:
    #     nonterminal_index = index_nodes_by_non_terminal(inp)
    #     indexes = nonterminal_index  # For simplicity, but you can include more indexes if needed
    #     context = {'start': inp}
    #     results = evaluate_constraints_batch(constraints_asts, context, indexes)
    #     print(str(inp), all(results.values()))
    #
    # new_time = time() - start
    # print("New time:", new_time)
    # for name, result in results.items():
    #     print(f"Constraint '{name}' satisfied: {result}")

    from isla.evaluator import evaluate
    from grammar_graph import gg

    # constraint_text = '''
    # (exists <function> elem in start:
    #    (= elem "sqrt") and
    # forall <digit> elem_0 in start:
    #   (= (str.to.int elem_0) (str.to.int "1")))
    #     '''

    constraint_text = '''
    forall <x> container in start: exists <digit> elem_0 in container: ( = (str.to.int elem_0) (str.to.int "9")) and 
    forall <y> elem in start: ( <= (str.to.int elem) (str.to.int "10"))
    '''
    graph = gg.GrammarGraph.from_grammar(grammar)

    start_time = time()
    # for inp in test_inputs:
    #     result = bool(evaluate(constraint_text, inp, grammar, graph=graph))
    #     print(str(inp), result)

    eval_time = time() - start_time

    print("New time:", new_time)
    print("Old time:", eval_time)
    print("Speedup:", eval_time / new_time)
