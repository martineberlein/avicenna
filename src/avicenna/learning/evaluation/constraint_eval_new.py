from typing import List, Tuple, Dict
from lark import Lark, Transformer, Token
from isla.language import DerivationTree, Path
from functools import lru_cache


constraint_grammar = r"""
    ?start: constraint

    constraint: quantifier

    quantifier: QUANTIFIER nonterminal CNAME "in" context ":" condition

    nonterminal: /<[^>]+>/

    context: CNAME | nonterminal

    condition: comparison | quantifier

    comparison: "(" operator expr expr ")"

    expr: function_call | variable | string | number

    function_call: "(" FUNCTION_NAME expr ")"

    FUNCTION_NAME: "str.len" | "str.to.int" | "inside"

    operator: OPERATOR

    OPERATOR: "=" | "<" | ">" | "<=" | ">="

    QUANTIFIER: "forall" | "exists"

    variable: CNAME

    string: ESCAPED_STRING

    number: INT

    %import common.CNAME
    %import common.ESCAPED_STRING
    %import common.INT
    %import common.WS
    %ignore WS
"""


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


def evaluate_constraint(ast_node, context, indexes):
    node_type = ast_node['type']
    if node_type == 'quantifier':
        quantifier = ast_node['quantifier']
        variable_name = ast_node['variable']
        nonterminal = ast_node['nonterminal']
        condition = ast_node['condition']

        matching_nodes = indexes.get(nonterminal, [])

        if quantifier == 'forall':
            if not matching_nodes:
                return False  # No nodes to satisfy the 'forall' condition
            return all(
                evaluate_constraint(condition, {**context, variable_name: node}, indexes)
                for node in matching_nodes
            )
        elif quantifier == 'exists':
            return any(
                evaluate_constraint(condition, {**context, variable_name: node}, indexes)
                for node in matching_nodes
            )
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
                return int(float(argument))
            else:
                raise NotImplementedError(f"Unknown function: {function}")
        elif expr_type == 'variable':
            variable_name = expr_node['name']
            node = context.get(variable_name)
            if node is not None:
                # Use the to_string method to get the node's value
                return node.to_string(show_open_leaves=False)
            else:
                return ''
        elif expr_type == 'string':
            return expr_node['value']
        elif expr_type == 'number':
            return expr_node['value']
        else:
            raise NotImplementedError(f"Unknown expression type: {expr_type}")
    else:
        return expr_node


def evaluate_constraints_batch(ast_nodes, context, indexes):
    results = {}
    for constraint_name, ast_node in ast_nodes.items():
        result = evaluate_constraint(ast_node, context, indexes)
        results[constraint_name] = result
    return results


def compare_values(left, operator, right):
    op = operator.value if isinstance(operator, Token) else operator
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


def index_nodes_by_non_terminal(tree: DerivationTree) -> dict[str, list[DerivationTree]]:
    index = {}

    def action(_, node):
        if node.value not in index:
            index[node.value] = []
        index[node.value].append(node)

    tree.traverse(action)
    return index



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

    # from debugging_benchmark.calculator.calculator import calculator_grammar as grammar
    from debugging_benchmark.middle.middle import middle_grammar as grammar
    from isla.fuzzer import GrammarFuzzer

    fuzzer = GrammarFuzzer(grammar)
    test_inputs = []
    for _ in range(100):
        test_inputs.append(fuzzer.fuzz_tree())

    # constraints_texts = {
    #     'sqrt-constraint1': 'exists <function> elem in start: ( = elem "sqrt")',
    #     'number-constraint2': 'forall <digit> elem in start: ( = (str.to.int elem) 1)',
    #     'number-1-constraint2': 'forall <one_nine> elem in start: ( = (str.to.int elem) 1)',
    # }

    constraints_texts = {
        # 'number-constraint2': 'forall <x> elem in start: ( <= (str.to.int elem) (str.to.int "10"))',
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


    from time import time

    start = time()
    for inp in test_inputs:
        nonterminal_index = index_nodes_by_non_terminal(inp)
        indexes = nonterminal_index  # For simplicity, but you can include more indexes if needed
        context = {'start': inp}
        results = evaluate_constraints_batch(constraints_asts, context, indexes)
        print(str(inp), all(results.values()))

    new_time = time() - start
    print("New time:", new_time)
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
    forall <x> container in start: exists <digit> elem_0 in container: ( = (str.to.int elem_0) (str.to.int "9"))
    '''
    graph = gg.GrammarGraph.from_grammar(grammar)

    start_time = time()
    for inp in test_inputs:
        result = bool(evaluate(constraint_text, inp, grammar, graph=graph))
        print(str(inp), result)

    eval_time = time() - start_time

    print("New time:", new_time)
    print("Old time:", eval_time)
    print("Speedup:", eval_time / new_time)
