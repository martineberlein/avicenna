from lark import Lark, Transformer, Token


def is_leaf(node):
    symbol, children = node
    return children is None or not children


def find_nodes_by_nonterminal(nodes, nonterminal):
    result = []
    for node in nodes:
        symbol, children = node
        if symbol == nonterminal:
            result.append(node)
        if children:
            # Ensure children is a list
            if isinstance(children, list):
                result.extend(find_nodes_by_nonterminal(children, nonterminal))
    return result


constraint_grammar = r"""
    ?start: constraint

    constraint: quantifier

    quantifier: QUANTIFIER nonterminal CNAME "in" context ":" condition

    nonterminal: /<[^>]+>/

    context: CNAME | nonterminal

    condition: comparison | quantifier

    comparison: "(" expr operator expr ")"

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


from lark import Transformer, Token

class ConstraintTransformer(Transformer):
    def constraint(self, items):
        return items[0]

    def quantifier(self, items):
        print(items)
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
        left = items[0]
        operator = items[1]
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
        print(items)
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



def evaluate_constraint(ast_node, context):
    node_type = ast_node['type']
    if node_type == 'quantifier':
        quantifier = ast_node['quantifier']
        variable_name = ast_node['variable']
        nonterminal = ast_node['nonterminal']
        in_context = ast_node['context']
        condition = ast_node['condition']

        # Get the nodes from the context
        if in_context == 'start':
            nodes = [context['start']]
        else:
            nodes = context.get(in_context)
            if nodes is None:
                return False
            if not isinstance(nodes, list):
                nodes = [nodes]

        # Find all nodes matching the non-terminal
        matching_nodes = find_nodes_by_nonterminal(nodes, nonterminal)

        if quantifier == 'forall':
            return all(
                evaluate_constraint(condition, {**context, variable_name: node})
                for node in matching_nodes
            )
        elif quantifier == 'exists':
            return any(
                evaluate_constraint(condition, {**context, variable_name: node})
                for node in matching_nodes
            )
    elif node_type == 'comparison':
        left = evaluate_expression(ast_node['left'], context)
        right = evaluate_expression(ast_node['right'], context)
        operator = ast_node['operator']
        return compare_values(left, operator, right)
    else:
        raise NotImplementedError(f"Unknown constraint type: {node_type}")


def get_node_value(node):
    symbol, children = node
    if is_leaf(node):
        # For leaf nodes, the symbol might be the actual value
        return symbol.strip('"')
    else:
        # For non-leaf nodes, concatenate the values of the children
        values = []
        for child in children:
            value = get_node_value(child)
            if value is not None:
                values.append(value)
        return ''.join(values)


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
            elif function == 'inside':
                # Implement 'inside' function if needed
                pass
            else:
                raise NotImplementedError(f"Unknown function: {function}")
        elif expr_type == 'variable':
            variable_name = expr_node['name']
            node = context.get(variable_name)
            if node is not None:
                # Return the node's value
                value = get_node_value(node)
                return value
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


if __name__ == "__main__":
    constraint_text = '''
    exists <function> elem in start:
        ( elem = "sqrt")
    '''

    parser = Lark(constraint_grammar)
    parse_tree = parser.parse(constraint_text)
    print(parse_tree.pretty())

    # Example syntax tree for 'sqrt(9)'
    syntax_tree = (
        "<start>",
        [
            (
                "<arith_expr>",
                [
                    ("<function>", [("sqrt", None)]),  # The function name is 'sqrt'
                    ("(", None),
                    ("<number>", [("9", None)]),
                    (")", None)
                ]
            )
        ]
    )

    constraint_text = '''
    exists <function> elem in start:
        (elem = "sqrt")
    '''

    parser = Lark(constraint_grammar)
    parse_tree = parser.parse(constraint_text)
    transformer = ConstraintTransformer()
    ast_sqrt = transformer.transform(parse_tree)
    print(ast_sqrt)

    context = {'start': syntax_tree}
    result = evaluate_constraint(ast_sqrt, context)
    print("Constraint satisfied:", result)

    constraint_text = '''
    forall <number> elem in start:
        ((str.to.int elem) < 1)
    '''

    parse_tree = parser.parse(constraint_text)
    ast_number = transformer.transform(parse_tree)

    syntax_tree = (
        "<start>",
        [
            ("<number>", [("-18", None)])
        ]
    )

    context = {'start': syntax_tree}
    result = evaluate_constraint(ast_number, context)
    print("Constraint satisfied:", result)

    from avicenna.generator.generator import ISLaGrammarBasedGenerator
    from debugging_benchmark.calculator.calculator import calculator_grammar
    from isla.fuzzer import GrammarFuzzer

    fuzzer = GrammarFuzzer(calculator_grammar)
    test_inputs = []

    from time import time

    NUM_INPUTS = 1000

    start = time()

    for _ in range(NUM_INPUTS):
        inp = fuzzer.fuzz_tree()
        context = {'start': inp}
        result = evaluate_constraint(ast_number, context) and evaluate_constraint(ast_sqrt, context)
        print(str(inp), result)

    new_time = time() - start

    # for inp in test_inputs:
    #     context = {'start': inp}
    #     result = evaluate_constraint(ast, context)
    #     print(str(inp), result)

    # # print(test_inputs[-1].tree.__repr__())
    # context = {'start': test_inputs[-1].tree}
    # result = evaluate_constraint(ast, context)
    # print("Constraint satisfied:", result)

    from isla.evaluator import evaluate
    from grammar_graph import gg

    constraint_text = '''
(exists <function> elem in start:
   (= elem "sqrt") and
forall <number> elem_0 in start:
  (< (str.to.int elem_0) (str.to.int "1")))
    '''
    graph = gg.GrammarGraph.from_grammar(calculator_grammar)

    start_time = time()
    for _ in range(NUM_INPUTS):
        inp = fuzzer.fuzz_tree()
        result = bool(evaluate(constraint_text, inp, calculator_grammar, graph=graph))
        print(str(inp), result)

    eval_time = time() - start_time

    print("New time:", new_time)
    print("Old time:", eval_time)
    print("Speedup:", eval_time / new_time)
