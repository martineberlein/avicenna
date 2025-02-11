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