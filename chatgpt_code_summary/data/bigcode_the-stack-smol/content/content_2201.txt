from lark import Lark, Transformer, v_args
from lark.visitors import Interpreter, visit_children_decor

p = Lark.open("rules.lark", parser="lalr", rel_to=__file__)

code = """
// Firrst win in my book
b = 4;
a = b*2;
print a+1
x = 7;
p = [1, 2, 3, 4]
print p
"""

tree = p.parse(code)


@v_args(inline=True)
class MyEval(Transformer):
    from operator import add, mul, neg, sub
    from operator import truediv as div

    number = float

    def __init__(self, ns):
        self.ns = ns

    def var(self, name):
        return self.ns[name]

    # def num_list(self, value):
    #     print(value)


def eval_expr(tree, ns):
    return MyEval(ns).transform(tree)


@v_args(inline=True)
class MyInterp(Interpreter):
    def __init__(self):
        self.namespace = {}

    def assign(self, var, expr):
        self.namespace[var] = eval_expr(expr, self.namespace)

    def print_statement(self, expr):
        # print(expr)
        res = eval_expr(expr, self.namespace)
        print(res)


print(tree.pretty())
# MyInterp().visit(tree)
