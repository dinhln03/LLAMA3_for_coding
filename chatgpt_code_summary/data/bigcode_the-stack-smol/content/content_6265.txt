#AST transform that puts programs in SSA form
import collections
from translate import *

class SSAVisitor(Visitor):
    def __init__(self):
        # Number of static assignments to that variable seen so far.
        self.definition_counter = collections.defaultdict(int)
        # Name of the live definition of each variable before a node.
        self.prev_definition = collections.defaultdict(dict)
        # Name of the last definition of each variable in a node.
        self.last_definition = collections.defaultdict(dict)
        # Node in SSA form.
        self.ssa_node = {}

    def format_name(self, name, definition_id):
        return "{}_{}".format(name, definition_id)

    def visit(self, node, is_leaving):
        if isinstance(node, Node) and not is_leaving:
            if node.kind == NT.IF:
                self.prev_definition[node] = dict(self.definition_counter)
                self.prev_definition[node.args[1]] = self.prev_definition[node]

                if len(node.args) == 3:
                    self.prev_definition[node.args[2]] = self.prev_definition[node]
            # The if branches have their prev_definition set by the parent,
            # so they don't redefine it here.
            elif node not in self.prev_definition:
                self.prev_definition[node] = dict(self.definition_counter)
        elif isinstance(node, Node) and is_leaving:
            if node.kind == NT.IF:
                then_stmts = self.ssa_node[node.args[1]].args
                has_else = len(node.args) == 3

                if has_else:
                    else_stmt = self.ssa_node[node.args[2]]

                    for name, last_name in self.last_definition[node.args[1]].items():
                        c = ASTConcretizer(last_name,
                                Name(
                                    self.format_name(name,
                                        self.prev_definition[node][name] - 1)))
                        walk(else_stmt, c)
                        else_stmt = c.modified_node[else_stmt]

                else_stmts = else_stmt.args if has_else else []

                assigned_variables = set(self.last_definition[node.args[1]].keys())
                if has_else:
                    assigned_variables.update(self.last_definition[node.args[2]].keys())

                phis = []

                for v in assigned_variables:
                    then_name = (self.last_definition[node.args[1]].get(v) or
                                 self.format_name(v, self.prev_definition[node][v]))
                    else_name = ((has_else and self.last_definition[node.args[2]].get(v)) or
                                 self.format_name(v, self.prev_definition[node][v] - 1))

                    phi_name = self.format_name(v, self.definition_counter[v])

                    phis.append(Node(NT.ASSIGNMENT, [
                        Name(phi_name),
                        Node(NT.PHI, [
                            self.ssa_node[node.args[0]],
                            Name(then_name),
                            Name(else_name),
                        ])
                    ]))

                    self.definition_counter[v] += 1
                    self.last_definition[node][v] = phi_name

                self.ssa_node[node] = Node(NT.STMTLIST, then_stmts + else_stmts + phis)

            elif node.kind == NT.ASSIGNMENT:
                new_name = self.format_name(
                            node.args[0].name,
                            self.definition_counter[node.args[0].name])

                self.ssa_node[node] = Node(NT.ASSIGNMENT, [
                    Name(new_name),
                    self.ssa_node[node.args[1]]
                    ])

                self.last_definition[node][node.args[0].name] = new_name
                self.definition_counter[node.args[0].name] += 1
            elif node.kind == NT.PARAMLIST:
                names = []
                for name in node.args:
                    self.last_definition[node][name.name] = self.format_name(name.name, 0)
                    self.definition_counter[name.name] = 1
                    names.append(Name(self.last_definition[node][name.name]))
                self.ssa_node[node] = Node(NT.PARAMLIST, names)
            else:
                children = []

                for a in node.args:
                    children.append(self.ssa_node[a])
                    for k, v in self.last_definition[a].items():
                        self.last_definition[node][k] = v

                self.ssa_node[node] = Node(node.kind, children)
        elif isinstance(node, Name):
            self.ssa_node[node] = Name(self.format_name(
                node.name,
                self.definition_counter[node.name] - 1))
        else:
            self.ssa_node[node] = node

class FlattenVisitor(Visitor):
    def __init__(self):
        self.flat_node = {}

    def visit(self, node, is_leaving):
        if not is_leaving:
            self.flat_node[node] = node
            return

        if isinstance(node, Node) and node.kind == NT.STMTLIST:
            children = []

            for a in node.args:
                c = self.flat_node[a]

                if c.kind == NT.STMTLIST:
                    children.extend(c.args)
                else:
                    children.append(c)

            self.flat_node[node] = Node(node.kind, children)
        elif isinstance(node, Node):
            children = []

            for a in node.args:
                children.append(self.flat_node[a])

            self.flat_node[node] = Node(node.kind, children)

    @staticmethod
    def flatten(node):
        v = FlattenVisitor()
        walk(node, v)
        return v.flat_node[node]

def ssa(node):
    unroller = ASTUnroller()
    walk(node, unroller)
    node = unroller.unrolled_node[node]
    v = SSAVisitor()
    walk(node, v)
    return FlattenVisitor.flatten(v.ssa_node[node])
