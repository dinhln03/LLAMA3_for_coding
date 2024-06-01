class Type:
    def __init__(self):
        pass

    def get_repr(self):
        return self

    def __repr__(self):
        return self.get_repr().stringify()

    def stringify(self):
        return ""

    def put_on_stack(self, stack):
        stack.put(self.get_repr())

    def take_from_stack(self, stack):
        stack.take(self.get_repr())

    def get_as_single_constant(self):
        repr = self.get_repr()
        if isinstance(repr, TypeConstant):
            return repr
        return None


class TypeConstant(Type):
    def __init__(self, name):
        self.name = name
    def stringify(self):
        return self.name

class TypeArrow(Type):
    def __init__(self, left, right, name = None):
        self.left = left
        self.right = right
        self.name = name
    def stringify(self):
        return "(" + str(self.left) + ")->" + str(self.right)
    def put_on_stack(self, stack):
        self.left.take_from_stack(stack)
        self.right.put_on_stack(stack)
    def take_from_stack(self, stack):
        raise ArrowOnTheLeftOfArrowError("Arrow type on the left hand side of the arrow type", self)

class TypeTuple(Type):
    def __init__(self, args):
        self.args = args
    def stringify(self):
        return "(" + str.join(", ", map(str, self.args)) + ")"
    def put_on_stack(self, stack):
        for arg in self.args:
            arg.put_on_stack(stack)
    def take_from_stack(self, stack):
        for arg in self.args:
            arg.take_from_stack(stack)

class TypeVar(Type):
    def __init__(self, name):
        self.name = name
        self.rank = 0
        self.parent = self

    def union(self, other):
        self_repr = self.get_repr()
        other_repr = other.get_repr()
        if self_repr == other_repr:
            return
        if isinstance(other, TypeVar):
            other_rank = other.rank
            self_rank = self.rank
            if self_rank < other_rank:
                self.parent = other_repr
            elif self_rank > other_rank:
                other.parent = self_repr
            else:
                other.parent = self_repr
                self.rank = self.rank + 1
        else:
            self.parent = other_repr

    def get_repr(self):
        if self.parent != self:
            self.parent = self.parent.get_repr()
        return self.parent
    def stringify(self):
        return "@" + self.name

class ArrowOnTheLeftOfArrowError(RuntimeError):
    def __init__(self, message, type):
        RuntimeError.__init__(self, message)
        self.message = message
        self.type = type

    def __str__(self):
        return self.message + " " + str(self.type)


class UnifiactionError(RuntimeError):
    def __init__(self, message):
        RuntimeError.__init__(self, message)
        self.message = message
        self.unify_stack = []

    def add(self, type_a, type_b):
        self.unify_stack.append((type_a, type_b))

    def __str__(self):
        return "Unification error: " + self.message + "\n" + str.join("\n", map(lambda p : "In unification of '%s' and '%s'" % p, self.unify_stack))

def types_equal(a, b):
    a = a.get_repr()
    b = b.get_repr()
    if a == b:
        return True
    if isinstance(a, TypeTuple) and isinstance(b, TypeTuple):
        if len(a.args) != len(b.args):
            return False
        return all(map(types_equal, zip(a.args, b.args)))
    elif isinstance(a, TypeArrow) and isinstance(b, TypeArrow):
        return types_equal(a.left, b.left) and types_equal(a.right, b.right)
    return False

def types_unify(a, b):
    try:
        a = a.get_repr()
        b = b.get_repr()
        if isinstance(a, TypeVar):
            a.union(b)
        elif isinstance(b, TypeVar):
            b.union(a)
        elif isinstance(a, TypeConstant) and isinstance(b, TypeConstant):
            if a != b:
                raise UnifiactionError("Different basic types")
        elif isinstance(a, TypeTuple) and isinstance(b, TypeTuple):
            if len(a.args) != len(b.args):
                raise UnifiactionError("Tuples size mismatch")
            for (a,b) in zip(a.args, b.args):
                types_unify(a, b)
        elif isinstance(a, TypeArrow) and isinstance(b, TypeArrow):
            types_unify(a.left, b.left)
            types_unify(a.right, b.right)
        else:
            raise UnifiactionError("Different kinds")
    except UnifiactionError as e:
        e.add(a, b)
        raise

def is_simple_arrow(a):
    a = a.get_repr()
    if isinstance(a, TypeArrow):
        lhs = a.left
        rhs = a.right
        if lhs.get_repr() == rhs.get_repr():
            return True
    return False

def is_type_empty(type):
    type = type.get_repr()
    return isinstance(type, TypeTuple) and len(type.args) == 0

def split_arrow(type):
    type = type.get_repr()
    lhs = []
    while isinstance(type, TypeArrow):
        lhs.append(type.left)
        type = type.right
    return (lhs, type)

class TypeStack:
    def __init__(self):
        self.given = []
        self.taken = []

    def take(self, type):
        if not isinstance(type, TypeConstant):
            raise RuntimeError("Non-constant type placed into typestack: %s" % type)
        if len(self.given) > 0:
            last = self.given.pop()
            types_unify(type, last)
        else:
            self.taken.append(type)
    def put(self, type):
        self.given.append(type)
    def form_type(self):
        if len(self.given) == 1:
            rhs = self.given[0]
        else:
            rhs = TypeTuple(self.given)
        t = rhs
        for type in reversed(self.taken):
            t = TypeArrow(type, t)
        return t

#Takes a sequence of types, produces a signle type matching the sequence
def infer_type_from_sequence(seq):
    stack = TypeStack()
    for type in seq:
        type.put_on_stack(stack)
    return stack.form_type()

if __name__ == "__main__":
    pass