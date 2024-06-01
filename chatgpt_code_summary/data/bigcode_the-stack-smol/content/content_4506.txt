from typing import *

T = TypeVar('T')

MAGIC_ATTR = "__cxxpy_s13s__"

def template(cls: T) -> T:
   s13s = {}
   setattr(cls, MAGIC_ATTR, s13s)
   def __class_getitem__(args):
      if not isinstance(args, tuple):
         args = (args,)
      if args not in s13s:
         name = cls.__name__ + ", ".join(map(str, args))
         class s12n(cls):
            ...
         s12n.__name__ = name
         s12n.__qualname__ = name
         s13s[args] = s12n
      return s13s[args]
   cls.__class_getitem__ = __class_getitem__
   return cls

NOCOPY = ("__dict__", "__doc__", "__module__", "__weakref__")

def implement(actual):
   def decorator(cls: Type[T]) -> None:
      for k, v in cls.__dict__.items():
         if k not in NOCOPY:
            setattr(actual, k, v)
   return decorator

@template
class Ops(Generic[T]):
   def add(a: T, b: T) -> T:
      ...

@implement(Ops[int])
class _:
   def add(a: int, b: int) -> int:
      return a + b

@implement(Ops[str])
class _:
   def add(a: str, b: str) -> str:
      return f"{a} {b}"

print(f"{Ops[int].add(1, 2) = }")
print(f"{Ops[str].add('hello', 'world') = }")
