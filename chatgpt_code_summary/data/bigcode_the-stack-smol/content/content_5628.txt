import typing 
import sys 
import numpy as np



def set_val(
  a: np.array,
  i: int,
  x: int,
) -> typing.NoReturn:
  while i < a.size:
    a[i] = max(a[i], x)
    i += i & -i


def get_mx(
  a: np.array,
  i: int,
) -> int:
  mx = 0 
  while i > 0:
    mx = max(mx, a[i])
    i -= i & -i
  return mx



def solve(
  n: int,
  h: np.array,
  a: np.array,
) -> typing.NoReturn:
  fw = np.zeros(
    n + 1,
    dtype=np.int64,
  )
  mx = 0 
  for i in range(n):
    v = get_mx(fw, h[i] - 1)
    set_val(fw, h[i], v + a[i])
  print(get_mx(fw, n))



def main() -> typing.NoReturn:
  n = int(input())
  h = np.array(
    sys.stdin.readline()
    .split(),
    dtype=np.int64,
  )
  a = np.array(
    sys.stdin.readline()
    .split(),
    dtype=np.int64,
  )
  solve(n, h, a)



OJ = 'ONLINE_JUDGE'
if sys.argv[-1] == OJ:
  from numba import njit, i8
  from numba.pycc import CC
  cc = CC('my_module')
  fn = solve
  sig = (i8, i8[:], i8[:])
  get_mx = njit(get_mx)
  set_val = njit(set_val)
  cc.export(
    fn.__name__,
    sig,
  )(fn)
  cc.compile()
  exit(0)


from my_module import solve
main()