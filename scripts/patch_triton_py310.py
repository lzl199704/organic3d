"""Work around a Python 3.10 inspect bug that breaks Triton kernels decorated with lambdas (flash-linear-attention).

On Python 3.10, inspect.getsourcelines() stops inside a decorator that contains a lambda
(e.g. @triton.heuristics({'X': lambda args: ...})), so Triton sees no `def` line and raises
"No function definition found for kernel". This re-reads the full block with ast when that happens.
Not needed on Python >= 3.11. Idempotent.

    /path/to/env/bin/python scripts/patch_triton_py310.py
"""
import re
import sys

import triton.runtime.jit as J

MARK = '# organic-py310-patch'
path = J.__file__
src = open(path).read()
if MARK in src:
    print('already patched:', path); sys.exit(0)
old = "            self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)\n"
new = old + f"""            if not re.search(r"^[ \\t]*def\\s+\\w+\\s*\\(", "".join(self.raw_src), re.MULTILINE):  {MARK}
                import ast as _ast, linecache as _lc
                _lines = _lc.getlines(fn.__code__.co_filename)
                _tree = _ast.parse("".join(_lines))
                for _n in _ast.walk(_tree):
                    if isinstance(_n, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and _n.name == fn.__name__ \\
                            and _n.lineno >= self.starting_line_number:
                        self.raw_src = _lines[self.starting_line_number - 1:_n.end_lineno]
                        break
"""
assert src.count(old) == 1, 'unexpected triton version; patch not applied'
src = src.replace(old, new)
if not re.search(r'^import re$', src, re.M):
    src = 'import re\n' + src
open(path, 'w').write(src)
print('patched', path)
