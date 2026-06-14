import sys, os
from pathlib import Path
_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
os.chdir(str(_root))
_train = _root / 'anima_train_addift.py'
with open(_train, encoding='utf-8') as _f:
    _code = compile(_f.read(), str(_train), 'exec')
exec(_code, {'__name__': '__main__', '__file__': str(_train)})
