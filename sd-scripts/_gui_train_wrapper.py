import sys, os
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_target = _script_dir / 'anima_train_network.py'

sys.path.insert(0, str(_script_dir))
os.chdir(_script_dir)
with open(_target, encoding='utf-8') as _f:
    _code = compile(_f.read(), _f.name, 'exec')
exec(_code, {'__name__': '__main__', '__file__': str(_target)})
