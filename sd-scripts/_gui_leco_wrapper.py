import sys, os
sys.path.insert(0, 'E:/Animerge/sd-scripts')
os.chdir('E:/Animerge/sd-scripts')
with open('E:/Animerge/sd-scripts/anima_train_leco.py', encoding='utf-8') as _f:
    _code = compile(_f.read(), _f.name, 'exec')
exec(_code, {'__name__': '__main__', '__file__': 'E:/Animerge/sd-scripts/anima_train_leco.py'})
