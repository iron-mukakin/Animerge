import sys, os
sys.path.insert(0, r'E:\Animerge\sd-scripts')
os.chdir(r'E:\Animerge\sd-scripts')
with open(r'E:\Animerge\sd-scripts\anima_train_network.py', encoding='utf-8') as _f:
    _code = compile(_f.read(), _f.name, 'exec')
exec(_code, {'__name__': '__main__', '__file__': r'E:\Animerge\sd-scripts\anima_train_network.py'})
