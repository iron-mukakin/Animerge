echo import sys > E:\tmp_check.py
echo lines = open(r'E:\Animerge\app\leco_train.py', encoding='utf-8').readlines() >> E:\tmp_check.py
echo for i,l in enumerate(lines,1): >> E:\tmp_check.py
echo     if any(x in l for x in ['def _drain','_ab_panel','tab_sample','after(2000']): >> E:\tmp_check.py
echo         print(i, l, end='') >> E:\tmp_check.py
E:\Animerge\.venv\Scripts\python.exe E:\tmp_check.py
pause