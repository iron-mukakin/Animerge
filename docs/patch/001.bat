echo import leco_train > E:\tmp_check2.py
echo import tkinter as tk >> E:\tmp_check2.py
echo import tkinter.ttk as ttk >> E:\tmp_check2.py
echo root = tk.Tk() >> E:\tmp_check2.py
echo f = ttk.Frame(root) >> E:\tmp_check2.py
echo try: >> E:\tmp_check2.py
echo     from app import lora_train as lt >> E:\tmp_check2.py
echo     print("lora_train import OK:", hasattr(lt, '_build_sample_tab_common')) >> E:\tmp_check2.py
echo except Exception as e: >> E:\tmp_check2.py
echo     print("lora_train import FAILED:", e) >> E:\tmp_check2.py
E:\Animerge\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'E:\\Animerge'); sys.path.insert(0,'E:\\Animerge\\app'); exec(open('E:\\tmp_check2.py').read())"
pause