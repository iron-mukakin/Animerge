"""sd-scripts/optimizer/ — ユーザー定義Optimizerの配置場所。

このフォルダに torch.optim.Optimizer のサブクラスを定義した .py ファイルを置くと、
Animerge GUI(LoRA/LECO/ADDifT学習タブ)の optimizer プルダウンに自動的に追加される
(app/optim_scheduler_discovery.py が起動時にこのパッケージを走査する)。

注意:
  - 検出はGUI起動時に一度だけ行われる。ファイルを追加/変更した場合はGUIの
    再起動が必要(実行中の動的反映は行わない)。
  - ファイル名が "_" で始まるモジュールは検出対象外。
  - モジュール内でクラスを再import(例: "from torch.optim import AdamW")した
    だけの名前は検出されない(そのモジュールで定義されたクラスのみを検出する)。

例(sd-scripts/optimizer/my_optimizer.py):

    from torch.optim import Optimizer

    class MyOptimizer(Optimizer):
        def __init__(self, params, lr=1e-3, **kwargs):
            ...
"""
