"""sd-scripts/scheduler/ — ユーザー定義LRSchedulerの配置場所。

このフォルダに torch.optim.lr_scheduler の LRScheduler(または旧torchでは
_LRScheduler)のサブクラスを定義した .py ファイルを置くと、Animerge GUI
(LoRA/LECO/ADDifT学習タブ)の lr_scheduler プルダウンに自動的に追加される
(app/optim_scheduler_discovery.py が起動時にこのパッケージを走査する)。

注意:
  - 検出はGUI起動時に一度だけ行われる。ファイルを追加/変更した場合はGUIの
    再起動が必要(実行中の動的反映は行わない)。
  - カスタムスケジューラ選択時、sd-scripts/library/train_util.py の
    get_scheduler_fix() は lr_warmup_steps に非0値を許容しない
    (Animerge GUI側が自動的に0へ強制し、ログに警告を出す)。

例(sd-scripts/scheduler/my_scheduler.py):

    from torch.optim.lr_scheduler import LRScheduler

    class MyScheduler(LRScheduler):
        def __init__(self, optimizer, **kwargs):
            ...
"""
