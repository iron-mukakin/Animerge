# STUB: custom_train_functions.py
# SD1.x/SDXL用カスタム学習関数。Anima学習パスでは条件フラグがFalseのため実行されない。
# train_network.py のトップレベルimportを通過させるためのスタブ。
# train_util.py が登録済みの引数は除外する。


def add_custom_train_arguments(parser, support_weighted_captions=True):
    # train_util.py が登録する引数（zero_terminal_snr等）は除外
    # ここでは未登録の引数のみ追加する
    parser.add_argument("--min_snr_gamma", type=float, default=None)
    parser.add_argument("--scale_v_pred_loss_like_noise_pred", action="store_true", default=False)
    parser.add_argument("--v_pred_like_loss", type=float, default=None)
    parser.add_argument("--debiased_estimation_loss", action="store_true", default=False)
    if support_weighted_captions:
        parser.add_argument("--weighted_captions", action="store_true", default=False)


def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler):
    raise NotImplementedError("fix_noise_scheduler_betas_for_zero_terminal_snr is not implemented in Anima stub.")


def prepare_scheduler_for_custom_training(noise_scheduler, device):
    pass


def apply_snr_weight(loss, timesteps, noise_scheduler, min_snr_gamma, v_parameterization):
    raise NotImplementedError("apply_snr_weight is not implemented in Anima stub.")


def get_weighted_text_embeddings(*args, **kwargs):
    raise NotImplementedError("get_weighted_text_embeddings is not implemented in Anima stub.")


def scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler):
    raise NotImplementedError("scale_v_prediction_loss_like_noise_prediction is not implemented in Anima stub.")


def add_v_prediction_like_loss(loss, timesteps, noise_scheduler, v_pred_like_loss):
    raise NotImplementedError("add_v_prediction_like_loss is not implemented in Anima stub.")


def apply_debiased_estimation(loss, timesteps, noise_scheduler, v_parameterization):
    raise NotImplementedError("apply_debiased_estimation is not implemented in Anima stub.")


def apply_masked_loss(loss, batch):
    raise NotImplementedError("apply_masked_loss is not implemented in Anima stub.")
