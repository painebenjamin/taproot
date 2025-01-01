from __future__ import annotations

import tempfile

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from libdf import DF # type: ignore[import-untyped,import-not-found,unused-ignore]
    from df.deepfilternet3 import DfNet # type: ignore[import-untyped,import-not-found,unused-ignore]

__all__ = ["DeepFilterNet3Model"]

CONFIG_INI = """
[train]
seed = 43
device = 
model = deepfilternet3
jit = false
mask_only = false
df_only = false
batch_size = 64
batch_size_eval = 64
num_workers = 16
max_sample_len_s = 3.0
p_atten_lim = 0.0
overfit = false
max_epochs = 120
log_freq = 100
log_timings = True
validation_criteria = loss
validation_criteria_rule = min
early_stopping_patience = 25
global_ds_sampling_f = 1
num_prefetch_batches = 8
dataloader_snrs = -100,-5,0,5,10,20,40
detect_anomaly = false
batch_size_scheduling = 0/16,2/24,5/32,10/64,20/128,40/9999
start_eval = false
validation_set_caching = false
cp_blacklist = 

[distortion]
p_reverb = 0.1
p_bandwidth_ext = 0.0
p_clipping = 0.0
p_air_absorption = 0.0
p_zeroing = 0.0
p_interfer_sp = 0.0

[df]
sr = 48000
fft_size = 960
hop_size = 480
nb_erb = 32
nb_df = 96
norm_tau = 1
lsnr_max = 35
lsnr_min = -15
min_nb_erb_freqs = 2
pad_mode = output
df_order = 5
df_lookahead = 2

[deepfilternet]
conv_lookahead = 2
conv_ch = 64
conv_depthwise = True
emb_hidden_dim = 256
emb_num_layers = 3
enc_linear_groups = 32
linear_groups = 16
conv_dec_mode = transposed
convt_depthwise = False
mask_pf = False
df_hidden_dim = 256
df_num_layers = 2
dfop_method = df
group_shuffle = False
conv_kernel = 1,3
df_gru_skip = groupedlinear
df_pathway_kernel_size_t = 5
enc_concat = False
conv_kernel_inp = 3,3
emb_gru_skip = none
df_n_iter = 1
convt_kernel = 1,3
lsnr_dropout = False

[localsnrloss]
factor = 1e-3

[maskloss]
factor = 0
mask = spec
gamma = 0.3
gamma_pred = 0.3
f_under = 1

[spectralloss]
factor_magnitude = 0
factor_complex = 0
gamma = 0.3
factor_under = 1

[dfalphaloss]
factor = 0.0

[multiresspecloss]
factor = 500
factor_complex = 500
gamma = 0.3
fft_sizes = 256,512,1024,2048

[optim]
lr = 0.001
momentum = 0
weight_decay = 1e-12
weight_decay_end = 0.01
optimizer = adamw
lr_min = 1e-06
lr_warmup = 0.0001
warmup_epochs = 3
lr_cycle_mul = 1.0
lr_cycle_decay = 0.5
lr_cycle_limit = 1
lr_update_per_epoch = False
lr_cycle_epochs = -1

[sdrloss]
factor = 0.0
segmental_ws = 0

[asrloss]
factor = 0
factor_lm = 0
loss_lm = CrossEntropy
model = base.en
"""

class DeepFilterNet3Model(PretrainedModelMixin):
    """
    Pretrained model for DeepFilterNet3.
    """
    model_url: Optional[str] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/speech-enhancement-deep-filter-net-3.safetensors"
    dtype = "float32"
    _df_state: DF

    @classmethod
    def get_model_class(cls) -> Type[DfNet]:
        """
        Returns the model class.
        """
        from df.deepfilternet3 import DfNet
        return DfNet # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        from df.modules import erb_fb # type: ignore[import-untyped,import-not-found,unused-ignore]
        df_state = cls.get_df_state()
        erb, erb_inverse = [
            erb_fb(df_state.erb_widths(), df_state.sr(), inverse=inverse)
            for inverse in [False, True]
        ]
        return {
            "erb_fb": erb,
            "erb_inv_fb": erb_inverse,
            "run_df": True,
            "train_mask": True
        }

    @classmethod
    def get_df_state(cls) -> DF:
        """
        Returns the DF state.
        """
        if not hasattr(cls, "_df_state"):
            from libdf import DF
            from df.deepfilternet3 import ModelParams # type: ignore[import-untyped,import-not-found,unused-ignore]
            from df.config import config # type: ignore[import-untyped,import-not-found,unused-ignore]
            with tempfile.NamedTemporaryFile("w") as f:
                f.write(CONFIG_INI)
                f.flush()
                config.load(f.name)
            p = ModelParams()
            cls._df_state = DF(
                sr=p.sr,
                fft_size=p.fft_size,
                hop_size=p.hop_size,
                nb_bands=p.nb_erb
            )
        return cls._df_state
