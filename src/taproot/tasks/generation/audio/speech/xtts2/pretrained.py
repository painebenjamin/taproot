from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union, List, Sequence, TYPE_CHECKING
from functools import partial
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig # type: ignore[import-not-found,import-untyped,unused-ignore]
    from .model import XTTS2

__all__ = ["XTTS2Model"]

class XTTS2Model(PretrainedModelMixin):
    """
    XTTS-V2 model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/speech-synthesis-xtts-v2.safetensors"
    init_file_urls = {
        "speakers": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/speech-synthesis-xtts-v2-speakers.pth",
        "vocab": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/speech-synthesis-xtts-v2-vocab.json",
    }
    use_eval: bool = False # Will do it ourselves

    @classmethod
    def get_config_class(cls) -> Optional[Type[XttsConfig]]:
        """
        Returns the configuration class.
        """
        from TTS.tts.configs.xtts_config import XttsConfig # type: ignore[import-not-found,import-untyped,unused-ignore]
        return XttsConfig # type: ignore[no-any-return]

    @classmethod
    def get_model_class(cls) -> Type[XTTS2]:
        """
        Returns the model class.
        """
        from .model import XTTS2
        return XTTS2

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        from TTS.tts.configs.shared_configs import BaseDatasetConfig # type: ignore[import-not-found,import-untyped,unused-ignore]
        from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig # type: ignore[import-not-found,import-untyped,unused-ignore]
        return {
            "output_path": "output",
            "logger_uri": None,
            "run_name": "run",
            "project_name": None,
            "run_description": "\ud83d\udc38Coqui trainer run.",
            "print_step": 25,
            "plot_step": 100,
            "model_param_stats": False,
            "wandb_entity": None,
            "dashboard_logger": "tensorboard",
            "save_on_interrupt": True,
            "log_model_step": None,
            "save_step": 10000,
            "save_n_checkpoints": 5,
            "save_checkpoints": True,
            "save_all_best": False,
            "save_best_after": 10000,
            "target_loss": None,
            "print_eval": False,
            "test_delay_epochs": 0,
            "run_eval": True,
            "run_eval_steps": None,
            "distributed_backend": "nccl",
            "distributed_url": "tcp://localhost:54321",
            "mixed_precision": False,
            "precision": "fp16",
            "epochs": 1000,
            "batch_size": 32,
            "eval_batch_size": 16,
            "grad_clip": 0.0,
            "scheduler_after_epoch": True,
            "lr": 0.001,
            "optimizer": "radam",
            "optimizer_params": None,
            "lr_scheduler": None,
            "lr_scheduler_params": {},
            "use_grad_scaler": False,
            "allow_tf32": False,
            "cudnn_enable": True,
            "cudnn_deterministic": False,
            "cudnn_benchmark": False,
            "training_seed": 54321,
            "model": "xtts",
            "num_loader_workers": 0,
            "num_eval_loader_workers": 0,
            "use_noise_augment": False,
            "audio": XttsAudioConfig(
                sample_rate=22050,
                output_sample_rate=24000
            ),
            "use_phonemes": False,
            "phonemizer": None,
            "phoneme_language": None,
            "compute_input_seq_cache": False,
            "text_cleaner": None,
            "enable_eos_bos_chars": False,
            "test_sentences_file": "",
            "phoneme_cache_path": None,
            "characters": None,
            "add_blank": False,
            "batch_group_size": 0,
            "loss_masking": None,
            "min_audio_len": 1,
            "max_audio_len": float("inf"),
            "min_text_len": 1,
            "max_text_len": float("inf"),
            "compute_f0": False,
            "compute_energy": False,
            "compute_linear_spec": False,
            "precompute_num_workers": 0,
            "start_by_longest": False,
            "shuffle": False,
            "drop_last": False,
            "datasets": [
                BaseDatasetConfig(
                    formatter="",
                    dataset_name="",
                    path="",
                    meta_file_train="",
                    ignored_speakers=None,
                    language="",
                    phonemizer="",
                    meta_file_val="",
                    meta_file_attn_mask=""
                )
            ],
            "test_sentences": [],
            "eval_split_max_size": None,
            "eval_split_size": 0.01,
            "use_speaker_weighted_sampler": False,
            "speaker_weighted_sampler_alpha": 1.0,
            "use_language_weighted_sampler": False,
            "language_weighted_sampler_alpha": 1.0,
            "use_length_weighted_sampler": False,
            "length_weighted_sampler_alpha": 1.0,
            "model_args": XttsArgs(
                gpt_batch_size=1,
                enable_redaction=False,
                kv_cache=True,
                gpt_checkpoint=None,
                clvp_checkpoint=None,
                decoder_checkpoint=None,
                num_chars=255,
                tokenizer_file="",
                gpt_max_audio_tokens=605,
                gpt_max_text_tokens=402,
                gpt_max_prompt_tokens=70,
                gpt_layers=30,
                gpt_n_model_channels=1024,
                gpt_n_heads=16,
                gpt_number_text_tokens=6681,
                gpt_start_text_token=None,
                gpt_stop_text_token=None,
                gpt_num_audio_tokens=1026,
                gpt_start_audio_token=1024,
                gpt_stop_audio_token=1025,
                gpt_code_stride_len=1024,
                gpt_use_masking_gt_prompt_approach=True,
                gpt_use_perceiver_resampler=True,
                input_sample_rate=22050,
                output_sample_rate=24000,
                output_hop_length=256,
                decoder_input_dim=1024,
                d_vector_dim=512,
                cond_d_vector_in_each_upsampling_layer=True,
                duration_const=102400
            ),
            "model_dir": None,
            "languages": [
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cs",
                "ar",
                "zh-cn",
                "hu",
                "ko",
                "ja",
                "hi"
            ],
            "temperature": 0.75,
            "length_penalty": 1.0,
            "repetition_penalty": 5.0,
            "top_k": 50,
            "top_p": 0.85,
            "num_gpt_outputs": 1,
            "gpt_cond_len": 30,
            "gpt_cond_chunk_len": 4,
            "max_ref_len": 30,
            "sound_norm_refs": False
        }

    @classmethod
    def instantiate_and_load(
        cls,
        model_file: Optional[Union[str, List[str]]]=None,
        init_files: Optional[Dict[str, Union[str, List[str]]]]=None,
        init_weights: bool=False,
        device: Optional[Union[str, torch.device, Sequence[Union[str, torch.device]]]]=None,
        dtype: Optional[Union[str, torch.dtype]]=None,
        strict: bool=True,
        pre_quantized: bool=False,
        **kwargs: Any
    ) -> Any:
        """
        Instantiates and loads the model.
        """
        from TTS.tts.layers.xtts.xtts_manager import SpeakerManager, LanguageManager # type: ignore[import-not-found,import-untyped,unused-ignore]
        from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer # type: ignore[import-not-found,import-untyped,unused-ignore]
        from .patch import StreamGenerationPatch # type: ignore[attr-defined,import-not-found,import-untyped,unused-ignore]
        model = super().instantiate_and_load(
            model_file=model_file,
            init_files=None,
            init_weights=init_weights,
            device=device,
            dtype=dtype,
            strict=strict,
            **kwargs
        )
        assert init_files is not None
        assert "speakers" in init_files
        assert "vocab" in init_files
        model.tokenizer = VoiceBpeTokenizer(vocab_file=init_files["vocab"])
        model.speaker_manager = SpeakerManager(init_files["speakers"])
        model.language_manager = LanguageManager(cls.get_default_config())
        model.hifigan_decoder.eval()
        model.gpt.init_gpt_for_inference(kv_cache=True)
        model.gpt.eval()
        model.gpt.gpt_inference.generate_stream = partial(StreamGenerationPatch.generate, model.gpt.gpt_inference)
        model.gpt.gpt_inference.sample_stream = partial(StreamGenerationPatch.sample_stream, model.gpt.gpt_inference)
        return model
