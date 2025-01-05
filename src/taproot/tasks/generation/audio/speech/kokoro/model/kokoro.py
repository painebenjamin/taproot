from __future__ import annotations

import os
import random
import torch
import torch.nn as nn

from typing import Any, Dict, List, Tuple, Union, Optional

from taproot.util import logger

from .bert import KokoroAlbert
from .istftnet import Decoder
from .layers import TextEncoder
from .predictor import ProsodyPredictor
from .util import length_to_mask, phonemize, tokenize, untokenize

__all__ = ["KokoroModel"]

class KokoroModel(nn.Module):
    """
    Kokoro TTS model.
    """
    voices: Dict[str, torch.Tensor]

    def __init__(
        self,
        bert: KokoroAlbert,
        predictor: ProsodyPredictor,
        decoder: Decoder,
        text_encoder: TextEncoder,
        voices: Optional[Dict[str, torch.Tensor]]=None
    ) -> None:
        """
        :param bert: The pretrained BERT model.
        :param predictor: The prosody predictor.
        :param decoder: The decoder model.
        :param text_encoder: The text encoder.
        """
        super().__init__()
        self.bert = bert
        self.predictor = predictor
        self.decoder = decoder
        self.text_encoder = text_encoder
        self.bert_encoder = nn.Linear(
            bert.config.hidden_size,
            decoder.in_dim,
        )
        self.voices = {} if voices is None else voices
        self.flatten_grandchildren()

    def flatten_grandchildren(self) -> None:
        """
        Flattens the parameters of the grandchildren of the model.
        """
        for module in self.children():
            for child in module.children():
                if isinstance(child, nn.RNNBase):
                    child.flatten_parameters()

    @torch.no_grad()
    def forward(
        self,
        tokens: List[int],
        ref_s: torch.Tensor,
        speed: float=1.0
    ) -> torch.Tensor:
        """
        :param tokens: The input tokens.
        :param ref_s: The reference spectrogram.
        :param speed: The speed factor.
        :return: The generated spectrogram.
        """
        next_param = next(self.parameters())
        device = next_param.device
        dtype = next_param.dtype

        ref_s = ref_s.to(device, dtype)
        tokens = torch.LongTensor([[0, *tokens, 0]]).to(device) # type: ignore[assignment]
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device) # type: ignore[attr-defined]
        text_mask = length_to_mask(input_lengths).to(device)
        bert_dur = self.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]

        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed # type: ignore[call-overload]
        pred_dur = torch.round(duration).clamp(min=1).long()
        pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item()) # type: ignore[call-overload]
        pred_aln_trg = pred_aln_trg.to(device, dtype)
        c_frame = 0

        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + pred_dur[0,i].item()] = 1 # type: ignore[misc]
            c_frame += pred_dur[0,i].item() # type: ignore[assignment]

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
        f0_pred, n_pred = self.predictor.f0_n_train(en, s)

        t_en = self.text_encoder(tokens, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg.unsqueeze(0)

        return self.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze() # type: ignore[no-any-return]

    @torch.no_grad()
    def generate(
        self,
        text: str,
        voice: Optional[str]=None,
        voice_embed: Optional[torch.Tensor]=None,
        lang: str="en-us", # american english
        speed: float=1.0,
        return_text: bool=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, str]]:
        """
        :param text: The input text.
        :param voice: The voice to use.
        :param voice_embed: The voice embedding to use.
        :param lang: The language of the text.
        :param speed: The speed factor.
        :param return_text: Whether to return the text or not.
        :return: The generated spectrogram.
        """
        if voice is not None:
            voice_embed = self.voices.get(voice, None)
        if voice_embed is None:
            if voice is None:
                if self.voices:
                    voice = random.choice(list(self.voices.keys()))
                    logger.warning(f"No voice provided, using {voice}.")
                    voice_embed = self.voices[voice]
                else:
                    raise ValueError("Voice not provided and no voices available.")
            elif voice not in self.voices:
                raise ValueError(f"Voice {voice} not found. Voices available: {list(self.voices.keys())}")
            else:
                raise ValueError("No voice embedding provided.")

        if voice is not None:
            if "en.us" in voice:
                lang = "en-us"
            elif "en.gb" in voice:
                lang = "en-gb"

        phonemes = phonemize(text, lang)
        tokens = tokenize(phonemes)

        if not tokens:
            raise ValueError("Text is empty.")
        if len(tokens) > 510:
            tokens = tokens[:510]
            logger.warning("Text is too long, truncating to 510 tokens.")

        reference = voice_embed[len(tokens)] # Voices are packed as a matrix
        out = self.forward(tokens, reference, speed)

        if return_text:
            return out, untokenize(tokens)
        return out

    @classmethod
    def from_config(cls, **config: Any) -> KokoroModel:
        """
        Creates a new KokoroModel from a configuration dictionary.
        """
        bert = KokoroAlbert(**config.get("bert", {}))
        predictor = ProsodyPredictor(**config.get("predictor", {}))
        decoder = Decoder(**config.get("decoder", {}))
        text_encoder = TextEncoder(**config.get("text_encoder", {}))
        voice_pack = config.get("voices", None)

        voices: Optional[Dict[str, torch.Tensor]] = None
        if voice_pack is not None:
            # Load voices
            _, ext = os.path.splitext(voice_pack)
            if ext == ".safetensors":
                import safetensors
                voices = {}
                with safetensors.safe_open(voice_pack, framework="pt", device="cpu") as f: # type: ignore[no-untyped-call,attr-defined]
                    for key in f.keys():
                        voices[key] = f.get_tensor(key)
            else:
                voices = torch.load(voice_pack, weights_only=True, map_location="cpu")

        return cls(bert, predictor, decoder, text_encoder, voices=voices)
