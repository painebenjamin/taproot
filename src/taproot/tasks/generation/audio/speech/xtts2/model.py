import torch
import random
import warnings

from typing import Any, Dict, Iterator, Optional, Union
from torch.nn import functional as F

from TTS.tts.layers.xtts.tokenizer import split_sentence # type: ignore[import-untyped,import-not-found,unused-ignore]
from TTS.tts.models.xtts import Xtts # type: ignore[import-untyped,import-not-found,unused-ignore]

__all__ = ["XTTS2"]

class XTTS2(Xtts): # type: ignore[misc]
    """
    Extend the XTTS model to modify some methods.
    Change 'inference' to return tensors instead of NDArrays.
    Create '__call__' method to call 'inference' or 'inference_stream' based on input type.
    """
    def inference(
        self,
        text: str,
        language: str,
        gpt_cond_latent: torch.Tensor,
        speaker_embedding: torch.Tensor,
        # GPT inference
        temperature: float=0.75,
        length_penalty: float=1.0,
        repetition_penalty: float=10.0,
        top_k: float=50,
        top_p: float=0.85,
        do_sample: bool=True,
        num_beams: int=1,
        speed: float=1.0,
        enable_text_splitting: bool=False,
        **hf_generate_kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate speech from text using the XTTS model.
        """
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)

        if enable_text_splitting:
            texts = split_sentence(text, language, self.tokenizer.char_limits[language])
        else:
            texts = [text]

        wavs = []
        gpt_latents_list = []
        for sent in texts:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            with torch.no_grad():
                gpt_codes = self.gpt.generate(
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    input_tokens=None,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=self.gpt_batch_size,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    output_attentions=False,
                    **hf_generate_kwargs,
                )
                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device
                )

                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                gpt_latents = self.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                )

                if length_scale != 1.0:
                    gpt_latents = F.interpolate(
                        gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                    ).transpose(1, 2)

                gpt_latents_list.append(gpt_latents.cpu())
                wavs.append(self.hifigan_decoder(gpt_latents, g=speaker_embedding).cpu().squeeze())

        return {
            "wav": torch.cat(wavs, dim=0),
            "gpt_latents": torch.cat(gpt_latents_list, dim=1),
            "speaker_embedding": speaker_embedding,
        }

    @torch.inference_mode()
    def inference_stream(
        self,
        text: str,
        language: str,
        gpt_cond_latent: torch.Tensor,
        speaker_embedding: torch.Tensor,
        # Streaming
        stream_chunk_size: int=20,
        overlap_wav_len: int=1024,
        # GPT inference
        temperature: float=0.75,
        length_penalty: float=1.0,
        repetition_penalty: float=10.0,
        top_k: int=50,
        top_p: float=0.85,
        do_sample: bool=True,
        speed: float=1.0,
        **hf_generate_kwargs: Any,
    ) -> Iterator[torch.Tensor]:
        """
        Generate speech from text using the XTTS model in streaming mode.
        """
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        text = split_sentence(text, language, self.tokenizer.char_limits[language])

        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            fake_inputs = self.gpt.compute_embeddings(
                gpt_cond_latent.to(self.device),
                text_tokens,
            )
            gpt_generator = self.gpt.get_generator(
                fake_inputs=fake_inputs,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=1,
                num_return_sequences=1,
                length_penalty=float(length_penalty),
                repetition_penalty=float(repetition_penalty),
                output_attentions=False,
                output_hidden_states=True,
                **hf_generate_kwargs,
            )

            last_tokens = []
            all_latents = []
            wav_gen_prev = None
            wav_overlap = None
            is_end = False

            while not is_end:
                try:
                    x, latent = next(gpt_generator)
                    last_tokens += [x]
                    all_latents += [latent]
                except StopIteration:
                    is_end = True

                if is_end or (stream_chunk_size > 0 and len(last_tokens) >= stream_chunk_size):
                    gpt_latents = torch.cat(all_latents, dim=0)[None, :]
                    if length_scale != 1.0:
                        gpt_latents = F.interpolate(
                            gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                        ).transpose(1, 2)
                    wav_gen = self.hifigan_decoder(gpt_latents, g=speaker_embedding.to(self.device))
                    wav_chunk, wav_gen_prev, wav_overlap = self.handle_chunks(
                        wav_gen.squeeze(), wav_gen_prev, wav_overlap, overlap_wav_len
                    )
                    last_tokens = []
                    yield wav_chunk

    def __call__(
        self,
        text: str,
        language: str="en",
        speaker_wav: Optional[torch.Tensor]=None,
        speaker_id: Optional[str]=None,
        stream: bool=False,
        stream_chunk_size: int=20,
        overlap_wav_len: int=1024,
        **kwargs: Any
    ) -> Union[Dict[str, torch.Tensor], Iterator[torch.Tensor]]:
        """
        Synthesize speech with the given input text.

        Args:
            text (str): Input text.
            speaker_wav (list): List of paths to the speaker audio files to be used for cloning.
            language (str): Language ID of the speaker.
            **kwargs: Inference settings. See `inference()`.

        Returns:
            A dictionary of the output values with `wav` as output waveform, `deterministic_seed` as seed used at inference,
            `text_input` as text token IDs after tokenizer, `voice_samples` as samples used for cloning, `conditioning_latents`
            as latents used at inference.

        """
        assert (
            "zh-cn" if language == "zh" else language in self.config.languages
        ), f"❗ Language {language} is not supported. Supported languages are {self.config.languages}"
        # Use generally found best tuning knobs for generation.
        settings: Dict[str, Any] = {
            "temperature": self.config.temperature,
            "length_penalty": self.config.length_penalty,
            "repetition_penalty": self.config.repetition_penalty,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
        }
        settings.update(kwargs)  # allow overriding of preset settings with kwargs
        if speaker_wav is not None:
            gpt_cond_latent, speaker_embedding = self.get_conditioning_latents(
                audio_path=speaker_wav,
                gpt_cond_len=self.config.gpt_cond_len,
                gpt_cond_chunk_len=self.config.gpt_cond_chunk_len,
                max_ref_length=self.config.max_ref_len,
                sound_norm_refs=self.config.sound_norm_refs,
            )
        else:
            if speaker_id is None:
                speaker_id = next(iter(self.speaker_manager.name_to_id))
                warnings.warn(
                    f"❗ Neither speaker wave nor ID provided. Using first speaker ID: {speaker_id}."
                )
            elif speaker_id == "random":
                speaker_id = random.choice(list(self.speaker_manager.name_to_id))
            gpt_cond_latent, speaker_embedding = self.speaker_manager.speakers[speaker_id].values()
        if stream:
            return self.inference_stream(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                stream_chunk_size=stream_chunk_size,
                overlap_wav_len=overlap_wav_len,
                **settings
            )
        return self.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            **settings
        )
