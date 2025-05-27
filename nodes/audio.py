from typing import Any, TypedDict

import torch
import torchaudio
from comfy.model_management import get_torch_device
from huggingface_hub import snapshot_download
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# from transformers import (
#     AutoFeatureExtractor,
#     WhisperForConditionalGeneration,
#     WhisperModel,
#     WhisperProcessor,
# )
from ..log import log
from ..utils import get_model_path

WHISPER_SAMPLE_RATE = 16000


class AudioTensor(TypedDict):
    """Comfy's representation of AUDIO data."""

    sample_rate: int
    waveform: torch.Tensor


class WhisperData(TypedDict):
    """Whisper transcription data with timestamps and speaker info."""

    text: str
    chunks: list[dict[str, Any]]
    language: str


AudioData = AudioTensor | list[AudioTensor]


class MtbAudio:
    """Base class for audio processing."""

    @classmethod
    def is_stereo(
        cls,
        audios: AudioData,
    ) -> bool:
        if isinstance(audios, list):
            return any(cls.is_stereo(audio) for audio in audios)
        else:
            return audios["waveform"].shape[1] == 2

    @staticmethod
    def resample(audio: AudioTensor, common_sample_rate: int) -> AudioTensor:
        current_rate = audio["sample_rate"]
        if current_rate != common_sample_rate:
            log.debug(
                f"Resampling audio from {current_rate} to {common_sample_rate}"
            )
            resampler = torchaudio.transforms.Resample(
                orig_freq=current_rate, new_freq=common_sample_rate
            )
            return {
                "sample_rate": common_sample_rate,
                "waveform": resampler(audio["waveform"]),
            }
        else:
            return audio

    @staticmethod
    def to_stereo(audio: AudioTensor) -> AudioTensor:
        if audio["waveform"].shape[1] == 1:
            return {
                "sample_rate": audio["sample_rate"],
                "waveform": torch.cat(
                    [audio["waveform"], audio["waveform"]], dim=1
                ),
            }
        else:
            return audio

    @classmethod
    def preprocess_audios(
        cls, audios: list[AudioTensor]
    ) -> tuple[list[AudioTensor], bool, int]:
        max_sample_rate = max([audio["sample_rate"] for audio in audios])

        resampled_audios = [
            cls.resample(audio, max_sample_rate) for audio in audios
        ]

        is_stereo = cls.is_stereo(audios)
        if is_stereo:
            audios = [cls.to_stereo(audio) for audio in resampled_audios]

        return (audios, is_stereo, max_sample_rate)


class WhisperPipeline(TypedDict):
    """Whisper model pipeline."""

    processor: WhisperProcessor
    model: WhisperForConditionalGeneration


class MTB_LoadWhisper:
    """Load Whisper model and processor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_size": (
                    [
                        "tiny",
                        "small",
                        "medium",
                        "medium.en",
                        "base",
                        "large",
                        "large-v2",
                        "large-v3",
                        "large-v3-turbo",
                    ],
                    {"default": "tiny"},
                ),
            },
            "optional": {
                "download_missing": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Download missing models if missing,"
                            "otherwise they must be in ComfyUI/models/whisper"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("WHISPER_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    CATEGORY = "mtb/audio"
    FUNCTION = "load"

    def load(self, model_size="tiny", download_missing=False):
        """Load Whisper model and processor."""
        whisper_dir = get_model_path("whisper")
        tag = f"whisper-{model_size}"
        model_dir = whisper_dir / tag

        if not (whisper_dir.exists() or model_dir.exists()):
            if not download_missing:
                raise RuntimeError(
                    "Models not found and download_missing=False"
                )
            else:
                whisper_dir.mkdir(exist_ok=True)
                model_dir.mkdir(exist_ok=True)

                snapshot_download(
                    repo_id=f"openai/{tag}",
                    resume_download=True,
                    ignore_patterns=["*.msgpack", "*.bin", "*.h5"],
                    local_dir=model_dir.as_posix(),
                    local_dir_use_symlinks=False,
                )

        device = get_torch_device()
        log.debug(
            f"Loading Whisper model {model_size} on {device} from {model_dir}"
        )

        processor = WhisperProcessor.from_pretrained(model_dir.as_posix())
        model = WhisperForConditionalGeneration.from_pretrained(
            model_dir.as_posix()
        ).to(device)

        model.eval()
        model.requires_grad_(False)

        return ({"processor": processor, "model": model},)


class MTB_AudioToText(MtbAudio):
    """Transcribe audio to text using Whisper."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("WHISPER_PIPELINE",),
                "audio": ("AUDIO",),
                "language": (
                    ["auto"]
                    + sorted(
                        [
                            "en",
                            "fr",
                            "es",
                            "de",
                            "it",
                            "pt",
                            "nl",
                            "ru",
                            "zh",
                            "ja",
                            "ko",
                        ]
                    ),
                    {"default": "auto"},
                ),
                "return_timestamps": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "WHISPER_OUTPUT")
    FUNCTION = "transcribe"
    CATEGORY = "mtb/audio"

    def transcribe(
        self,
        pipeline: WhisperPipeline,
        audio: AudioTensor,
        language="auto",
        return_timestamps=True,
    ):
        """Transcribe audio to text using Whisper."""
        processor = pipeline["processor"]
        model = pipeline["model"]
        device = model.device

        audio = self.resample(audio, WHISPER_SAMPLE_RATE)

        waveform = audio["waveform"]
        log.debug(f"Processed waveform shape: {waveform.shape}")

        # - Mono: [1, 1, samples] or [1, samples] or [samples]
        # - Stereo: [1, 2, samples] or [2, samples] or [samples, 2]
        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(0)

        if len(waveform.shape) == 2:
            if waveform.shape[0] == 2:  # [channels, samples]
                waveform = waveform.mean(dim=0)
            elif waveform.shape[1] == 2:  # [samples, channels]
                waveform = waveform.mean(dim=1)
            else:  # mono
                waveform = waveform.squeeze(0)

        sample_rate = audio["sample_rate"]
        chunk_duration = 30
        chunk_samples = chunk_duration * sample_rate
        total_samples = waveform.shape[-1]
        total_duration = total_samples / sample_rate

        log.debug(f"Audio duration: {total_duration:.2f}s")

        all_tokens = []
        all_text = []
        chunk_offsets = []

        last_time = 0.0
        accumulated_offset = 0.0

        for chunk_start in range(0, total_samples, chunk_samples):
            chunk_end = min(chunk_start + chunk_samples, total_samples)
            chunk_waveform = waveform[chunk_start:chunk_end]
            chunk_offset = chunk_start / sample_rate
            chunk_offsets.append(chunk_offset)

            log.debug(
                f"Processing chunk {chunk_offset:.1f}s - {chunk_end / sample_rate:.1f}s"
            )

            max_length = model.config.max_length or 448
            attention_mask = torch.ones((1, max_length))

            input_features = processor(
                chunk_waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
            ).input_features.to(device)

            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    attention_mask=attention_mask.to(device),
                    task="transcribe",
                    language=None if language == "auto" else language,
                    return_timestamps=return_timestamps,
                    no_repeat_ngram_size=3,
                    num_beams=5,
                    length_penalty=1.0,
                    max_length=max_length,
                )

            chunk_tokens = processor.tokenizer.convert_ids_to_tokens(
                predicted_ids[0]
            )

            adjusted_tokens = []
            for token in chunk_tokens:
                if token.startswith("<|") and token.endswith("|>"):
                    try:
                        time_str = token[2:-2]
                        if time_str.replace(".", "").isdigit():
                            time_val = float(time_str)

                            # If this timestamp is less than the last one, we've started a new sequence
                            if time_val < last_time:
                                accumulated_offset += last_time

                            adjusted_time = time_val + accumulated_offset
                            adjusted_tokens.append(f"<|{adjusted_time:.2f}|>")
                            last_time = time_val
                        else:
                            adjusted_tokens.append(token)
                    except ValueError:
                        adjusted_tokens.append(token)
                else:
                    adjusted_tokens.append(token)

            all_tokens.extend(adjusted_tokens)
            chunk_text = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            all_text.append(chunk_text)

        detected_language = "en"
        if language == "auto":
            try:
                log.debug("Detecting language")
                with torch.no_grad():
                    first_chunk_features = processor(
                        waveform[:chunk_samples],
                        sampling_rate=sample_rate,
                        return_tensors="pt",
                    ).input_features.to(device)

                    predicted_probs = model.detect_language(
                        first_chunk_features
                    )[0]
                    language_token = processor.tokenizer.convert_ids_to_tokens(
                        predicted_probs.argmax(-1).item()
                    )
                    detected_language = (
                        language_token[2:-2]
                        if language_token.startswith("<|")
                        else "en"
                    )
                    log.debug(f"Detected language: {detected_language}")

            except Exception as e:
                log.warning(f"Language detection failed: {e}")

        full_transcription = " ".join(all_text)

        whisper_output = {
            "text": full_transcription,
            "language": detected_language,
            "tokens": all_tokens,
            "audio": audio,
            "chunk_offsets": chunk_offsets,
        }

        return full_transcription, whisper_output


class MTB_ProcessWhisperOutput:
    """Process Whisper output into timestamped chunks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "whisper_output": ("WHISPER_OUTPUT",),
                "min_chunk_length": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "WHISPER_CHUNKS")
    FUNCTION = "process"
    CATEGORY = "mtb/audio"

    def process(self, whisper_output, min_chunk_length=0.0):
        """Process Whisper output into timestamped chunks."""
        tokens = whisper_output["tokens"]
        audio = whisper_output["audio"]
        timestamp_tokens = []

        audio_duration = audio["waveform"].shape[-1] / audio["sample_rate"]
        log.debug(f"Audio duration: {audio_duration:.2f}s")

        for i, token in enumerate(tokens):
            if token.startswith("<|") and token.endswith("|>"):
                try:
                    time_str = token[2:-2]
                    if time_str.replace(".", "").isdigit():
                        time_val = float(time_str)
                        if 0 <= time_val <= audio_duration:
                            timestamp_tokens.append((i, time_val))
                            log.debug(f"Token {i}: {time_val}")
                except ValueError:
                    continue

        chunks = []
        if len(timestamp_tokens) > 1:
            for i in range(len(timestamp_tokens) - 1):
                start_pos, start_time = timestamp_tokens[i]
                end_pos, end_time = timestamp_tokens[i + 1]

                if end_time - start_time < min_chunk_length:
                    continue

                chunk_tokens = tokens[start_pos + 1 : end_pos]
                text = " ".join(
                    t
                    for t in chunk_tokens
                    if not (t.startswith("<|") and t.endswith("|>"))
                )

                if text.strip():
                    chunks.append(
                        {
                            "text": text.strip(),
                            "timestamp": [start_time, end_time],
                        }
                    )

        if timestamp_tokens:
            start_pos, start_time = timestamp_tokens[-1]
            if start_pos < len(tokens) - 1:
                text = " ".join(
                    t
                    for t in tokens[start_pos + 1 :]
                    if not (t.startswith("<|") and t.endswith("|>"))
                )
                if text.strip():
                    if chunks:
                        prev_chunk = chunks[-1]
                        prev_duration = (
                            prev_chunk["timestamp"][1]
                            - prev_chunk["timestamp"][0]
                        )
                        end_time = min(
                            start_time + prev_duration, audio_duration
                        )
                    else:
                        end_time = audio_duration

                    if (
                        end_time > start_time
                        and end_time - start_time >= min_chunk_length
                    ):
                        chunks.append(
                            {
                                "text": text.strip(),
                                "timestamp": [start_time, end_time],
                            }
                        )

        result = {
            "text": whisper_output["text"],
            "chunks": chunks,
            "language": whisper_output["language"],
        }

        return whisper_output["text"], result


class MTB_AudioCut(MtbAudio):
    """Basic audio cutter, values are in ms."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "length": (
                    ("FLOAT"),
                    {
                        "default": 1000.0,
                        "min": 0.0,
                        "max": 999999.0,
                        "step": 1,
                    },
                ),
                "offset": (
                    ("FLOAT"),
                    {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("cut_audio",)
    CATEGORY = "mtb/audio"
    FUNCTION = "cut"

    def cut(self, audio: AudioTensor, length: float, offset: float):
        sample_rate = audio["sample_rate"]
        start_idx = int(offset * sample_rate / 1000)
        end_idx = min(
            start_idx + int(length * sample_rate / 1000),
            audio["waveform"].shape[-1],
        )
        cut_waveform = audio["waveform"][:, :, start_idx:end_idx]

        return (
            {
                "sample_rate": sample_rate,
                "waveform": cut_waveform,
            },
        )


class MTB_AudioStack(MtbAudio):
    """Stack/Overlay audio inputs (dynamic inputs).
    - pad audios to the longest inputs.
    - resample audios to the highest sample rate in the inputs.
    - convert them all to stereo if one of the inputs is.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("stacked_audio",)
    CATEGORY = "mtb/audio"
    FUNCTION = "stack"

    def stack(self, **kwargs: AudioTensor) -> tuple[AudioTensor]:
        audios, is_stereo, max_rate = self.preprocess_audios(
            list(kwargs.values())
        )

        max_length = max([audio["waveform"].shape[-1] for audio in audios])

        padded_audios: list[torch.Tensor] = []
        for audio in audios:
            padding = torch.zeros(
                (
                    1,
                    2 if is_stereo else 1,
                    max_length - audio["waveform"].shape[-1],
                )
            )
            padded_audio = torch.cat([audio["waveform"], padding], dim=-1)
            padded_audios.append(padded_audio)

        stacked_waveform = torch.stack(padded_audios, dim=0).sum(dim=0)

        return (
            {
                "sample_rate": max_rate,
                "waveform": stacked_waveform,
            },
        )


class MTB_AudioSequence(MtbAudio):
    """Sequence audio inputs (dynamic inputs).
    - adding silence_duration between each segment
      can now also be negative to overlap the clips, safely bound
      to the the input length.
    - resample audios to the highest sample rate in the inputs.
    - convert them all to stereo if one of the inputs is.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "silence_duration": (
                    ("FLOAT"),
                    {"default": 0.0, "min": -999.0, "max": 999, "step": 0.01},
                )
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("sequenced_audio",)
    CATEGORY = "mtb/audio"
    FUNCTION = "sequence"

    def sequence(self, silence_duration: float, **kwargs: AudioTensor):
        audios, is_stereo, max_rate = self.preprocess_audios(
            list(kwargs.values())
        )

        sequence: list[torch.Tensor] = []
        for i, audio in enumerate(audios):
            if i > 0:
                if silence_duration > 0:
                    silence = torch.zeros(
                        (
                            1,
                            2 if is_stereo else 1,
                            int(silence_duration * max_rate),
                        )
                    )
                    sequence.append(silence)
                elif silence_duration < 0:
                    overlap = int(abs(silence_duration) * max_rate)
                    previous_audio = sequence[-1]
                    overlap = min(
                        overlap,
                        previous_audio.shape[-1],
                        audio["waveform"].shape[-1],
                    )
                    if overlap > 0:
                        overlap_part = (
                            previous_audio[:, :, -overlap:]
                            + audio["waveform"][:, :, :overlap]
                        )
                        sequence[-1] = previous_audio[:, :, :-overlap]
                        sequence.append(overlap_part)
                        audio["waveform"] = audio["waveform"][:, :, overlap:]

            sequence.append(audio["waveform"])

        sequenced_waveform = torch.cat(sequence, dim=-1)
        return (
            {
                "sample_rate": max_rate,
                "waveform": sequenced_waveform,
            },
        )


class MTB_AudioResample(MtbAudio):
    """Resample audio to a different sample rate."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": (
                    "INT",
                    {
                        "default": 16000,
                        "min": 1000,
                        "max": 192000,
                        "step": 100,
                        "tooltip": "Target sample rate in Hz. Whisper requires 16000.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("resampled_audio",)
    CATEGORY = "mtb/audio"
    FUNCTION = "resample_audio"

    def resample_audio(
        self, audio: AudioTensor, sample_rate: int
    ) -> tuple[AudioTensor]:
        resampled = self.resample(audio, sample_rate)
        return (resampled,)


class MTB_AudioIsolateSpeaker(MtbAudio):
    """Isolate or mute specific speakers using WhisperData"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "whisper_data": ("WHISPER_CHUNKS",),
                "target_speaker": ("STRING", {"default": "SPEAKER_00"}),
                "mode": (["isolate", "mute"], {"default": "isolate"}),
                "fade_ms": (
                    "FLOAT",
                    {
                        "default": 100.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 10,
                        "tooltip": "Fade duration in milliseconds to avoid clicks",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("processed_audio",)
    CATEGORY = "mtb/audio"
    FUNCTION = "process_audio"

    def process_audio(
        self,
        audio: dict, # 在ComfyUI中，AUDIO通常是字典
        whisper_data: dict, # WHISPER_CHUNKS 也是字典
        target_speaker: str,
        mode: str = "isolate",
        fade_ms: float = 100.0,
    ) -> tuple[dict]: # 返回类型也是元组包含字典
        print("\n--- MTB_AudioIsolateSpeaker: process_audio START ---")
        print(f"原始输入 target_speaker: '{target_speaker}' (类型: {type(target_speaker)})")
        print(f"原始输入 mode: '{mode}'")
        print(f"原始输入 fade_ms: {fade_ms}")
        
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")

        if waveform is None or sample_rate is None:
            print("错误: 音频数据无效 (waveform 或 sample_rate 为 None).")
            # 返回一个有效的空/静音 AUDIO 对象
            return ({ "waveform": torch.zeros((1,1,1)), "sample_rate": 44100 },)

        print(f"输入 audio sample_rate: {sample_rate}, waveform shape: {waveform.shape}")

        if not isinstance(waveform, torch.Tensor):
            print(f"错误: audio['waveform'] 不是 torch.Tensor (实际类型: {type(waveform)}).")
            return ({ "waveform": torch.zeros((1,1,1)), "sample_rate": sample_rate },)
        if waveform.ndim != 3:
            print(f"警告: audio['waveform'] 期望是3维 [batch, channels, samples], 但得到 {waveform.ndim} 维. 尝试调整.")
            if waveform.ndim == 1: # samples -> 1,1,samples
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2: # channels,samples -> 1,channels,samples or batch,samples -> batch,1,samples
                                     # 假设是 channels,samples for now
                waveform = waveform.unsqueeze(0)
            else:
                print(f"错误: 无法处理 {waveform.ndim} 维的 waveform.")
                return ({ "waveform": torch.zeros((1,1,waveform.shape[-1] if waveform.ndim > 0 else 1)), "sample_rate": sample_rate },)
            print(f"调整后 waveform shape: {waveform.shape}")
            audio["waveform"] = waveform # 更新字典中的 waveform


        # 仅打印 whisper_data 的摘要
        if whisper_data and "chunks" in whisper_data:
            print(f"输入 whisper_data: {len(whisper_data['chunks'])} chunks found. First chunk (摘要): {str(whisper_data['chunks'][0])[:200] if whisper_data['chunks'] else 'No chunks'}")
        else:
            print("输入 whisper_data 无效或不包含 'chunks'")

        effective_target_speaker = ""
        if isinstance(target_speaker, str):
            effective_target_speaker = target_speaker.strip()
        else:
            print(f"警告: target_speaker 不是字符串 (实际类型: {type(target_speaker)}). 将视为空字符串处理.")
        print(f"处理后的 effective_target_speaker: '{effective_target_speaker}'")

        fade_samples = int((fade_ms / 1000.0) * sample_rate)

        if mode == "isolate":
            mask = torch.zeros_like(waveform)
            print("模式: isolate. 初始化 mask 为全零.")
        else:  # mode == "mute"
            mask = torch.ones_like(waveform)
            print("模式: mute. 初始化 mask 为全一.")

        if not whisper_data or "chunks" not in whisper_data or not whisper_data["chunks"]:
            print("警告: whisper_data 为空或不包含 'chunks'. 应用当前初始化的 mask.")
            processed_waveform = waveform * mask
            return ({ "sample_rate": sample_rate, "waveform": processed_waveform, },)

        for i, chunk in enumerate(whisper_data["chunks"]):
            print(f"\n  正在处理 Chunk {i+1}/{len(whisper_data['chunks'])}:")
            print(f"    Chunk 内容 (摘要): {str(chunk)[:200]}")

            current_chunk_speaker_original = chunk.get("speaker")
            if not isinstance(current_chunk_speaker_original, str):
                print(f"    警告: Chunk {i+1} 的 'speaker' 字段不是字符串 (实际: {type(current_chunk_speaker_original)}) 或不存在. 跳过此 chunk.")
                continue
            
            current_chunk_speaker_cleaned = current_chunk_speaker_original.strip()
            print(f"    Chunk speaker (原始): '{current_chunk_speaker_original}', (清理后): '{current_chunk_speaker_cleaned}'")

            speaker_is_target = (effective_target_speaker == current_chunk_speaker_cleaned)
            print(f"    比较: effective_target_speaker ('{effective_target_speaker}') == cleaned_chunk_speaker ('{current_chunk_speaker_cleaned}')? -> {speaker_is_target}")

            timestamp = chunk.get("timestamp")
            if not timestamp or not isinstance(timestamp, list) or len(timestamp) != 2:
                print(f"    错误: Chunk {i+1} 的 'timestamp' 格式无效: {timestamp}. 跳过此 chunk.")
                continue
            
            try:
                start_time = float(timestamp[0])
                end_time = float(timestamp[1])
            except (ValueError, TypeError):
                print(f"    错误: Chunk {i+1} 的 'timestamp' 值无法转换为浮点数: {timestamp}. 跳过此 chunk.")
                continue

            start_sample = int(start_time * sample_rate)
            # 使用 waveform.shape[-1] 获取样本数维度
            end_sample = min(int(end_time * sample_rate), waveform.shape[-1])
            print(f"    时间戳: [{start_time:.2f}s, {end_time:.2f}s] -> 样本范围: [{start_sample}, {end_sample}]")

            if start_sample >= end_sample or start_sample >= waveform.shape[-1] or end_sample < 0:
                print(f"    警告: Chunk {i+1} 计算得到的样本范围无效 ({start_sample}, {end_sample} vs {waveform.shape[-1]}). 跳过此 chunk.")
                continue
            
            if mode == "isolate":
                if speaker_is_target:
                    print(f"    模式 isolate, 目标说话人匹配. mask[:, :, {start_sample}:{end_sample}] 设置为 1.0")
                    mask[:, :, start_sample:end_sample] = 1.0
                else:
                    print(f"    模式 isolate, 目标说话人不匹配. mask[:, :, {start_sample}:{end_sample}] 保持 0.0")
            elif mode == "mute":
                if speaker_is_target:
                    print(f"    模式 mute, 目标说话人匹配. mask[:, :, {start_sample}:{end_sample}] 设置为 0.0")
                    mask[:, :, start_sample:end_sample] = 0.0
                else:
                    print(f"    模式 mute, 目标说话人不匹配. mask[:, :, {start_sample}:{end_sample}] 保持 1.0")

        print("\n  开始处理淡入淡出...")
        if fade_samples > 0 and mask.numel() > 0 :
            # 确保 mask 是3维 [batch, channels, samples] 且有足够的样本进行比较
            if mask.ndim == 3 and mask.shape[0] > 0 and mask.shape[1] > 0 and mask.shape[2] > 1:
                fade_device = waveform.device
                fade_slope = torch.linspace(0, 1, fade_samples, device=fade_device)
                
                # 假设我们基于第一个批次和第一个通道的掩码来确定过渡
                # 如果你的批次或通道数可能变化且行为需要不同，这里需要调整
                batch_idx_for_transitions = 0
                channel_idx_for_transitions = 0
                
                # active_mask_slice 是一维的，代表沿时间轴的掩码值
                active_mask_slice = mask[batch_idx_for_transitions, channel_idx_for_transitions, :]

                if active_mask_slice.shape[0] > 1: # 确保有足够的样本来检测过渡
                    # 找到掩码值变化的点 (0->1 或 1->0)
                    # transitions 是一维张量，包含发生变化的索引 (相对于 active_mask_slice)
                    transitions = torch.where(active_mask_slice[1:] != active_mask_slice[:-1])[0] + 1
                    print(f"    找到 {len(transitions)} 个转变点在 batch {batch_idx_for_transitions}, channel {channel_idx_for_transitions}: {transitions.tolist()}")
                else:
                    transitions = torch.empty(0, dtype=torch.long, device=fade_device)
                    print(f"    active_mask_slice 样本数 ({active_mask_slice.shape[0]}) 不足以计算转变点.")

                for trans_idx_tensor in transitions:
                    trans_idx = trans_idx_tensor.item() # 从 tensor 获取 int 值
                    print(f"    处理转变点索引: {trans_idx} (在 active_mask_slice 中)")

                    # 状态改变发生在 trans_idx，即 active_mask_slice[trans_idx] 是新状态
                    # active_mask_slice[trans_idx-1] 是旧状态
                    
                    new_state_at_transition = active_mask_slice[trans_idx]
                    old_state_before_transition = active_mask_slice[trans_idx-1] if trans_idx > 0 else (1.0 - new_state_at_transition) # 假设开始处是相反状态

                    # Fade-in: 从 0 (在 trans_idx-1) 变为 1 (在 trans_idx)
                    if new_state_at_transition == 1.0 and old_state_before_transition == 0.0:
                        # 淡入应用于 [trans_idx, trans_idx + fade_samples)
                        start_fade_sample_idx = trans_idx
                        end_fade_sample_idx = min(trans_idx + fade_samples, active_mask_slice.shape[0])
                        
                        if start_fade_sample_idx < end_fade_sample_idx: # 确保有实际长度应用fade
                            current_fade_len = end_fade_sample_idx - start_fade_sample_idx
                            print(f"      应用淡入: 范围 [{start_fade_sample_idx}, {end_fade_sample_idx}), 长度 {current_fade_len}")
                            # 应用到原始三维mask的所有批次和通道
                            # fade_slope 需要是 [1, 1, current_fade_len] 才能与 mask[:,:,slice] 相乘
                            mask[:, :, start_fade_sample_idx : end_fade_sample_idx] *= fade_slope[:current_fade_len].view(1, 1, -1)
                        else:
                            print(f"      淡入范围无效或长度为0，跳过.")
                    
                    # Fade-out: 从 1 (在 trans_idx-1) 变为 0 (在 trans_idx)
                    elif new_state_at_transition == 0.0 and old_state_before_transition == 1.0:
                        # 淡出应用于 [trans_idx - fade_samples, trans_idx)
                        start_fade_sample_idx = max(trans_idx - fade_samples, 0)
                        end_fade_sample_idx = trans_idx
                        
                        if start_fade_sample_idx < end_fade_sample_idx: # 确保有实际长度应用fade
                            current_fade_len = end_fade_sample_idx - start_fade_sample_idx
                            print(f"      应用淡出: 范围 [{start_fade_sample_idx}, {end_fade_sample_idx}), 长度 {current_fade_len}")
                            # fade.flip(0) 是从1到0的斜坡
                            # 我们需要取这个斜坡的尾部，长度为 current_fade_len
                            inverted_fade_slope = fade_slope.flip(0)
                            mask[:, :, start_fade_sample_idx : end_fade_sample_idx] *= inverted_fade_slope[fade_samples - current_fade_len:].view(1, 1, -1)
                        else:
                            print(f"      淡出范围无效或长度为0，跳过.")
                    else:
                        print(f"      转变点 {trans_idx} 不是清晰的 0->1 或 1->0 转变. 新状态: {new_state_at_transition}, 旧状态: {old_state_before_transition}. 跳过对此转变点的淡化.")
            else:
                print(f"    mask 形状 ({mask.shape if mask is not None else 'None'}) 或 fade_samples ({fade_samples}) 不适合淡入淡出处理.")
        elif mask.numel() == 0: # mask is not None but has no elements
            print(f"    mask 为空 ({mask.shape if mask is not None else 'None'}). 跳过淡入淡出.")
        else: # fade_samples == 0
            print(f"    fade_samples ({fade_samples}) 为 0. 跳过淡入淡出.")


        processed_waveform = waveform * mask
        print("--- MTB_AudioIsolateSpeaker: process_audio END ---")
        return (
            {
                "sample_rate": sample_rate,
                "waveform": processed_waveform,
            },
        )

import torch
import os
import json
import tempfile
import soundfile as sf
from omegaconf import OmegaConf
import folder_paths # ComfyUI 的路径管理模块

# --- 定义 NeMo 模型和配置文件的路径 ---
_NEMO_MODELS_SUBDIR_STR = "nemo_models" # 我们为 NeMo 模型创建的子目录名
_NEMO_CONFIG_FILENAME_STR = "nemo_diar_config.yaml" # NeMo 配置文件名

NEMO_MODELS_BASE_DIR = None
NEMO_CONFIG_PATH = None
_NEMO_PATHS_INITIALIZED_SUCCESSFULLY = False
_NEMO_INIT_LOG_MESSAGES = [] # 用于收集日志信息

try:
    # 获取 ComfyUI 的主模型目录(列表)
    # folder_paths.get_model_paths() 通常不带参数或带特定类型参数
    # 我们需要的是 ComfyUI 根目录下的 "models" 文件夹
    # folder_paths.models_dir 是一个更直接的属性 (如果存在且是你期望的)
    # 或者通过 folder_paths.get_folder_paths("") 然后找到 "models" ? 不太对
    # 最安全的方式通常是获取 ComfyUI 的根目录，然后拼接 "models"

    comfyui_root_dir = folder_paths.base_path # ComfyUI 的根目录
    _NEMO_INIT_LOG_MESSAGES.append(f"[MTB Nodes Path Setup] ComfyUI 根目录 (folder_paths.base_path): {comfyui_root_dir}")

    if not (comfyui_root_dir and isinstance(comfyui_root_dir, str) and os.path.isdir(comfyui_root_dir)):
        _NEMO_INIT_LOG_MESSAGES.append(f"错误: ComfyUI 根目录 '{comfyui_root_dir}' 无效或不是目录。")
        raise ValueError("无法确定有效的 ComfyUI 根目录。")

    # 构建 ComfyUI 的主 "models" 目录的路径
    comfyui_main_models_dir = os.path.join(comfyui_root_dir, "models")
    _NEMO_INIT_LOG_MESSAGES.append(f"[MTB Nodes Path Setup] 构建的 ComfyUI 主 'models' 目录: {comfyui_main_models_dir}")

    if not os.path.isdir(comfyui_main_models_dir):
        _NEMO_INIT_LOG_MESSAGES.append(f"错误: ComfyUI 主 'models' 目录 '{comfyui_main_models_dir}' 未找到或不是目录。")
        # 尝试 folder_paths.get_folder_paths("checkpoints") 然后取其父目录作为 models 目录的另一种尝试
        # 但这更复杂，暂时坚持使用 base_path + "models"
        
        # 作为备选，检查 folder_paths.models_dir (如果存在)
        models_dir_attr = getattr(folder_paths, 'models_dir', None)
        if models_dir_attr and isinstance(models_dir_attr, str) and os.path.isdir(models_dir_attr):
            _NEMO_INIT_LOG_MESSAGES.append(f"信息: 使用 folder_paths.models_dir 属性作为主 'models' 目录: {models_dir_attr}")
            comfyui_main_models_dir = models_dir_attr
        else:
             _NEMO_INIT_LOG_MESSAGES.append(f"警告: folder_paths.models_dir 属性 ('{models_dir_attr}') 无效或未找到。")
             raise ValueError(f"主 'models' 目录 '{comfyui_main_models_dir}' 未找到，且 folder_paths.models_dir 也无效。")


    # 现在我们有了 ComfyUI 的主 "models" 目录，在其下构建我们的 nemo_models 路径
    NEMO_MODELS_BASE_DIR = os.path.join(comfyui_main_models_dir, _NEMO_MODELS_SUBDIR_STR)
    NEMO_CONFIG_PATH = os.path.join(NEMO_MODELS_BASE_DIR, _NEMO_CONFIG_FILENAME_STR)
    
    _NEMO_PATHS_INITIALIZED_SUCCESSFULLY = True
    _NEMO_INIT_LOG_MESSAGES.append(f"成功: NeMo 模型基础目录设置为: {NEMO_MODELS_BASE_DIR}")
    _NEMO_INIT_LOG_MESSAGES.append(f"成功: NeMo 配置文件路径设置为: {NEMO_CONFIG_PATH}")

except Exception as e:
    _NEMO_INIT_LOG_MESSAGES.append(f"错误: [MTB Nodes Path Setup] 初始化 NeMo 路径时发生严重错误: {e} (类型: {type(e)})")
    # 设置无效的占位路径，以便后续检查可以明确失败
    NEMO_MODELS_BASE_DIR = "/path/to/nemo_models/global_init_failed"
    NEMO_CONFIG_PATH = "/path/to/nemo_config/global_init_failed.yaml"
    _NEMO_PATHS_INITIALIZED_SUCCESSFULLY = False

# 打印所有收集到的初始化日志信息
for msg in _NEMO_INIT_LOG_MESSAGES:
    print(msg)

if not _NEMO_PATHS_INITIALIZED_SUCCESSFULLY:
    print(f"[MTB Nodes Path Setup] 最终状态: NeMo 路径初始化失败。NeMo 后端将不可用。")
else:
    # 额外检查一下最终路径是否存在，以便在节点实际使用前给出更明确的提示
    if not os.path.isdir(NEMO_MODELS_BASE_DIR):
        print(f"警告: [MTB Nodes Path Setup] NeMo 模型基础目录 '{NEMO_MODELS_BASE_DIR}' 配置完成但实际不存在。请确保已创建该目录并放入模型。")
    if not os.path.isfile(NEMO_CONFIG_PATH):
        print(f"警告: [MTB Nodes Path Setup] NeMo 配置文件 '{NEMO_CONFIG_PATH}' 配置完成但实际不存在。请确保已创建该文件。")


# ------------------------------------------------------------------------------------
# 你的 MTB_ProcessWhisperDiarization 类定义和后续代码从这里开始 (保持不变)
# ------------------------------------------------------------------------------------
class MTB_ProcessWhisperDiarization:
    # ... (类的所有代码，包括 process_pyannote, process_nemo, _assign_speakers_to_chunks, process)
    # 在 process_nemo 方法的开头，仍然保留对 _NEMO_PATHS_INITIALIZED_SUCCESSFULLY 的检查：
    # if not _NEMO_PATHS_INITIALIZED_SUCCESSFULLY:
    #     print(f"严重错误: NeMo 路径未成功初始化。NeMo 后端无法继续。请检查启动日志中的 '[MTB Nodes Path Setup]' 相关信息。")
    #     return []
    # 并且，process_nemo 中拼接模型文件名的部分不需要改变，因为它期望 YAML 中的 model_path 只是文件名。
    # 例如：config.diarizer.vad.model_path = os.path.join(NEMO_MODELS_BASE_DIR, vad_model_filename)

    # 【保持你上一版本中 MTB_ProcessWhisperDiarization 类的完整内容，这里不再重复】
    # 确保你的类定义从这里正确开始
    """使用 pyannote 或 NeMo (本地模型) 处理 Whisper chunks 以进行说话人分离。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "whisper_chunks": ("WHISPER_CHUNKS",),
                "audio": ("AUDIO",),
                "backend": (["pyannote", "nemo"], {"default": "pyannote"}),
                "num_speakers": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "说话人数量。0 表示自动检测 (依赖后端实现)。"},
                ),
            },
            "optional": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("WHISPER_CHUNKS",)
    FUNCTION = "process"
    CATEGORY = "mtb/audio" # 保持你的分类

    def process_pyannote(self, audio: dict, num_speakers: int, device: str):
        """使用 pyannote 后端处理音频。"""
        print(f"[MTB_ProcessWhisperDiarization] Pyannote: num_speakers 设置为 {num_speakers}")
        try:
            from pyannote.audio import Pipeline
            # from pyannote.audio.pipelines.utils.hook import ProgressHook # ProgressHook 可能导致兼容性问题
        except ImportError:
            error_msg = "错误: pyannote.audio 未找到。请运行: pip install pyannote.audio"
            print(error_msg)
            raise ImportError(error_msg) # 重新抛出以在 ComfyUI 中显示

        # 注意: 理想情况下，HF Token 应作为输入或配置
        hf_token = "hf_12345678" # 示例 Token，用户应替换为自己的
        
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=hf_token if hf_token else None
            )
            pipeline.to(torch.device(device))
        except Exception as e:
            print(f"加载 pyannote 流水线时出错: {e}")
            return [] 

        audio_input_for_pipeline = {
            "waveform": audio["waveform"][0], 
            "sample_rate": audio["sample_rate"],
        }

        diarization_result = None
        try:
            if num_speakers > 0:
                print(f"[MTB_ProcessWhisperDiarization] Pyannote: 使用固定说话人数量 = {num_speakers}")
                diarization_result = pipeline(audio_input_for_pipeline, num_speakers=num_speakers)
            else:
                print("[MTB_ProcessWhisperDiarization] Pyannote: 尝试自动检测说话人数量。")
                diarization_result = pipeline(audio_input_for_pipeline)
        except Exception as e:
            print(f"pyannote 分离过程中出错: {e}")
            return []

        speaker_segments = []
        if diarization_result:
            for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
                speaker_segments.append(
                    {"start": turn.start, "end": turn.end, "speaker": speaker_label}
                )
        print(f"[MTB_ProcessWhisperDiarization] Pyannote: 找到 {len(speaker_segments)} 个说话人分段。")
        return speaker_segments

    def process_nemo(self, audio: dict, num_speakers: int, device: str):
        """使用 NeMo 后端处理音频 (本地模型和 YAML 配置)。"""
        print(f"[MTB_ProcessWhisperDiarization] NeMo: num_speakers 设置为 {num_speakers}, Device: {device}")
        
        # 检查全局路径是否成功初始化
        if not _NEMO_PATHS_INITIALIZED_SUCCESSFULLY: # 使用全局标志
            print(f"严重错误: NeMo 基础路径未成功初始化。NeMo 后端无法继续。请检查启动日志中的 '[MTB Nodes Path Setup]' 相关信息。")
            return []

        print(f"[MTB_ProcessWhisperDiarization] NeMo: 使用模型基础目录: {NEMO_MODELS_BASE_DIR}") # 全局变量
        print(f"[MTB_ProcessWhisperDiarization] NeMo: 使用配置文件路径: {NEMO_CONFIG_PATH}")   # 全局变量

        try:
            from nemo.collections.asr.models import ClusteringDiarizer
        except ImportError:
            error_msg = "错误: NeMo (nemo_toolkit[asr]) 未找到或未正确配置。请运行: pip install nemo_toolkit[asr]"
            print(error_msg)
            raise ImportError(error_msg)

        if not os.path.isfile(NEMO_CONFIG_PATH):
            print(f"严重错误: NeMo YAML 配置文件未在 {NEMO_CONFIG_PATH} 找到或不是一个文件。")
            return []
        if not os.path.isdir(NEMO_MODELS_BASE_DIR):
            print(f"严重错误: NeMo 模型基础目录未在 {NEMO_MODELS_BASE_DIR} 找到或不是一个目录。")
            return []

        with tempfile.TemporaryDirectory() as temp_dir_for_run:
            print(f"[MTB_ProcessWhisperDiarization] NeMo: 本次运行的临时目录: {temp_dir_for_run}")
            try:
                # 1. 保存临时音频文件
                waveform_tensor = audio["waveform"][0]
                waveform_to_save = None
                if waveform_tensor.ndim == 1: 
                    waveform_to_save = waveform_tensor.cpu().numpy()
                elif waveform_tensor.ndim == 2 and waveform_tensor.shape[0] == 1: 
                    waveform_to_save = waveform_tensor.squeeze(0).cpu().numpy()
                elif waveform_tensor.ndim == 2 and waveform_tensor.shape[0] > 1: 
                    print(f"[MTB_ProcessWhisperDiarization] NeMo: 音频有 {waveform_tensor.shape[0]} 个通道。仅使用第一个通道。")
                    waveform_to_save = waveform_tensor[0].cpu().numpy()
                else:
                    print(f"错误: 不支持的音频波形维度: {waveform_tensor.ndim}，形状: {waveform_tensor.shape}")
                    return []
                
                temp_wav_path = os.path.join(temp_dir_for_run, "input.wav")
                sf.write(temp_wav_path, waveform_to_save, audio["sample_rate"])
                print(f"[MTB_ProcessWhisperDiarization] NeMo: 已保存临时音频到 {temp_wav_path}")

                # 2. 从 YAML 加载基础配置
                config = OmegaConf.load(NEMO_CONFIG_PATH)

                # 3. 动态更新/确保必要的配置参数

                # 3.1 manifest 和 out_dir
                config.diarizer.manifest_filepath = os.path.join(temp_dir_for_run, "manifest.json")
                config.diarizer.out_dir = temp_dir_for_run 
                
                # 3.2 diarizer.collar (如果不存在则添加默认值)
                default_collar_value = 0.25
                if not OmegaConf.select(config, "diarizer.collar", default=None): # OmegaConf.select 更安全
                    print(f"[MTB_ProcessWhisperDiarization] NeMo: 设置默认 diarizer.collar = {default_collar_value}")
                    OmegaConf.update(config, "diarizer.collar", default_collar_value, merge=True)
                else:
                    print(f"[MTB_ProcessWhisperDiarization] NeMo: diarizer.collar 已存在于配置中，值为: {config.diarizer.collar}")

                # diarizer.ignore_overlap
                default_ignore_overlap_value = False 
                if OmegaConf.select(config.diarizer, "ignore_overlap", default=None) is None: # target config.diarizer
                    print(f"[MTB_ProcessWhisperDiarization] NeMo: 设置默认 diarizer.ignore_overlap = {default_ignore_overlap_value}")
                    OmegaConf.update(config.diarizer, "ignore_overlap", default_ignore_overlap_value, merge=True)
                else:
                    print(f"[MTB_ProcessWhisperDiarization] NeMo: diarizer.ignore_overlap 已存在于配置中，值为: {config.diarizer.ignore_overlap}")

                # 3.3 diarizer.clustering.parameters.* (如果不存在则添加默认值)
                if not OmegaConf.select(config, "diarizer.clustering.parameters", default=None):
                    print("[MTB_ProcessWhisperDiarization] NeMo: 为 diarizer.clustering 创建了空的 parameters 节")
                    OmegaConf.update(config, "diarizer.clustering.parameters", OmegaConf.create({}), merge=True)
                
                # 获取聚类参数的引用
                cluster_params_node = config.diarizer.clustering.parameters

                # oracle_num_speakers 和 max_num_speakers
                default_max_speakers_for_auto = 8 
                node_input_max_speakers = 10 # 从我们节点定义的INPUT_TYPES中的max值

                if num_speakers > 0: 
                    print(f"[MTB_ProcessWhisperDiarization] NeMo: 用户指定说话人数: {num_speakers}")
                    if hasattr(config.diarizer.speaker_embeddings, 'parameters'): # 确保 speaker_embeddings.parameters 存在
                        OmegaConf.update(config.diarizer.speaker_embeddings.parameters, "oracle_num_speakers", num_speakers, merge=True)
                    else: # 如果不存在，也尝试在 speaker_embeddings 下创建
                        OmegaConf.update(config.diarizer.speaker_embeddings, "parameters.oracle_num_speakers", num_speakers, merge=True)
                        
                    OmegaConf.update(cluster_params_node, "oracle_num_speakers", num_speakers, merge=True)
                    
                    current_max_speakers_val = max(num_speakers, default_max_speakers_for_auto)
                    final_max_speakers = min(current_max_speakers_val, node_input_max_speakers)
                    OmegaConf.update(cluster_params_node, "max_num_speakers", final_max_speakers, merge=True)
                else: 
                    print("[MTB_ProcessWhisperDiarization] NeMo: 使用自动说话人数量检测。")
                    if hasattr(config.diarizer.speaker_embeddings, 'parameters'):
                        OmegaConf.update(config.diarizer.speaker_embeddings.parameters, "oracle_num_speakers", None, merge=True)
                    else:
                        OmegaConf.update(config.diarizer.speaker_embeddings, "parameters.oracle_num_speakers", None, merge=True)
                        
                    OmegaConf.update(cluster_params_node, "oracle_num_speakers", None, merge=True)
                    OmegaConf.update(cluster_params_node, "max_num_speakers", default_max_speakers_for_auto, merge=True)
                
                print(f"[MTB_ProcessWhisperDiarization] NeMo: speaker_embeddings.parameters.oracle_num_speakers 设置为 {OmegaConf.select(config, 'diarizer.speaker_embeddings.parameters.oracle_num_speakers', default='未找到')}")
                print(f"[MTB_ProcessWhisperDiarization] NeMo: clustering.parameters.oracle_num_speakers 设置为 {cluster_params_node.oracle_num_speakers}")
                print(f"[MTB_ProcessWhisperDiarization] NeMo: clustering.parameters.max_num_speakers 设置为 {cluster_params_node.max_num_speakers}")

                # 其他常见聚类参数的默认值
                default_clustering_params_dict = {
                    "max_rp_threshold": 0.09,       
                    "sparse_search_volume": 30,     
                    "maj_smooth_factor": 7,         
                    "enhanced_count_thres": 80,     
                    "affinity_type": "cos"        
                }
                for param_name, param_value in default_clustering_params_dict.items():
                    if OmegaConf.select(cluster_params_node, param_name, default=None) is None: 
                        print(f"[MTB_ProcessWhisperDiarization] NeMo: 设置默认/缺失的聚类参数 diarizer.clustering.parameters.{param_name} = {param_value}")
                        OmegaConf.update(cluster_params_node, param_name, param_value, merge=True)
                    else:
                        print(f"[MTB_ProcessWhisperDiarization] NeMo: diarizer.clustering.parameters.{param_name} 已存在于配置中，值为: {OmegaConf.select(cluster_params_node, param_name)}")

                # 3.4 顶层配置参数
                config.device = device
                top_level_num_workers = 0 # 保持为0以避免多进程和Numba问题
                config.num_workers = top_level_num_workers
                vad_expected_sample_rate = 16000 # VAD 和 Speaker model 通常期望这个
                config.sample_rate = vad_expected_sample_rate
                config.verbose = True
                print(f"[MTB_ProcessWhisperDiarization] NeMo: 顶层配置: device='{config.device}', num_workers={config.num_workers}, sample_rate={config.sample_rate}, verbose={config.verbose}")

                # 3.5 解析并设置完整的模型文件路径
                try:
                    model_configs_to_resolve = {
                        "vad_model": config.diarizer.vad.model_path,
                        "speaker_model": config.diarizer.speaker_embeddings.model_path,
                        "msdd_model": config.diarizer.msdd_model.model_path
                    }
                    for key, filename_in_yaml in model_configs_to_resolve.items():
                        if not isinstance(filename_in_yaml, str):
                            raise TypeError(f"YAML中 {key} 的 model_path ('{filename_in_yaml}') 不是字符串。")
                        
                        full_model_path = os.path.join(NEMO_MODELS_BASE_DIR, filename_in_yaml)
                        if not os.path.isfile(full_model_path):
                            print(f"严重错误: NeMo 模型文件 ({key}: {filename_in_yaml}) 未在 {full_model_path} 找到。")
                            return []
                        
                        # 更新回 config 对象
                        if key == "vad_model":
                            config.diarizer.vad.model_path = full_model_path
                        elif key == "speaker_model":
                            config.diarizer.speaker_embeddings.model_path = full_model_path
                        elif key == "msdd_model":
                            config.diarizer.msdd_model.model_path = full_model_path
                        print(f"[MTB_ProcessWhisperDiarization] NeMo: 解析后的 {key} 模型路径为 {full_model_path}")
                
                except Exception as e_resolve_path:
                    print(f"严重错误: 解析或检查模型文件路径时出错: {e_resolve_path}")
                    return []
                
                # 4. 初始化 ClusteringDiarizer
                print(f"[MTB_ProcessWhisperDiarization] NeMo: 使用最终配置初始化 ClusteringDiarizer...")
                diar_model = ClusteringDiarizer(cfg=config)
                # diar_model 内部会根据 cfg.device 进行 .to(device) 操作
                print(f"[MTB_ProcessWhisperDiarization] NeMo: 模型已在 {config.device} 上初始化 (由ClusteringDiarizer内部处理)。")

                # 5. 创建 manifest 文件 (NeMo 会从 config.diarizer.manifest_filepath 读取)
                file_duration = sf.info(temp_wav_path).duration
                meta = {
                    'audio_filepath': temp_wav_path, 'offset': 0, 'duration': file_duration,
                    'label': 'infer', 'text': '-', 
                    'num_speakers': num_speakers if num_speakers > 0 else None,
                    'rttm_filepath': None, 'uem_filepath': None
                }
                with open(config.diarizer.manifest_filepath, 'w', encoding='utf-8') as fp:
                    json.dump(meta, fp)
                    fp.write('\n')
                print(f"[MTB_ProcessWhisperDiarization] NeMo: 已在 {config.diarizer.manifest_filepath} 创建 manifest 文件")
                
                # 6. 运行说话人分离
                print(f"[MTB_ProcessWhisperDiarization] NeMo: 开始运行 diarization...")
                diar_model.diarize() 

                # 7. 解析 RTTM 文件
                rttm_filename = os.path.basename(temp_wav_path).replace(".wav", ".rttm")
                predicted_rttm_path = os.path.join(config.diarizer.out_dir, "pred_rttms", rttm_filename)
                
                print(f"[MTB_ProcessWhisperDiarization] NeMo: 检查 RTTM 文件于 {predicted_rttm_path}")
                speaker_segments = []
                if os.path.isfile(predicted_rttm_path):
                    with open(predicted_rttm_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if not parts or parts[0] != "SPEAKER": 
                                continue 
                            
                            start_time = float(parts[3])
                            duration = float(parts[4])
                            raw_nemo_speaker_id = parts[7] # NeMo RTTM 中的原始说话人ID

                            # --- 新的标签格式化逻辑 ---
                            # 目标格式: SPEAKER_XX (例如 SPEAKER_00, SPEAKER_01, SPEAKER_10)
                            
                            processed_id_str = ""
                            if raw_nemo_speaker_id.startswith("speaker_"):
                                # 处理 "speaker_0", "speaker_1" 等格式
                                try:
                                    # 提取数字部分
                                    num_part = int(raw_nemo_speaker_id.split('_')[-1])
                                    processed_id_str = f"{num_part:02d}" # 格式化为两位数字，不足补零
                                except ValueError:
                                    # 如果 "speaker_" 之后不是纯数字，则使用原始ID作为后备
                                    processed_id_str = raw_nemo_speaker_id 
                            elif raw_nemo_speaker_id.isdigit():
                                # 处理纯数字 "0", "1" 等格式
                                try:
                                    num_part = int(raw_nemo_speaker_id)
                                    processed_id_str = f"{num_part:02d}"
                                except ValueError:
                                    processed_id_str = raw_nemo_speaker_id # 不太可能发生，但作为保险
                            else:
                                # 对于其他无法识别的格式 (例如 "UNK", "SPEAKER_A")，
                                # 保留原始ID或进行特定处理。
                                # 为了与 Pyannote 的 SPEAKER_XX 格式对齐，如果它是非数字，
                                # 我们可能需要一个映射或只保留原始ID，并接受它与Pyannote格式不同。
                                # 但通常 diarization RTTM 的 speaker_id 字段是数字或 "speaker_数字" 形式。
                                # 如果是 "SPEAKER_00" 这种格式，我们直接用。
                                if raw_nemo_speaker_id.startswith("SPEAKER_") and len(raw_nemo_speaker_id.split('_')[-1]) == 2 and raw_nemo_speaker_id.split('_')[-1].isdigit():
                                     processed_id_str = raw_nemo_speaker_id.split('_')[-1] # 直接取 "00"
                                else: # 其他未知格式，保留原始ID作为数字部分（可能导致非两位数）
                                     processed_id_str = raw_nemo_speaker_id 


                            # 最终的 speaker_label
                            if processed_id_str.isdigit() and len(processed_id_str) < 2 : # 如果是单个数字，补零
                                speaker_label = f"SPEAKER_{int(processed_id_str):02d}"
                            elif processed_id_str.isdigit() and len(processed_id_str) == 2: # 如果已经是两位数字
                                speaker_label = f"SPEAKER_{processed_id_str}"
                            else: # 如果处理后不是纯两位数字 (例如，原始就是 "SPEAKER_A" 或处理出错)
                                  # 为了避免错误，可以创建一个基于原始ID的标签，或者标记为未知
                                  # 但我们的目标是尽可能得到 SPEAKER_XX
                                  # 如果 processed_id_str 已经是 "00", "01" 这种，就直接用
                                  # 鉴于上面处理逻辑，processed_id_str 应该是 "XX" 形式的数字或原始非数字ID
                                  # 我们再统一一下，确保是 SPEAKER_XX 格式
                                  
                                  # 重新整理逻辑，目标是得到数字部分并格式化
                                  final_numeric_id = -1
                                  if raw_nemo_speaker_id.startswith("speaker_"):
                                      try: final_numeric_id = int(raw_nemo_speaker_id.split('_')[-1])
                                      except: pass
                                  elif raw_nemo_speaker_id.isdigit():
                                      try: final_numeric_id = int(raw_nemo_speaker_id)
                                      except: pass
                                  elif raw_nemo_speaker_id.startswith("SPEAKER_") and raw_nemo_speaker_id.split('_')[-1].isdigit():
                                       try: final_numeric_id = int(raw_nemo_speaker_id.split('_')[-1])
                                       except: pass
                                  
                                  if final_numeric_id != -1:
                                      speaker_label = f"SPEAKER_{final_numeric_id:02d}"
                                  else: # 无法解析为数字，使用原始ID加上SPEAKER_前缀（如果还没有）
                                      if raw_nemo_speaker_id.startswith("SPEAKER_"):
                                          speaker_label = raw_nemo_speaker_id
                                      else:
                                          speaker_label = f"SPEAKER_{raw_nemo_speaker_id}"


                            speaker_segments.append({
                                "start": start_time,
                                "end": start_time + duration,
                                "speaker": speaker_label,
                            })
                    print(f"[MTB_ProcessWhisperDiarization] NeMo: 从 RTTM 解析了 {len(speaker_segments)} 个分段。")
                else:
                    print(f"[MTB_ProcessWhisperDiarization] NeMo: RTTM 文件未在 {predicted_rttm_path} 找到。")
                
                return speaker_segments
            # ... (except 块) ...

            except FileNotFoundError as e_fnf:
                print(f"NeMo 处理过程中文件未找到错误: {e_fnf}")
                import traceback
                traceback.print_exc()
                return []
            except Exception as e:
                print(f"NeMo 处理过程中出错: {e}")
                import traceback
                traceback.print_exc()
                return []

    def _assign_speakers_to_chunks(self, whisper_chunks_data: dict, speaker_segments: list) -> dict:
        """将说话人分段信息融合到 Whisper 的 chunks 中。"""
        processed_chunks = []
        if not isinstance(whisper_chunks_data, dict) or not isinstance(whisper_chunks_data.get("chunks"), list):
            return {
                "text": whisper_chunks_data.get("text", "") if isinstance(whisper_chunks_data, dict) else "",
                "chunks": [{"timestamp": [0,0], "text":"输入 whisper_chunks 结构错误", "speaker": "error_input_format"}],
                "language": whisper_chunks_data.get("language", "") if isinstance(whisper_chunks_data, dict) else ""
            }
        
        original_chunks = whisper_chunks_data.get("chunks", [])

        for chunk_idx, chunk_orig in enumerate(original_chunks):
            current_chunk = {}
            if isinstance(chunk_orig, dict):
                current_chunk = chunk_orig.copy()
            else:
                current_chunk = {"timestamp": [0,0], "text": str(chunk_orig), "speaker": "error_malformed_chunk_input"}

            if "timestamp" not in current_chunk or \
               not isinstance(current_chunk["timestamp"], (list, tuple)) or len(current_chunk["timestamp"]) != 2:
                current_chunk["speaker"] = "error_malformed_chunk_structure"
                processed_chunks.append(current_chunk)
                continue

            try:
                chunk_start = float(current_chunk["timestamp"][0])
                chunk_end = float(current_chunk["timestamp"][1])
            except (ValueError, TypeError, IndexError):
                current_chunk["speaker"] = "error_invalid_timestamp_values"
                processed_chunks.append(current_chunk)
                continue
            
            overlapping_segments_for_chunk = []
            for seg_idx, segment in enumerate(speaker_segments):
                try:
                    seg_start = float(segment["start"])
                    seg_end = float(segment["end"])
                except (ValueError, TypeError, KeyError):
                    continue 

                overlap_start = max(chunk_start, seg_start)
                overlap_end = min(chunk_end, seg_end)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > 0.01: 
                    overlapping_segments_for_chunk.append({
                        "speaker": segment.get("speaker", "unknown_speaker_in_segment"), 
                        "overlap_duration": overlap_duration,
                    })

            if not overlapping_segments_for_chunk:
                current_chunk["speaker"] = "unknown_no_overlap"
            else:
                speaker_total_overlaps = {} 
                for ov_seg in overlapping_segments_for_chunk:
                    spk = ov_seg["speaker"]
                    speaker_total_overlaps[spk] = speaker_total_overlaps.get(spk, 0.0) + ov_seg["overlap_duration"]
                
                if speaker_total_overlaps:
                    sorted_speakers_by_overlap = sorted(speaker_total_overlaps.items(), key=lambda item: item[1], reverse=True)
                    best_speaker_candidate = sorted_speakers_by_overlap[0][0]
                    max_overlap_value = sorted_speakers_by_overlap[0][1]
                    ties = [spk_info[0] for spk_info in sorted_speakers_by_overlap if spk_info[1] == max_overlap_value]
                    if len(ties) > 1:
                        ties.sort() 
                        best_speaker_candidate = ties[0]
                    current_chunk["speaker"] = best_speaker_candidate
                else:
                    current_chunk["speaker"] = "unknown_logic_error" 
            
            processed_chunks.append(current_chunk)
        
        return {
            "text": whisper_chunks_data.get("text", ""), 
            "chunks": processed_chunks, 
            "language": whisper_chunks_data.get("language", "")
        }

    def process(
        self,
        whisper_chunks: dict, 
        audio: dict,         
        backend: str = "pyannote",
        num_speakers: int = 0, 
        device: str = "cuda",
    ):
        print(f"[MTB_ProcessWhisperDiarization] 流程开始。后端: {backend}, 目标说话人数: {num_speakers}, 设备: {device}")
        
        def create_error_chunks_output(original_text, original_lang, original_chunks_list, error_speaker_label):
            error_chunks = []
            if not isinstance(original_chunks_list, list):
                original_chunks_list = [] 
            
            if not original_chunks_list: 
                 error_chunks.append({"timestamp": [0,0], "text": original_text if original_text else "无有效输入 chunk", "speaker": error_speaker_label})
            else:
                for chk_orig in original_chunks_list:
                    error_chunk = {}
                    if isinstance(chk_orig, dict):
                        error_chunk = chk_orig.copy()
                    else: 
                        error_chunk = {"timestamp": [0,0], "text": str(chk_orig)}
                    error_chunk["speaker"] = error_speaker_label
                    error_chunks.append(error_chunk)
            return {"text":original_text, "chunks": error_chunks, "language":original_lang}

        if not isinstance(whisper_chunks, dict):
            print(f"错误: whisper_chunks 输入不是一个字典，而是 {type(whisper_chunks)}。")
            return (create_error_chunks_output("", "", [], "error_input_format"),)

        original_text = whisper_chunks.get("text", "")
        original_lang = whisper_chunks.get("language", "")
        original_chunks_list = whisper_chunks.get("chunks", []) 

        if audio is None or not isinstance(audio, dict) or \
           "waveform" not in audio or not isinstance(audio["waveform"], torch.Tensor) or \
           "sample_rate" not in audio or not isinstance(audio["sample_rate"], int) or \
           audio["waveform"].ndim < 1: 
            print("错误: 音频数据缺失、格式不正确或无效。")
            return (create_error_chunks_output(original_text, original_lang, original_chunks_list, "error_no_audio"),)

        speaker_segments = []
        try:
            if backend == "pyannote":
                speaker_segments = self.process_pyannote(audio, num_speakers, device)
            elif backend == "nemo":
                speaker_segments = self.process_nemo(audio, num_speakers, device)
            else:
                print(f"错误: 不支持的后端: {backend}")
                return (create_error_chunks_output(original_text, original_lang, original_chunks_list, f"error_unsupported_backend_{backend}"),)
        except ImportError as e_imp:
            print(f"后端处理时发生导入错误: {e_imp}")
            return (create_error_chunks_output(original_text, original_lang, original_chunks_list, "error_backend_import"),)
        except Exception as e_proc: 
            print(f"后端处理时发生未知错误: {e_proc}")
            import traceback
            traceback.print_exc()
            return (create_error_chunks_output(original_text, original_lang, original_chunks_list, "error_backend_processing"),)

        if not speaker_segments: 
            print("[MTB_ProcessWhisperDiarization] 后端未生成任何说话人分段。")
            return (create_error_chunks_output(original_text, original_lang, original_chunks_list, "unknown_no_segments"),)
        
        output_whisper_chunks = self._assign_speakers_to_chunks(whisper_chunks, speaker_segments)
        
        print("[MTB_ProcessWhisperDiarization] 流程结束。")
        return (output_whisper_chunks,)

print("--- MTB_ProcessWhisperDiarization Node (audio.py) Loaded ---")

class MTB_AudioDuration:
    """Get audio duration in milliseconds."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("duration_ms",)
    FUNCTION = "get_duration"
    CATEGORY = "mtb/audio"

    def get_duration(self, audio):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        duration_ms = int((waveform.shape[-1] / sample_rate) * 1000)
        log.debug(
            f"Audio duration: {duration_ms}ms ({duration_ms / 1000:.2f}s)"
        )

        return (duration_ms,)


__nodes__ = [
    MTB_AudioSequence,
    MTB_AudioStack,
    MTB_AudioCut,
    MTB_AudioResample,
    MTB_AudioIsolateSpeaker,
    MTB_LoadWhisper,
    MTB_AudioToText,
    MTB_ProcessWhisperOutput,
    MTB_ProcessWhisperDiarization,
    MTB_AudioDuration,
]
