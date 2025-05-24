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


class MTB_ProcessWhisperDiarization:
    """Process Whisper chunks with speaker diarization using either pyannote or NeMo."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "whisper_chunks": ("WHISPER_CHUNKS",),
                "audio": ("AUDIO",),
                "backend": (["pyannote", "nemo"], {"default": "pyannote"}),
                "num_speakers": (
                    "INT",
                    {"default": 2, "min": 1, "max": 10, "step": 1},
                ),
            },
            "optional": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("WHISPER_CHUNKS",)
    FUNCTION = "process"
    CATEGORY = "mtb/audio"

    def process_pyannote(self, audio, num_speakers, device):
        """Process audio using pyannote backend."""
        try:
            from pyannote.audio import Pipeline
            from pyannote.audio.pipelines.utils.hook import ProgressHook
        except ImportError:
            raise ImportError(
                "pyannote.audio not found. Install with: pip install pyannote.audio"
            )

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=None
        )
        pipeline.to(torch.device(device))
        with ProgressHook() as hook:
            diarization = pipeline(
                {
                    "waveform": audio["waveform"][0],
                    "sample_rate": audio["sample_rate"],
                },
                num_speakers=num_speakers,
                hook=hook,
            )

        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append(
                {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                }
            )

        return speaker_segments

    def process_nemo(self, audio, num_speakers, device):
        """Process audio using NeMo backend."""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError(
                "NeMo not found. Install with: pip install nemo_toolkit[asr]"
            )

        model = nemo_asr.models.ClusteringDiarizer.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        ).to(device)

        diarization = model.diarize(
            audio=audio["waveform"][0],
            sample_rate=audio["sample_rate"],
            num_speakers=num_speakers,
        )

        speaker_segments = []
        for segment in diarization:
            speaker_segments.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": f"SPEAKER_{segment['speaker']}",
                }
            )

        return speaker_segments

    def process(
        self,
        whisper_chunks,
        audio,
        backend="pyannote",
        num_speakers=2,
        device="cuda",
    ):
        if backend == "pyannote":
            speaker_segments = self.process_pyannote(
                audio, num_speakers, device
            )
        else:  # nemo
            speaker_segments = self.process_nemo(audio, num_speakers, device)

        for chunk in whisper_chunks["chunks"]:
            chunk_start, chunk_end = chunk["timestamp"]
            chunk_speakers = set()
            for segment in speaker_segments:
                if (
                    segment["start"] <= chunk_end
                    and segment["end"] >= chunk_start
                ):
                    chunk_speakers.add(segment["speaker"])

            if chunk_speakers:
                chunk["speaker"] = list(chunk_speakers)[0]
            else:
                chunk["speaker"] = "unknown"

        return (whisper_chunks,)


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
