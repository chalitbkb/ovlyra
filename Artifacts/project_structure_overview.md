# 📁 InworldTTS — Complete Project Architecture & Knowledge Base

> โปรเจค **InworldTTS** คือระบบ Text-to-Speech (TTS) แบบ Zero-Shot Voice Cloning ที่ใช้ **SpeechLM** (Speech Language Model) บนพื้นฐาน **Causal LLM** (เช่น Llama-3.2) เพื่อสร้างเสียงพูดที่เหมือนจริงจากตัวอย่างเสียงสั้นๆ เพียงไม่กี่วินาที

---

## 🌟 หลักการทำงานระดับสูง (High-Level Architecture)

### ทำไมโปรเจคนี้ถึง Clone เสียงได้จาก Zero-Shot?

โปรเจคนี้ใช้แนวคิด **"Speech as Language"** — แปลงเสียงพูดเป็น discrete tokens (เหมือนคำในภาษา) แล้วใช้ LLM ทำนาย speech tokens ต่อไปจาก context ที่มี:

1. **Audio Codec Encoder** (`tts/core/codec/encoder.py`) — แปลงเสียง `.wav` เป็นลำดับ VQ codes (integers) ที่อัตรา 50 tokens/วินาที โดยใช้ **Wav2Vec 2.0** (semantic) + **AcousticEncoder** (acoustic) → fuse → **ResidualFSQ** quantization
2. **Causal LLM** (`tts/core/modeling.py`) — ใช้ Llama-3.2 ที่ขยาย vocabulary จาก ~128K เป็น **193,856 tokens** (เพิ่ม 65,536 speech tokens `<|s_0|>` ถึง `<|s_65535|>` + special tokens) เพื่อทำ next-token prediction บน speech tokens
3. **Audio Codec Decoder** (`tts/core/codec/decoder.py`) — แปลง VQ codes กลับเป็น waveform ผ่าน codebook lookup → VocosBackbone (ConvNeXt) → optional UpSampler → ISTFT

### กระบวนการ Voice Cloning ตอน Inference:
```
[Prompt Audio WAV] → Encoder → [Prompt Speech IDs: 142, 8301, ...]
                                          ↓
[Prompt Transcript + Text-to-Synthesize] + [Prompt Speech IDs]
                                          ↓
                           ← compiled into prompt string →
                                          ↓
                   Tokenize → LLM autoregressive generation
                                          ↓
                        [Generated Speech IDs: 5021, 991, ...]
                                          ↓
              [Prompt IDs + Generated IDs] → Decoder → WAV
                                          ↓
                      ตัด prompt portion ออก → Final Speech WAV
```

ดังนั้น LLM จะ "เรียนรู้" ลักษณะเสียงจาก prompt speech tokens ที่อยู่ใน context แล้วสร้าง speech tokens ที่มีลักษณะเสียงเดียวกันออกมา — นี่คือหัวใจของ **Zero-Shot Voice Cloning**

---

## 🔧 Technology Stack & Dependencies

### Core ML/AI Frameworks
| Library | บทบาท |
|---------|--------|
| `torch` / `torchaudio` | พื้นฐาน tensor computation, audio I/O, STFT/ISTFT |
| `transformers` (HuggingFace) | โหลด pretrained LLM (Llama), tokenizer, Wav2Vec2BertModel, AutoFeatureExtractor |
| `lightning` (Fabric) | จัดการ distributed training (DDP/FSDP/DeepSpeed), mixed precision, gradient sync |
| `deepspeed` | ZeRO-2 gradient partitioning สำหรับ training ขนาดใหญ่ |
| `peft` | LoRA (Low-Rank Adaptation) สำหรับ efficient fine-tuning |
| `trl` | GRPOTrainer สำหรับ RLHF training |
| `vllm` | High-throughput inference engine สำหรับ RLHF generation server |
| `vector_quantize_pytorch` | ResidualFSQ (Finite Scalar Quantization) ใน codec encoder |
| `torchao` / `torchtune` | Quantization & optimization utilities |

### Audio Processing
| Library | บทบาท |
|---------|--------|
| `librosa` / `torchlibrosa` | Audio feature extraction |
| `silero-vad` | Voice Activity Detection |
| `openai-whisper` / `faster-whisper` | ASR สำหรับ WER reward ใน RLHF |
| `torchmetrics` | DNSMOS (Deep Noise Suppression Mean Opinion Score) |

### NLP & Text Processing
| Library | บทบาท |
|---------|--------|
| `nemo_text_processing` | Text normalization (ตัวเลข→คำ, ย่อ→เต็ม) สำหรับ 6 ภาษา (en/ja/zh/es/fr/de) |
| `pythainlp` | Thai text normalization (normalize ตัวอักษร + num_to_thaiword) |
| `lingua-language-detector` | ตรวจจับภาษาอัตโนมัติ (8 ภาษา) |
| `jiwer` | WER/CER computation |
| `zhconv` / `zhon` | Chinese text normalization |
| `unidecode` | Unicode → ASCII conversion |

### Infrastructure
| Library | บทบาท |
|---------|--------|
| `wandb` | Experiment tracking & logging |
| `absl-py` | Command-line flags & logging |
| `cattrs` | Dataclass ↔ dict serialization สำหรับ config |
| `einops` | Tensor reshaping |
| `uv` | Package manager (แทน pip) |
| `hatch` | Build system & versioning |
| `ruff` | Linter & formatter |

---

## 🏗️ Root Configuration Files

| File | รายละเอียด |
|------|------------|
| `pyproject.toml` | กำหนด dependencies ทั้งหมด (core: torch/transformers/lightning, audio: librosa/silero-vad, NLP: nemo/lingua, RLHF: trl/openai-whisper), build system (hatchling), Ruff config (target Python 3.10, line-length 100) |
| `Makefile` | คำสั่งพัฒนา: `make install` (ใช้ `uv sync`), `make test` (pytest), `make lint`/`lint-fix` (ruff), `make version` (hatch version) |
| `README.md` | คู่มือโปรเจค: สถาปัตยกรรม SpeechLM, setup ด้วย uv, รองรับ CUDA 12.4/12.8, ตัวอย่าง SFT/RLHF training, inference guide |
| `.envrc` | ตั้งค่า dev environment: เพิ่ม `.venv/bin` ใน PATH, export `GIT_ROOT`, ตรวจหา `CUDA_VERSION` จาก nvcc, source `.env` |
| `.tool-versions` | ล็อค Python 3.10.12, uv 0.6.9 |
| `.pre-commit-config.yaml` | Pre-commit hooks: Ruff linting + formatting เฉพาะ `tts/` directory |
| `CONTRIBUTING.md` | แนวทาง contribute: code style, testing, PR process, commit message format |

---

## 📦 `tts/` — Main Package

### `tts/__init__.py`
กำหนด `__version__ = "0.0.1"` — เวอร์ชั่นของ package

---

### 🧠 `tts/core/` — Core ML Logic

#### `constants.py` — ค่าคงที่ส่วนกลาง
ไฟล์นี้ถูกอ้างอิงจากเกือบทุก module ในโปรเจค:
- **Tokenization tokens**: `SPEECH_START_TOKEN` (`<|speech_start|>`), `SPEECH_END_TOKEN` (`<|speech_end|>`), `TEXT_PROMPT_START_TOKEN`/`END`, `VOICE_DESCRIPTION_START_TOKEN`/`END`, `SOUND_EFFECT_START_TOKEN`/`END`, `END_HEADER_ID` — ใช้ใน prompt compilation, tokenization, inference
- **`SPEECH_TOKEN_PATTERN`** = `<|s_{}|>` — template สร้าง speech token เช่น `<|s_142|>`, ใช้ทั้ง tokenizer, prompting, inferencing
- **Audio constants**: `CODEC_SAMPLE_RATE` = 16000 Hz, `CODEC_TOKENS_RATE` = 50 tokens/sec — กำหนดอัตราแปลง audio↔tokens
- **`LOSS_IGNORE_TOKEN_ID`** = -100 — ใช้ mask labels ที่ไม่ต้องการคำนวณ loss (เช่น prompt portion ใน SFT)
- **21 `NONVERBAL_TOKENS`**: `<breathe>`, `<laugh>`, `<cough>`, `<sigh>`, `<yawn>` ฯลฯ — ให้โมเดลสร้างเสียงที่ไม่ใช่คำพูดได้
- **Reward function names**: `WER_REWARD_FUNC`, `DNSMOS_REWARD_FUNC`, `SIMILARITY_REWARD_FUNC` — ใช้ใน RLHF

#### `tokenization.py` — สร้าง Tokenizer
- `build_tokenizer()`: โหลด pretrained tokenizer (เช่น Llama-3.2-1B-Instruct) → เพิ่ม 8 special tokens → เพิ่ม **65,536 speech tokens** (`<|s_0|>` ถึง `<|s_65535|>`) → pad ด้วย `<|extra_token_N|>` จนได้ **vocab size = 193,856** (เป็นผลคูณของ 64 เพื่อ GPU efficiency)
- ตั้ง `pad_token = eos_token`, `padding_side = "right"`
- **ความสัมพันธ์**: ถูกเรียกจาก `main.py` (SFT training) → tokenizer ที่ได้ถูกส่งต่อไปยัง dataset, model, inference ทุกส่วน

#### `modeling.py` — สร้างและโหลดโมเดล
- `_construct_model()`: ใช้ `AutoModelForCausalLM.from_pretrained()` โหลด Llama model, ใช้ **Flash Attention 2** (`_ATTN_IMPLEMENTATION`), resize embeddings ให้ตรงกับ tokenizer vocab size (ทั้ง `embed_tokens` และ `lm_head`)
- `build_model()`: wrapper สำหรับ training — ใช้ `fabric.init_module()` สำหรับ DeepSpeed, รองรับ `gradient_checkpointing_enable()` เพื่อลด VRAM
- `load_model_from_checkpoint()`: โหลดจาก `.pt` checkpoint → (optional) apply LoRA adapter → load state_dict
- `load_tokenizer_config_and_model()`: โหลด tokenizer + `training_config.json` + model + LoRA config จาก checkpoint directory — ใช้โดย `convert_checkpoint.py`
- **ความสัมพันธ์**: เรียกใช้ `lora.py` (apply_lora), `caching.py` (HF cache dir), `configuration.py` (LoraConfig)

#### `lora.py` — LoRA Adapter Management
- `apply_lora()`: ใช้ `peft.get_peft_model()` ด้วย `LoraConfig` — ถ้า `target_modules` ว่าง จะ auto-discover linear modules ทั้งหมด, รองรับโหลด adapter weights จาก `adapter_path`
- `load_lora()` / `save_lora()`: บันทึก/โหลด LoRA weights แยกจาก base model
- **ผลลัพธ์**: fine-tune เฉพาะ LoRA parameters (~0.1-1% ของ total params) แทนที่จะ update ทั้ง model

#### `optimization.py` — Optimizer & LR Scheduler
- `CosineLrScheduler`: linear warmup → cosine decay ถึง 10% ของ LR เริ่มต้น — `get_lr(step)` คืนค่า LR ณ step ปัจจุบัน
- `ConstantLr`: linear warmup → constant LR
- `create_optimizer()`: สร้าง `AdamW` ด้วย `fused=True` (CUDA-optimized) — filter เฉพาะ `requires_grad=True` params
- **ความสัมพันธ์**: ใช้โดย `main.py`, `train_codec.py`

#### `prompting.py` — Prompt Compilation (หัวใจของ Voice Cloning)
ไฟล์นี้กำหนดวิธี **จัดรูปแบบ prompt** ที่ส่งให้ LLM:

- **`TrainingPromptCompiler`** (สำหรับ SFT):
  - User message = transcript ของ prompt audio (+ optional voice description)
  - Assistant message = `<|speech_start|>` + speech tokens + `<|speech_end|>`
  - รวม: `"transcript<|speech_start|><|s_142|><|s_8301|>...<|speech_end|>"`
  - **ไม่ใช้** `text_to_synthesize` — เพราะ SFT เรียนรู้จาก audio + transcript คู่กัน

- **`InferencePromptCompiler`** (สำหรับ inference/RLHF):
  - User message = transcript + text_to_synthesize (เช่น `"Hello world. Please read this new text."`)
  - Assistant message = `<|speech_start|>` + prompt speech tokens (ไม่มี `<|speech_end|>` — ให้ LLM generate ต่อ)
  - **นี่คือกลไก voice cloning**: prompt speech tokens บอก LLM ว่า "เสียงแบบนี้" → LLM สร้าง speech tokens ที่มีลักษณะเสียงเดียวกัน

---

### 🔊 `tts/core/codec/` — Audio Codec Engine

ระบบ codec นี้สร้างจากสถาปัตยกรรม **XCodec2** — แปลงระหว่าง audio waveform ↔ discrete tokens

#### Encoding Pipeline (Audio → Tokens)

**`encoder_modules.py`** — Building blocks:
- `SemanticEncoder`: Conv1d projection (1024→1024 channels) จาก wav2vec features — จับ **ความหมาย** ของเสียง
- `AcousticEncoder`: Multi-stage downsampling (ratios [2,2,4,4,5]) จาก raw waveform ด้วย `EncoderBlock` (strided conv + `ResidualUnit` ด้วย dilated conv [1,3,9] + **Snake activation**) → output 1024-dim — จับ **คุณภาพเสียง** (timbre, pitch)
- `ResidualUnit`: dilated convolution + skip connection สำหรับ multi-scale feature extraction

**`encoder.py`** — `Encoder` class:
- โหลด **Wav2Vec2BertModel** (`facebook/w2v-bert-2.0`) สำหรับ semantic features (ใช้ hidden state layer 16)
- Fuse semantic (1024-dim) + acoustic (1024-dim) → Linear(2048, 2048) → **ResidualFSQ** quantization (levels=[4,4,4,4,4,4,4,4], 1 quantizer → 4^8 = 65,536 codebook entries)
- `encode()`: pad audio → extract wav2vec features → forward → VQ codes (50 codes/sec)
- รองรับ **XCodec2 checkpoint format** (parse `CodecEnc.`, `SemanticEncoder_module.`, `generator.quantizer.`, `fc_prior.` prefixes)

**`encoding.py`** — Interface layer:
- `AudioEncoderInterface` (ABC): กำหนด `encode()`, `sample_rate`, `token_rate`
- `AudioEncoder`: wrapper ของ `Encoder` สำหรับ inference
- `CachingAudioEncoder`: **cache prompt encodings by ID** — ถ้า prompt เดิมถูก encode แล้ว ดึงจาก cache ไม่ต้อง re-encode (สำคัญมากสำหรับ quality validation ที่ใช้ prompt ซ้ำ)

#### Decoding Pipeline (Tokens → Audio)

**`decoder_modules.py`** — Building blocks:
- `ISTFTHead`: ทำนาย magnitude + phase → `torch.istft()` สร้าง waveform — ใช้ `isqrt_s` scaling
- `VocosBackbone`: ConvNeXt-style architecture ด้วย `AdaLayerNorm` (adaptive layer norm), `ConvNeXtBlock` (depthwise conv + pointwise conv + GeLU)
- `Generator`: full pipeline — `ResidualFSQ` (codebook lookup) → `VocosBackbone` → `ISTFTHead`
- `TransformerBlock`: multi-head `Attention` + `MLP` (SiLU activation) + `RMSNorm` — สำหรับ modeling ใน backbone

**`decoder.py`** — `Decoder` & `TrainableDecoder`:
- `Decoder`: VQ codes → codebook lookup (`quantizer.get_output_from_indices`) → `fc_post_a` (Linear 2048→1024) → `VocosBackbone` → optional `UpSamplerBlock` → `ISTFTHead` → waveform
- ตรวจสอบ: `sample_rate / hop_length / total_ups == 50` (ต้องได้ 50 tokens/sec)
- `TrainableDecoder` (GAN training wrapper): รวม Decoder + MPD + MSD + losses
  - **Training step**: forward → compute disc loss (backprop) → compute gen loss (backprop) — ใน single step
  - **Generator losses**: mel_loss (×15), adv_loss (×1), fm_loss (×1), rms_loss (×1)
  - **Discriminator loss**: hinge loss (×1)
  - `create_optimizer()`: **freeze VQ quantizer** — train เฉพาะ backbone/upsampler/head/fc_post_a + discriminators

**`criterion.py`** — Loss functions:
- `GANLoss`: hinge loss — `disc_loss` = max(0, 1-real) + max(0, 1+fake), `gen_loss` = -fake
- `MultiResolutionMelSpectrogramLoss`: mel spectrogram L1 loss ที่ FFT sizes หลายขนาด
- `MultiResolutionSTFTLoss`: spectral convergence + log STFT magnitude loss

**`discriminator.py`** — GAN discriminators:
- `HiFiGANPeriodDiscriminator` (MPD): 5 sub-discriminators ที่ periods [2,3,5,7,11] — reshape 1D→2D แล้ว conv บนแต่ละ period
- `SpecDiscriminator` (MSD): 8 STFT resolutions (FFT sizes 78→2296) — spectral discrimination

**`activations.py`**: `Snake`/`SnakeBeta` = x + sin²(αx)/α (periodic, trainable α), `Activation1d` = activation + anti-aliased resampling

**`filters.py`**: Kaiser-windowed sinc filter, `LowPassFilter1d`, `UpSample1d`/`DownSample1d` — anti-aliasing

**`upsampler.py`**: `ConvTranspose1d` + `ResnetBlock` sequence — upsample codec features (e.g. factors [3,2] → 6×)

**`decoding.py`**: `AudioDecoderInterface` (ABC), `AudioDecoder` (wrapper), `create()` factory

---

### 📊 `tts/data/` — Data Pipeline

#### `data_sample.py` — Data Structure
`Sample` dataclass มีฟิลด์: `id`, `wav_path`, `speaker_id`, `language`, `emotion`, `transcript`, `voice_description`, `sound_effect`, `duration`, `sample_rate`, `dataset_name`, `dnsmos_mos_ovr`, `style`
- Validate: ต้องมีอย่างน้อย 1 ใน transcript/voice_description/sound_effect
- `from_json()`: รองรับ relative path (prepend dataset_dir)
- `to_json()`: serialize กลับเป็น dict

#### `data_utils.py` — Data Loading Utilities
- `load_samples()`: อ่าน `.jsonl` → apply filters → return (samples, total_hours)
- `load_and_filter_audio_codes_and_samples()`: โหลด pre-vectorized codes (`_codes.npy` memmap) + index (`_codes_index.npy`) + samples → apply configurable filters (language, sample_rate, DNSMOS, duration) → return (samples, codes_memmap, index_pairs)
- `load_wav()`: `torchaudio.load()` + optional resample via `torchaudio.functional.resample()`
- `chunk_work()`: แบ่ง work items ให้ distributed workers ตาม rank

#### `filtering.py` — Dataset Filters
- Static: `filter_empty_transcript`, `filter_non_english` (ไม่ใช้แล้วใน `load_samples()` — ยังคงมีฟังก์ชันอยู่แต่ถูกลบออกจาก pipeline), `filter_long_duration` (>30s), `filter_punct_or_space_only_transcript`
- Factory (closures): `filter_allowed_languages(langs)`, `filter_min_sample_rate(rate)`, `filter_min_dnsmos_score(score)`, `filter_min_audio_duration(dur)`
- แต่ละ filter return `None` (pass) หรือ reason string (filtered out)

#### `text_normalization.py` — Text Normalization
- `NemoTextNormalizer`: ใช้ NVIDIA NeMo ITN สำหรับ 6 ภาษา (en/ja/zh/es/fr/de) + **Thai ใช้ `pythainlp`** (`pythainlp.util.normalize` + `num_to_thaiword`) — รวม 7 ภาษา, auto-detect ภาษาด้วย `lingua` (8 ภาษา)
- `NoOpTextNormalizer`: ไม่ทำอะไร (bypass)
- `create_text_normalizer(enable)`: factory

#### `tts_datasets.py` — Dataset Assembly
- `_build_dataset()`: route ไปยัง pretraining/finetuning/RLHF dataset ตาม config flags
- `WeightedDataset`: dataset + epoch weight (เช่น weight=2.0 → ใช้ dataset 2 รอบ)
- `CombinedDataset`: virtually concatenate weighted datasets, รองรับ `fast_forward(steps)` สำหรับ resume จาก checkpoint
- `get_collate_fn()`: pad `input_ids`/`labels`/`attention_mask` ใน batch ให้ยาวเท่ากัน

#### `tts/data/datasets/` — Dataset Implementations

**`finetuning.py`** — SFT Datasets:
- `TtsFineTuningDataset`: โหลด pre-vectorized codes + samples → compile prompt ด้วย `TrainingPromptCompiler` → tokenize → **mask labels** ก่อน `<|speech_start|>` ด้วย `LOSS_IGNORE_TOKEN_ID` (-100) เพื่อให้ loss คำนวณเฉพาะ speech tokens
- `TextFineTuningDataset`: สำหรับ text-only SFT (เช่น OIG dataset) — parse `<human>:`/`<bot>:` format → `apply_chat_template()` → mask non-assistant tokens

**`pretraining.py`** — Pretraining Datasets:
- `TtsPretrainingDataset`: next-token prediction บน raw speech codes จาก `.npy` memmap — แปลง codec index → speech token IDs
- `TextPretrainingDataset`: next-token prediction บน tokenized text จาก `.npy` memmap

**`rlhf.py`** — RLHF Dataset:
- `TtsRLHFDataset`: ใช้ `InferencePromptCompiler` — ใช้ sample ปัจจุบันเป็น prompt audio, ใช้ **sample ถัดไป** เป็น `text_to_synthesize` — return `prompt` string + `prompt_speech_ids` + `completion_truth` + `prompt_wav_path` + `language` + `prompt_transcript` สำหรับ GRPO training

---

### 🎤 `tts/inference/` — Inference Engine

#### `inferencing.py` — Core Inference Logic
- `InferenceSettings`: temperature (0.8), max_tokens (1792), min_tokens (10), top_p (1.0), top_k (50), repetition_penalty (1.1), frequency_penalty (0.3), seed (42)
- `extract_speech_ids()`: parse `<|s_N|>` tokens → integers — ตรวจสอบ format ที่ถูกต้อง
- `_generate_speech_tokens()`: รองรับ 2 backends:
  - **HuggingFace**: `model.generate()` ด้วย `do_sample=True`, eos=`speech_end_id`
  - **vLLM**: `vllm.SamplingParams` + `model.generate(prompt_token_ids=...)`
- `_synthesize_audio()`: tokenize prompt → generate speech tokens → **extract generated portion** (ตัด input_ids ออก) → decode tokens → **ตัด prompt audio portion** ออก (คำนวณจาก `len(speech_ids) / token_rate * sample_rate`)
- `LocalTtsModel`: orchestrator class — `synthesize_speech()`: encode prompt → compile prompt → generate → decode → return `InferenceResult` (wav + timing)
- `complete_prompt()`: audio continuation mode — ไม่มี text input, แค่ต่อเสียงจาก prompt

#### `quality_validation.py` — Training Quality Monitoring
- `RandomPhrasesSynthesizer`: ทุก checkpoint step → สร้างเสียงจาก **19 test phrases × 3 prompt voices** → บันทึก `.wav` → กระจายงานข้าม ranks — ใช้ `LocalTtsModel` จริง
- `PromptContinuationValidator`: ต่อเสียงจาก prompt โดยไม่มี text — ทดสอบ pretraining quality
- `_unwrap_model()`: unwrap DDP/FSDP/torch.compile wrapper เพื่อ access model จริง
- `create_quality_validator()`: factory ตาม `validation_type` ("random_phrases" หรือ "prompt_continuation")

---

### 🏋️ `tts/training/` — Training Pipeline

#### `main.py` — SFT Training Entry Point
ลำดับการทำงาน: Config → Hardware (Fabric) → Tokenizer (`build_tokenizer`) → Model (`build_model` + optional LoRA) → Datasets (`merge_datasets`) → Optimizer (AdamW + CosineLR) → Quality Validator → `training_loop.run()` → Save final checkpoint
- Flags: `--config_path`, `--run_name`, `--experiment_dir`, `--pretraining_mode`, `--compile_model`, `--use_wandb`, `--slurm_distributed`, `--seed`

#### `training_loop.py` — Main Training Loop
- `run()`: หลัก loop — คำนวณ LR (cosine schedule) → `_train_micro_batch()` → logging → evaluation → checkpointing → quality validation
- `_train_micro_batch()`: gradient accumulation ด้วย `fabric.no_backward_sync()` — forward pass → backward → clip gradients → optimizer step
- `_resume_from_checkpoint()`: โหลด checkpoint → fast-forward dataloader ด้วย `CombinedDataset.fast_forward()`
- Metrics: `loss`, `tokens_processed`, `audio_processed_sec` — tracked per-source (dataset name)

#### `checkpointing.py` — Model Checkpointing
- `save_to_checkpoint()`: `fabric.save()` → model + optimizer + statistics + config — optional ลบ checkpoint เก่า (`keep_only_last_n_checkpoints`)
- `load_from_checkpoint()`: โหลด full (model+optimizer+stats) หรือ weights-only
- `save_config()`: เขียน `training_config.json` + optional upload ไป WandB

#### `environment.py` — Distributed Environment
- `EnvironmentContext`: encapsulate rank/device/world_size/is_main_process
- `initialize_distributed_environment_context()`: ตรวจจับ SLURM (`SLURM_PROCID`) หรือ local → init CUDA (tf32=True, cudnn.benchmark=True)
- `initialize_fabric()`: สร้าง Lightning Fabric ด้วย strategy DDP/FSDP/DeepSpeed, precision (bf16/fp32)

#### `evaluation.py` — Model Evaluation
- `compute_metrics()`: eval loss บน validation set + optional health stats
- `_get_health_stats()`: max/avg absolute gradients + parameters ข้าม ranks (ตรวจสอบ training stability)
- `_estimate_eval_loss()`: per-source eval loss tracking

#### Codec GAN Training (`tts/training/codec/`)

**`train_codec.py`**: Entry point — `decoder.create()` สร้าง TrainableDecoder (Generator+MPD+MSD) → โหลด datasets → สร้าง dual optimizers (gen+disc) → `gan_training_loop.run()` → save final + codec model config JSON

**`gan_training_loop.py`**: GAN-specific loop — ทุก step: zero_grad ทั้ง gen+disc → micro-batch training (forward → disc loss → gen loss → backward ทั้งคู่) → clip gradients ทั้งคู่ → step ทั้งคู่ — log 6 metrics: disc_loss, gen_loss, fm_loss, mel_loss, adv_loss, rms_loss

**`codec_datasets.py`**: `CodecTrainingDataset` — โหลด wav + codes → random-crop ไปที่ `audio_window_size` → repeat ถ้าสั้นเกินไป — `collate_fn` stack audio_codes + wav

**`codec_quality_validation.py`**: `FixedBatchCodecValidator` — reconstruct audio จาก fixed batch → save generated vs true WAVs เปรียบเทียบ

#### RLHF Training (`tts/training/rlhf/`)

**`rlhf_main.py`**: Entry point — ใช้ TRL `GRPOTrainer` (Group Relative Policy Optimization) + `GRPOConfig` — โหลด tokenizer จาก base model → สร้าง RLHF dataset → สร้าง reward functions → train — รองรับ **vLLM server** แยก node สำหรับ generation

**`rewards.py`** — 3 Reward Functions:
- `WERRewardFunc`: decode speech tokens → audio → **Whisper large-v3** transcribe → WER/CER → reward = exp(-2.5 × WER) — ยิ่ง WER ต่ำ reward ยิ่งสูง
- `DNSMOSRewardFunc`: decode → **DNSMOS** (audio quality score 1-5) → normalize [0,1] — วัดคุณภาพเสียง
- `SimilarityRewardFunc`: decode → **ECAPA-TDNN** speaker embedding → cosine similarity กับ prompt audio → normalize [0,1] — วัดความเหมือนเสียง
- `RewardFunc` (base): ทุก reward function มี `_decode_audio()` ที่ใช้ codec decoder แปลง generated tokens → wav

**`reward_utils.py`**: transcription ด้วย Whisper, text normalization (Chinese→Simplified, remove punctuation, CER สำหรับ zh/ja/ko/th), DNSMOS via `torchmetrics`, cosine similarity computation

**`ecapa_tdnn.py`**: Third-party ECAPA-TDNN model สำหรับ speaker verification — ใช้ **WavLM-Large** features (s3prl), SE-Res2Blocks, AttentiveStatsPool → speaker embedding vector

**`run_rlhf_combine.sh`**: SLURM script 2-node — Node 0: `accelerate launch` GRPO training, Node 1: `trl vllm-serve` generation server

---

### 🔧 `tts/utils/` — Utilities

#### `configuration.py` — All Config Dataclasses
- `ExperimentConfig`: top-level — รวม TrainingConfig + ModelingConfig + CheckpointingConfig + DatasetConfig + optional LoraConfig/RLHFConfig/CodecTrainingConfig
- `TrainingConfig`: seed, logging_steps, eval_steps, gradient_accumulation_steps, gradient_clip_value, learning_rate, betas, warmup_ratio, batch_size, weight_decay, precision, strategy (DDP/FSDP/DeepSpeed), gradient_checkpointing, num_workers
- `CheckpointingConfig`: save_steps, directory, collect_health_stats, save_intermediate_generations, validation_type, keep_only_last_n_checkpoints, checkpoint_file_to_resume_from, only_load_model_weights
- `DatasetConfig`: allowed_languages, min_dnsmos_score, min_sample_rate, enable_rlhf_training, min_audio_duration, allowed_ift_annotations
- `RLHFConfig`: base_model_dir, top_p/k, repetition_penalty, temperature, num_generations, max_prompt/completion_length, use_vllm, reward_funcs/weights, kl_beta
- `LoraConfig`: task_type, r (rank), lora_alpha, target_modules, lora_dropout, bias, adapter_path
- `CodecTrainingConfig`: audio_window_size, sample_rate, hop_length, upsample_factors, kernel_sizes
- `TrainingStrategy` enum: DDP, FSDP, DEEPSPEED
- `from_json()`: โหลด config → validate required keys → reset dynamic variables
- `maybe_setup_wandb_and_update_config()`: init WandB ด้วย project_name + run_name

#### `custom_logging.py` — Logging & Statistics
- `Statistics`: track per-source metrics (loss per dataset source + total) — serializable สำหรับ checkpoint resume
- `Timer`: context manager วัดเวลา
- `get_logging_stats()`: all-reduce averaged metrics ข้าม ranks
- `_HostnameLogFormatter`: เพิ่ม hostname + rank ใน log messages

---

## 🧹 `setup/` — Environment Automation

#### `setup_python.sh` — The Project Housekeeper
- **หน้าที่หลัก**: ล้างบาง (Delete) `.venv` เก่าทิ้ง และสร้างใหม่พร้อมติดตั้ง Library ทั้งหมดแบบ Clean Install (ป้องกันปัญหา Library ตีกัน)
- **Features**:
  - บังคับใช้ **Python 3.10** + **uv** (Package manager ใหม่ที่เร็วกว่า pip แบบก้าวกระโดด)
  - **Cross-Platform**: เช็คว่าเป็น Mac (ลง PyTorch แบบ CPU-only) หรือ Linux/Windows (ลง PyTorch แบบมี CUDA) อัตโนมัติ
  - **Auto-Config CUDA**: เลือกโหลด PyTorch เวอร์ชั่นที่ตรงกับ CUDA (12.4 หรือ 12.8) 
  - **Flash-Attention Shortcut**: โหลด Flash-Attention รุ่น Prebuilt (สำเร็จรูป) มาลงให้ทันที ช่วยประหยัดเวลา Compile จาก 30 นาทีเหลือ 10 วินาที (เฉพาะ Linux)

---

## 🛠️ `tools/` — CLI Utilities

#### `data/data_vectorizer.py` — Batch Audio Encoder
- โหลด raw audio samples จาก `.jsonl` → สร้าง `WaveDataset` (pad audio, extract wav2vec features ด้วย `AutoFeatureExtractor`) → batch encode ด้วย codec encoder → save sharded `{split}_codes_{rank}.npy` + `{split}_codes_index_{rank}.npy` + `{split}_samples_{rank}.jsonl`
- Auto split train/val (default 99.9%/0.1%)
- รองรับ SLURM distributed processing + WandB logging
- **ขั้นตอนแรกของ data pipeline** — ต้องรันก่อน training

#### `data/data_merger.py` — Shard Merger
- อ่าน sharded files จาก vectorizer → validate consistency → concatenate เป็น single `train_codes.npy`/`val_codes.npy` + index + samples
- Validate: ตรวจสอบ codes เป็น contiguous, จำนวน samples = codes_index
- Optional: `--remove_shards` ลบ shard files หลัง merge

#### `serving/convert_checkpoint.py` — Checkpoint Converter
- แปลง training checkpoint → HuggingFace serving format (safetensors)
- `--merge_lora`: merge LoRA weights เข้า base model (ไม่ต้องโหลด adapter ตอน serve)
- `--update_eos_token`: ตั้ง EOS = `<|speech_end|>` (สำคัญสำหรับ vLLM serving)
- `--add_missing_nonverbal_tokens`: เพิ่ม nonverbal tokens + pad vocab ให้เป็นผลคูณของ 64

#### `serving/inference.py` — Standalone Inference Script
- Full CLI: `--model_checkpoint_path`, `--audio_encoder_path`, `--audio_decoder_path`, `--prompt_wav_path`, `--prompt_transcription`, `--text`, `--output_path`
- รองรับ HuggingFace model หรือ vLLM
- สร้าง `LocalTtsModel` → `synthesize_speech()` → save WAV

---

## 📝 `example/` — Example Configs & Data

| File | รายละเอียด |
|------|------------|
| `configs/sft.json` | SFT config: Llama-3.2-1B-Instruct, codebook=65536, max_seq_len=2048, bf16, batch=4, LR=1e-4, warmup 5%, cosine decay, DDP |
| `configs/rlhf.json` | RLHF config: GRPO, vLLM, 8 generations/prompt, WER reward, KL beta=0.04, LR=1e-6 |
| `configs/codec_training_config.json` | Codec training: 48kHz output, hop=160, upsample [3,2], fp32, batch=32 |
| `configs/samples.jsonl` | 100 English audio samples (transcripts, durations 0.5-20s, 24kHz) สำหรับทดสอบ pipeline |
| `codec/model_config.json` | Codec model config: 16kHz, 50 tokens/sec, hop=320 |

---

## 🔄 Complete Data Flow

```
┌─────────────── DATA PREPARATION ───────────────┐
│                                                  │
│  Raw Audio (.wav) + Transcripts (samples.jsonl)  │
│                       │                          │
│                       ▼                          │
│         [tools/data/data_vectorizer.py]           │
│    (Wav2Vec features → Codec Encoder → VQ codes) │
│                       │                          │
│                       ▼                          │
│     Sharded codes_N.npy + index_N.npy + samples  │
│                       │                          │
│                       ▼                          │
│          [tools/data/data_merger.py]              │
│        (Concatenate shards → single files)       │
│                       │                          │
│                       ▼                          │
│  train_codes.npy + train_codes_index.npy         │
│  + train_samples.jsonl (+ val equivalents)       │
└──────────────────────┬───────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ SFT      │  │ RLHF     │  │ Codec    │
   │ Training │  │ Training │  │ Training │
   │ main.py  │  │ rlhf_    │  │ train_   │
   │          │  │ main.py  │  │ codec.py │
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        │              │              │
        ▼              ▼              ▼
   checkpoint.pt  HF Model Dir  decoder.pt
        │              │
        ▼              ▼
  [convert_checkpoint.py]
        │
        ▼
  Serving Model (safetensors + tokenizer)
        │
        ▼
  [inference.py / LocalTtsModel]
  Prompt WAV → Encode → LLM Generate → Decode → Output WAV
```

---

## 🔑 Key Design Insights

### ทำไม Speech Token Approach ถึงดีกว่า Traditional TTS?
- **LLM ทำหน้าที่เป็น "สมอง"** — เข้าใจ context, prosody, emotion จาก text
- **Codec ทำหน้าที่เป็น "ปากและหู"** — แปลงระหว่าง audio ↔ discrete tokens
- **Zero-shot** เพราะ LLM เรียนรู้ relationship ระหว่าง prompt speech tokens กับ generated speech tokens ตอน training → ตอน inference แค่ให้ prompt tokens ใหม่ LLM ก็ "clone" เสียงได้

### Training Strategy ที่รองรับ
- **SFT (Supervised Fine-Tuning)**: เรียนรู้จาก (transcript, audio) pairs — LoRA หรือ full fine-tuning
- **RLHF (GRPO)**: ปรับปรุงคุณภาพด้วย reward signals (WER, DNSMOS, Speaker Similarity)
- **Codec Decoder Training**: GAN training เพื่อปรับปรุง audio reconstruction quality
- **Pretraining**: next-token prediction บน speech tokens หรือ text (ขยายความรู้ของ base LLM)

---

## ⚠️ กับดัก (Traps), ข้อจำกัด & Hardcoded Values

> **สำคัญมาก**: ส่วนนี้รวบรวมค่า hardcoded, ข้อบังคับ, และจุดอ่อนที่ซ่อนอยู่ในโค้ด ซึ่งอาจทำให้การปรับใช้หรือขยายโปรเจคล้มเหลวโดยไม่คาดคิด

### 🚫 Hardcoded Trap #1: `load_samples()` บังคับ English-only

**ไฟล์**: `tts/data/data_utils.py` บรรทัด 41-45
```python
filters = [
    filtering.filter_empty_transcript,
    # filter_non_english ถูกลบออกแล้ว ✅
    filtering.filter_long_duration,
    filtering.filter_punct_or_space_only_transcript,
]
```
- ฟังก์ชัน `load_samples()` ที่ใช้โดย `data_vectorizer.py` (data preparation step แรก) มี `filter_non_english` **hardcoded** อยู่ในรายการ filter
- `filter_non_english` คือ: `return "non_english" if sample.language != "en" else None`
- **ผลกระทบ**: ~~ข้อมูลทุกภาษาที่ไม่ใช่ English จะถูก **กรองทิ้ง** ตอน vectorize~~ → **แก้ไขแล้ว**: ลบ `filter_non_english` ออกจาก `load_samples()` เพื่อรองรับทุกภาษา
- **สถานะ**: ✅ แก้ไขแล้ว

> ⚡ **หมายเหตุ**: `load_and_filter_audio_codes_and_samples()` (ใช้ตอน training) ใช้ `filter_allowed_languages()` จาก config แทน ดังนั้นการควบคุมภาษายังทำได้ผ่าน `allowed_languages` ใน config JSON

### 🚫 Hardcoded Trap #2: Vocab Size ล็อคที่ 193,856

**ไฟล์**: `tts/core/tokenization.py` บรรทัด 8
```python
_EXPECTED_VOCAB_SIZE = 193856
```
- Tokenizer **บังคับ** ให้ vocab size ต้องเป็น 193,856 พอดี — ถ้าไม่ตรงจะ `raise ValueError`
- ค่านี้คำนวณจาก: Llama base vocab (~128K) + 8 special tokens + 65,536 speech tokens + extra padding tokens
- **ผลกระทบ**: ถ้าเปลี่ยน base model ที่มี vocab size ต่างจาก Llama 3.x จะ error ทันที
- **วิธีแก้**: เปลี่ยน `_EXPECTED_VOCAB_SIZE` ให้ตรงกับ base model ใหม่ หรือทำให้คำนวณ dynamic

### 🚫 Hardcoded Trap #3: Flash Attention 2 บังคับ

**ไฟล์**: `tts/core/modeling.py` บรรทัด 15
```python
_ATTN_IMPLEMENTATION = "flash_attention_2"
```
- Model **บังคับ** ใช้ Flash Attention 2 เสมอ — ไม่สามารถเลือก attention implementation อื่นได้
- **ผลกระทบ**: ต้องมี `flash-attn` package และ GPU ที่รองรับ (Ampere+, CUDA) — ไม่สามารถรันบน CPU หรือ GPU รุ่นเก่า
- **วิธีแก้**: ทำให้เป็น configurable parameter ใน config JSON

### 🚫 Hardcoded Trap #4: Codec Encoder ล็อคค่าหลายตัว

**ไฟล์**: `tts/core/codec/encoder.py`
```python
_HOP_LENGTH = 320                              # บรรทัด 13
semantic_encoder: input/output_channels=1024   # บรรทัด 28-32
acoustic_encoder: up_ratios=[2,2,4,4,5]        # บรรทัด 38
fusion_layer = nn.Linear(2048, 2048)           # บรรทัด 42
quantizer: levels=[4,4,4,4,4,4,4,4]            # บรรทัด 46
wav2vec: "facebook/w2v-bert-2.0"               # บรรทัด 52-55
hidden_states[16]                               # บรรทัด 63
```
- **ทุกค่าข้างต้น hardcoded** — ไม่สามารถเปลี่ยนผ่าน config ได้
- โดยเฉพาะ `w2v-bert-2.0` และ `hidden_states[16]` (ใช้ layer 16 เฉพาะ) — ผูกติดกับ pretrained model นี้
- Quantizer levels `[4]*8` → codebook size = 4^8 = 65,536 — ตรงกับจำนวน speech tokens ใน tokenizer
- **ผลกระทบ**: ถ้าต้องการเปลี่ยน codec architecture ต้องแก้โค้ดโดยตรง + แก้ `codebook_size` ใน config + แก้ `_EXPECTED_VOCAB_SIZE`

### 🚫 Hardcoded Trap #5: Codec Decoder GAN Loss Weights

**ไฟล์**: `tts/core/codec/decoder.py` บรรทัด 148-157
```python
# TODO: add them to config json instead of hardcoding here.
self.lambda_disc = 1.0
self.lambda_fm = 1.0
self.lambda_mel = 15.0
self.lambda_adv = 1.0
self.lambda_rms = 1.0
self.grad_clip_disc = 1.0
self.grad_clip_gen = 1.0
```
- **ผู้พัฒนาดั้งเดิมระบุ TODO ไว้เอง** ว่าควรย้ายไป config แต่ยังไม่ได้ทำ
- ถ้าต้อง tune GAN loss weights ต้องแก้โค้ดโดยตรง

### 🚫 Hardcoded Trap #6: Quality Validation Paths ไม่มีจริง

**ไฟล์**: `tts/inference/quality_validation.py` บรรทัด 14-26
```python
_DEFAULT_ENCODER_CHECKPOINT_PATH = "/path/to/some-982760.pt"
_DEFAULT_DECODER_CHECKPOINT_PATH = "/path/to/some-9a5f5d.pt"
_DEFAULT_PROMPT_WAVS = {
    "/path/to/some-91f247.wav": "It was extremely dark...",
    "/path/to/some-438d0c.wav": "Unveiling our summer...",
    "/path/to/some-c3f992.wav": "Dogs are sitting...",
}
```
- Paths **ชี้ไปที่ไฟล์ที่ไม่มีอยู่จริง** — เป็น placeholder ของ Inworld internal paths
- **ผลกระทบ**: ถ้าเปิด `save_intermediate_generations: true` ใน config → quality validation จะ **crash** เพราะหาไฟล์ไม่เจอ
- **วิธีแก้**: ปิด `save_intermediate_generations` หรือแก้ paths ให้ชี้ไปที่ไฟล์จริง

### 🚫 Hardcoded Trap #7: RLHF Reward Default Paths

**ไฟล์**: `tts/training/rlhf/rewards.py` บรรทัด 20-21
```python
_DEFAULT_CODEC_CHECKPOINT_PATH = "/path/to/some-9a5f5d.pt"
_DEFAULT_SIM_CHECKPOINT_PATH = "/path/to/some-3aac85.pth"
```
- Codec decoder path และ speaker similarity model path เป็น placeholder
- **ผลกระทบ**: RLHF training จะ crash ถ้าไม่แก้ paths เหล่านี้

### 🚫 Hardcoded Trap #8: Codec constants ที่ต้องสัมพันธ์กัน

**ไฟล์หลายไฟล์ที่ต้องตรงกัน**:
```
constants.py:     CODEC_SAMPLE_RATE = 16000, CODEC_TOKENS_RATE = 50
encoder.py:       _HOP_LENGTH = 320  (16000 / 320 = 50 ✓)
decoder.py:       sample_rate / hop_length / prod(upsample_factors) == 50 (บังคับ)
criterion.py:     sample_rate=16000 (hardcoded ใน MultiResolutionMelSpectrogramLoss)
```
- **ข้อบังคับ**: `sample_rate / hop_length / product(upsample_factors) == 50` → ต้องได้ 50 tokens/sec เสมอ
- ถ้าเปลี่ยน sample rate ต้องปรับ hop_length + upsample_factors ให้สอดคล้อง

### 🚫 Hardcoded Trap #9: DDP Memory Leak (Known Bug)

**ไฟล์**: `tts/core/modeling.py` บรรทัด 134
```python
# TODO: debug why using DDP multi-GPU training leaks memory.
model_init_context = contextlib.nullcontext()
```
- **Bug ที่ยังไม่ได้แก้**: DDP multi-GPU training มี memory leak
- ปัจจุบัน `fabric.init_module()` ถูกใช้เฉพาะ single-GPU หรือ DeepSpeed เท่านั้น

### 🚫 Hardcoded Trap #10: Codec GAN Loss/Discriminator Windows

**ไฟล์**: `tts/core/codec/criterion.py` และ `tts/core/codec/discriminator.py`
```python
hop_sizes = [120, 240, 50]
win_lengths = [600, 1200, 240]
max_downsample_channels = 512
```
- `MultiResolutionMelSpectrogramLoss` และ `SpecDiscriminator` (MSD) มีการใช้ค่า hop_sizes และ win_lengths แบบ **hardcoded**
- **ผลกระทบ**: ค่าเหล่านี้ถูกจูนมาสำหรับ `CODEC_SAMPLE_RATE` ที่ 16kHz เท่านั้น หากมีการเปลี่ยน sample_rate ของ codec ค่า parameter หรือขนาด window ของ FFT เหล่านี้อาจทำให้ไม่ได้ประสิทธิภาพสูงสุดและต้องแก้โค้ดโดยตรง

### 🚫 Hardcoded Trap #11: WandB Default Project Name (Env Var Dependency)

**ไฟล์**: `tts/utils/configuration.py` บรรทัด 319-323
```python
if not project_name:
    if "WANDB_PROJECT" in os.environ:
        project_name = os.environ["WANDB_PROJECT"]
    else:
        project_name = "inworld_{}".format(os.environ["USER"])
```
- หากไม่ได้ระบุ `project_name` ชัดเจน ระบบจะพยายามดึงตัวแปรสภาพแวดล้อม `USER` มาตั้งชื่อ project ให้กับ WandB
- **ผลกระทบ**: ถ้ารันโปรเจคใน Docker container หรือสภาพแวดล้อมที่ **ไม่มีตัวแปร `USER` (เช่น CI/CD pipelines)** โปรเซสจะ **crash** ทันทีด้วย `KeyError: 'USER'`
- **วิธีแก้**: ตั้งค่า `WANDB_PROJECT` เสมอเวลาจะใช้ `use_wandb=True` หรือแก้โค้ดให้มี fallback เป็น `"inworld_tts"`

### 🚫 Hardcoded Trap #12: SLURM Distributed Environment Fallbacks

**ไฟล์**: `tts/training/environment.py` บรรทัด 205-206
- โปรเจคนี้ถูกออกแบบมาเพื่อรันบน HPC cluster ที่ใช้ **SLURM** เป็นหลัก
- ถ้าใช้ flag `--slurm_distributed` โค้ดบังคับการมีตัวแปร `SLURM_JOB_NUM_NODES` และ `SLURM_NTASKS_PER_NODE`
- สำหรับ Local Multi-GPU ระบบใช้วิธี detect ผ่าน `LOCAL_RANK` หรือ `RANK` และ fallback `world_size = 1` หากไม่เจอ (ทำให้รัน DDP ผิดพลาดกลายเป็น Single GPU ได้หากรัน script ผิดวิธี)

---

## 🌐 ข้อจำกัดด้านภาษา (Language Limitations)

### Text Normalization — รองรับ 7 ภาษา

**ไฟล์**: `tts/data/text_normalization.py`
```python
_supported_languages = ["en", "ja", "zh", "es", "fr", "de", "th"]
```
- NeMo Text Normalizer รองรับ 6 ภาษา: English, Japanese, Chinese, Spanish, French, German
- **Thai** ใช้ `pythainlp` (`pythainlp.util.normalize` + `num_to_thaiword`) แทน NeMo — รองรับ normalization ตัวอักษรไทยและแปลงตัวเลขเป็นคำไทย
- ภาษาอื่น (เช่น Korean, Arabic) → `normalize_with_language()` จะ **return text เดิม** โดยไม่ normalize
- **ผลกระทบ**: ตัวเลข, ตัวย่อ, สัญลักษณ์ ในภาษาที่ไม่รองรับจะไม่ถูกแปลง → LLM อาจออกเสียงผิด

### Language Detection — 8 ภาษา

**ไฟล์**: `tts/data/text_normalization.py`
```python
self.lang_detector = lingua.LanguageDetectorBuilder.from_languages(
    lingua.Language.KOREAN,    # detect ได้แต่ไม่มี normalizer
    lingua.Language.JAPANESE,
    lingua.Language.CHINESE,
    lingua.Language.ENGLISH,
    lingua.Language.SPANISH,
    lingua.Language.FRENCH,
    lingua.Language.GERMAN,
    lingua.Language.THAI,      # ✅ detect + normalize ด้วย pythainlp
).build()
```
- Language detector รองรับ 8 ภาษา — Korean ไม่มี normalizer → detect ได้ แต่ fall through ไป `return text`

### ~~English-only Bias ใน Data Pipeline~~ (แก้ไขแล้ว)

- ~~`load_samples()` (data_utils.py) → hardcoded `filter_non_english`~~ → **ลบออกแล้ว** ข้อมูลทุกภาษาผ่าน vectorization ได้
- `convert_to_ascii()` ใน NemoTextNormalizer → ใช้ `unidecode` แปลง Unicode เป็น ASCII **เฉพาะ English** → ภาษาอื่นจะไม่ถูก convert (เป็นพฤติกรรมที่ถูกต้อง)

### RLHF WER/CER Reward — รองรับหลายภาษา

**ไฟล์**: `tts/training/rlhf/reward_utils.py`
- WER reward ใช้ `openai-whisper` large-v3 — รองรับหลายภาษา
- CER ใช้สำหรับภาษาที่ไม่มี word boundary: `["zh", "ja", "ko", "th"]` (เพิ่ม Thai แล้ว)
- RLHF dataset (`rlhf.py`) ส่ง `language` field ไปยัง reward functions เพื่อเลือก WER/CER ได้ถูกต้อง

### RLHF Config ตัวอย่าง — ล็อค English

**ไฟล์**: `example/configs/rlhf.json`
```json
"allowed_languages": ["en"],
"enable_asr_agreement": true
```
- เปลี่ยน `allowed_languages` เป็น `["en", "th"]` เพื่อรองรับ Thai

---

## 📋 TODO Items ที่ผู้พัฒนาเดิมทิ้งไว้ (ตรวจพบในโค้ด)

| ไฟล์ | บรรทัด | TODO |
|------|--------|------|
| `modeling.py` | 134 | debug DDP multi-GPU memory leak |
| `decoder.py` | 148 | ย้าย GAN loss weights ไป config JSON |
| `training_loop.py` | 56 | หาวิธีที่ elegant กว่าสำหรับ internal mechansim |
| `training_loop.py` | 222 | eval อาจทำเฉพาะ main rank ก็พอ |
| `training_loop.py` | 306 | best eval loss ไม่ถูก save ถ้า step ไม่ตรงกับ eval_steps |
| `environment.py` | 85 | ทำให้ CUDA settings configurable จาก config file |
| `checkpointing.py` | 12 | โหลด tokenizer ด้วยเพื่อลด human mistakes |
| `checkpointing.py` | 83 | wandb sweep value อาจถูก override โดย training code |
| `quality_validation.py` | 135 | ปรับปรุงให้ cover voice description use case |
| `quality_validation.py` | 207 | รองรับ batch inference |
| `gan_training_loop.py` | 81 | เพิ่ม source ให้ statistics |
| `gan_training_loop.py` | 232 | best eval loss ไม่ถูก save (เหมือน training_loop) |
| `train_codec.py` | 191 | ทดสอบ torch.compile สำหรับ training |
| `train_codec.py` | 278 | เพิ่ม sweeps support |
| `tts_datasets.py` | 286 | หาวิธีที่ elegant กว่า |
| `codec_datasets.py` | 88 | ตรวจสอบว่าต้อง extra padding หรือไม่ |

---

## 🔧 คู่มือการบำรุงรักษา: เพิ่มภาษาใหม่

### ขั้นตอนที่ 1: แก้ Data Pipeline ให้รับภาษาใหม่
1. ~~**ลบ `filter_non_english`** ออกจาก `tts/data/data_utils.py`~~ ✅ ทำแล้ว
2. **เพิ่ม language code** ใน dataset `.jsonl` (เช่น `"language": "th"`)
3. **ตั้ง `allowed_languages`** ใน config JSON (เช่น `["en", "th"]`)

### ขั้นตอนที่ 2: เพิ่ม Text Normalization (แนะนำ)
1. **เพิ่ม language code** ใน `_supported_languages` ใน `text_normalization.py`
2. **สร้าง normalizer** สำหรับภาษาใหม่ (NeMo ถ้ารองรับ หรือ custom เช่น `pythainlp` สำหรับ Thai)
3. **เพิ่มใน language detector**: เพิ่ม `lingua.Language.XXX` ใน `init_lang_detector()`
4. **เพิ่ม elif branch** ใน `normalize()` method
5. **เพิ่ม dependency** ใน `pyproject.toml` ถ้าใช้ library ใหม่

> **ตัวอย่าง Thai**: ใช้ `pythainlp.util.normalize` + `num_to_thaiword` ✅ ทำแล้ว

### ขั้นตอนที่ 3: ปรับ RLHF Rewards (ถ้าใช้)
1. **WER reward**: ตรวจสอบว่า Whisper รองรับภาษาใหม่ → เพิ่มใน `_CER_LANG_LIST` ถ้าภาษาไม่มี word boundary
2. **Text normalization ใน `reward_utils.py`**: ตรวจสอบ punctuation removal, character normalization
3. **เพิ่ม `language` + `prompt_transcript`** ใน RLHF dataset output ถ้ายังไม่มี (✅ ทำแล้ว)

### ขั้นตอนที่ 4: เตรียม Audio Data
1. Audio ต้อง resample เป็น **16kHz** (ตรงกับ `CODEC_SAMPLE_RATE`)
2. Audio duration ต้อง **≤ 30 วินาที** (จาก `filter_long_duration`)
3. Transcript ต้องไม่ว่าง, ไม่ใช่แค่ spaces/punctuation
4. แนะนำให้มี `dnsmos_mos_ovr` score สำหรับ quality filtering

### ขั้นตอนที่ 5: Vectorize & Train
1. รัน `data_vectorizer.py`
2. รัน `data_merger.py`
3. ตั้ง config ให้ `allowed_languages` รวมภาษาใหม่
4. Set `enable_text_normalization: true` ถ้ามี normalizer สำหรับภาษานั้น (Thai ✅)

---

## 📊 ข้อกำหนดด้าน Audio Data สำหรับการฝึก

| Parameter | ค่า | ที่มา | หมายเหตุ |
|-----------|-----|-------|----------|
| **Encoder sample rate** | 16,000 Hz | `constants.py` CODEC_SAMPLE_RATE | Hardcoded, audio จะถูก resample อัตโนมัติ |
| **Decoder sample rate** | Configurable (16k/48k) | `model_config.json` | Codec decoder output, ตัวอย่าง 48kHz |
| **Token rate** | 50 tokens/sec | `constants.py` CODEC_TOKENS_RATE | Hardcoded, ต้องตรงกับ encoder/decoder |
| **Max audio duration** | 30 วินาที | `filtering.py` filter_long_duration | Hardcoded ≤30s |
| **Max sequence length** | 2,048 tokens | config `max_seq_len` | ≈40 sec of speech (2048/50) |
| **Min codec data sample rate** | 24,000 Hz default | `CodecTrainingConfig` | สำหรับ codec decoder training เท่านั้น |
| **Codebook size** | 65,536 | config + encoder quantizer | 4^8 levels, ต้องตรงกัน 3 จุด |
| **Hop length (encoder)** | 320 | `encoder.py` _HOP_LENGTH | Hardcoded |
| **Audio format** | WAV | `data_utils.py` load_wav() | รองรับ .wav เท่านั้น (ผ่าน torchaudio) |
| **Channels** | Mono | `data_utils.py` load_wav() | Multi-channel จะถูก mean เป็น mono |

---

## 🔗 XCodec2 Compatibility

โปรเจคนี้มีความเข้ากันได้กับ **XCodec2** checkpoint:
- `encoder.py` บรรทัด 86-111: parse `state_dict` ด้วย prefix mapping (`CodecEnc.`, `SemanticEncoder_module.`, `generator.quantizer.`, `fc_prior.`)
- `decoder.py` บรรทัด 94-119: parse `state_dict` ด้วย prefix mapping (`generator.`, `fc_post_a.`)
- Checkpoint ที่ใช้ได้: `https://huggingface.co/HKUSTAudio/xcodec2/tree/main/ckpt`
- ถ้า checkpoint ไม่มี `state_dict` key → load ตรงจาก `ckpt` (InworldTTS native format)
