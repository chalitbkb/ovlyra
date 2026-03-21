#!/usr/bin/env python3
"""Simple inference script for TTS model.

This script demonstrates how to load a trained TTS model checkpoint and generate
speech from text using an audio prompt.

Example usage:
    python tools/serving/inference.py \
        --model_checkpoint_path /path/to/your/trained_model \
        --audio_encoder_path /path/to/encoder.pt \
        --audio_decoder_path /path/to/decoder.pt \
        --prompt_wav_path /path/to/your_prompt.wav \
        --prompt_transcription "This is what the speaker says in the prompt." \
        --text "Hello, this is a test of the text-to-speech system." \
        --output_path ./audios/output.wav
"""

import os
import time
import torch
import torchaudio
import transformers
from absl import app, flags, logging

from tts.core import constants, modeling, prompting
from tts.core.codec import decoding, encoding
from tts.data import data_utils
from tts.inference import inferencing

FLAGS = flags.FLAGS

# Required flags
_MODEL_CHECKPOINT_PATH = flags.DEFINE_string(
    "model_checkpoint_path", None,
    "Path to trained model checkpoint directory (transformer format)")
_AUDIO_ENCODER_PATH = flags.DEFINE_string(
    "audio_encoder_path", None,
    "Path to audio encoder checkpoint")
_AUDIO_DECODER_PATH = flags.DEFINE_string(
    "audio_decoder_path", None,
    "Path to audio decoder checkpoint (must be in same directory as model_config.json)")
_PROMPT_WAV_PATH = flags.DEFINE_string(
    "prompt_wav_path", None,
    "Path to audio prompt (.wav file)")
_PROMPT_TRANSCRIPTION = flags.DEFINE_string(
    "prompt_transcription", None,
    "Transcription of the audio prompt")
_TEXT = flags.DEFINE_string(
    "text", None,
    "Text to synthesize")

# Optional flags
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "./audios/output.wav",
    "Output audio file path")
_USE_VLLM = flags.DEFINE_bool(
    "use_vllm", False,
    "Whether to use vLLM for inference")
_SEED = flags.DEFINE_integer(
    "seed", 42,
    "Random seed for inference")
_TEMPERATURE = flags.DEFINE_float(
    "temperature", 0.8,
    "Sampling temperature")
_MAX_TOKENS = flags.DEFINE_integer(
    "max_tokens", 1500,
    "Maximum generated speech tokens")
_MIN_TOKENS = flags.DEFINE_integer(
    "min_tokens", 10,
    "Minimum generated speech tokens")
_TOP_P = flags.DEFINE_float(
    "top_p", 0.95,
    "Nucleus sampling top-p")
_TOP_K = flags.DEFINE_integer(
    "top_k", 50,
    "Top-k sampling")
_REPETITION_PENALTY = flags.DEFINE_float(
    "repetition_penalty", 1.0,
    "Repetition penalty")
_FREQUENCY_PENALTY = flags.DEFINE_float(
    "frequency_penalty", 0.0,
    "Frequency penalty")


def main(argv: list[str]) -> None:
    del argv  # Unused.

    # Get flag values
    model_checkpoint_path = _MODEL_CHECKPOINT_PATH.value
    audio_encoder_path = _AUDIO_ENCODER_PATH.value
    audio_decoder_path = _AUDIO_DECODER_PATH.value
    prompt_wav_path = _PROMPT_WAV_PATH.value
    prompt_transcription = _PROMPT_TRANSCRIPTION.value
    text_to_synthesize = _TEXT.value
    output_path = _OUTPUT_PATH.value
    use_vllm = _USE_VLLM.value
    seed = _SEED.value
    temperature = _TEMPERATURE.value
    max_tokens = _MAX_TOKENS.value
    min_tokens = _MIN_TOKENS.value
    top_p = _TOP_P.value
    top_k = _TOP_K.value
    repetition_penalty = _REPETITION_PENALTY.value
    frequency_penalty = _FREQUENCY_PENALTY.value
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model components
    logging.info("Loading model...")
    if use_vllm:
        import vllm
        start_time = time.time()
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint_path)
        logging.info("Tokenizer with size: %d loaded in %.2f seconds.",
                     len(tokenizer), time.time() - start_time)

        model = vllm.LLM(
            model=model_checkpoint_path,
            seed=seed,
            gpu_memory_utilization=0.7,
            max_model_len=3072
        )
    else:
        start_time = time.time()
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint_path)
        logging.info("Tokenizer with size: %d loaded in %.2f seconds.",
                     len(tokenizer), time.time() - start_time)

        start_time = time.time()
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_checkpoint_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        logging.info("SpeechLM loaded with %.2fM parameters in %.2f seconds.",
                     num_params, time.time() - start_time)

    # Initialize audio codec components
    logging.info("Loading audio encoder...")
    audio_encoder = encoding.CachingAudioEncoder(
        model_path=audio_encoder_path, device=device)

    logging.info("Loading audio decoder...")
    audio_decoder = decoding.create(
        model_path=audio_decoder_path, device=device)

    # Create TTS model
    prompt_compiler = prompting.InferencePromptCompiler()
    tts_model = inferencing.LocalTtsModel(
        model=model,
        device=device,
        tokenizer=tokenizer,
        audio_encoder=audio_encoder,
        audio_decoder=audio_decoder,
        prompt_compiler=prompt_compiler,
        use_vllm=use_vllm
    )

    # Load audio prompt
    logging.info(f"Loading audio prompt: {prompt_wav_path}")
    prompt_wav, _ = data_utils.load_wav(
        prompt_wav_path, target_sample_rate=constants.CODEC_SAMPLE_RATE)

    # Inference settings
    inference_settings = inferencing.InferenceSettings(
        temperature=temperature,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        seed=seed,
    )

    audio_prompt_transcription = prompt_transcription

    # Generate speech
    logging.info(f"Generating speech for text: '{text_to_synthesize}'")
    inference_result = tts_model.synthesize_speech(
        inference_settings=inference_settings,
        text_to_synthesize=text_to_synthesize,
        prompt_id="simple_inference",
        prompt_wav=prompt_wav,
        audio_prompt_transcription=audio_prompt_transcription,
        voice_description="",
        enable_instruction=True
    )

    # Save output audio
    output_wav = inference_result.wav

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, output_wav, audio_decoder.sample_rate)

    # Print results
    logging.info(f"Generated audio saved to: {output_path}")
    logging.info(f"Encoding time: {inference_result.encoding_time:.2f}s")
    logging.info(f"Decoding time: {inference_result.decoding_time:.2f}s")


if __name__ == "__main__":
    flags.mark_flags_as_required([
        "model_checkpoint_path",
        "audio_encoder_path",
        "audio_decoder_path",
        "prompt_wav_path",
        "prompt_transcription",
        "text"
    ])
    app.run(main)
