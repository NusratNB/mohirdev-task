"""
End-to-end pipeline: Uzbek audio → transcription → summary
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime

import torch
import librosa
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# --- paths (update if your folders are different) ---
STT_MODEL_PATH = "./models/whisper-small-uzbek-checkpoints/checkpoint-800"
LLM_MODEL_PATH = "./models/qwen-uz-summarizer-ckpts/checkpoint-750"
LLM_BASE_MODEL = "Qwen/Qwen2.5-1.5B"
RESULTS_DIR = "./results"

os.makedirs(RESULTS_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(f"{RESULTS_DIR}/pipeline_{ts}.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def load_audio(path):
    log.info(f"Loading audio: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")

    audio, sr = librosa.load(path, sr=16000)
    duration = len(audio) / sr
    log.info(f"  Duration: {duration:.1f}s | Samples: {len(audio):,} | SR: {sr}")
    return audio, sr, duration


def transcribe(audio_path):
    """load whisper, transcribe, free from gpu"""
    audio, sr, duration = load_audio(audio_path)

    log.info(f"Loading Whisper from {STT_MODEL_PATH}...")
    processor = WhisperProcessor.from_pretrained(STT_MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(STT_MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    model.generation_config.language = "uz"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    log.info("Transcribing...")
    t0 = time.time()

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(inputs, max_length=448)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    elapsed = time.time() - t0

    log.info(f"  Transcription ({elapsed:.1f}s): {text[:200]}")

    # free gpu before loading llm
    del model, processor
    torch.cuda.empty_cache()
    log.info("  Whisper freed from GPU")

    return text, duration, elapsed


def summarize(text):
    """load qwen + lora, summarize, return result"""
    log.info(f"Loading Qwen from {LLM_BASE_MODEL} + adapter {LLM_MODEL_PATH}...")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        LLM_BASE_MODEL, quantization_config=bnb_cfg,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, LLM_MODEL_PATH)
    model.eval()

    log.info(f"  Model loaded ({time.time()-t0:.1f}s)")

    # same prompt used during training
    prompt = f"""Quyidagi matnni qisqacha umumlashtiring.

### Matn:
{text}

### Xulosa:
"""

    log.info("Summarizing...")
    t0 = time.time()

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            num_beams=1,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    summary = tok.decode(gen_ids, skip_special_tokens=True).strip()
    elapsed = time.time() - t0

    log.info(f"  Summary ({elapsed:.1f}s): {summary[:200]}")

    return summary, elapsed


def main(audio_path):
    start = time.time()

    log.info("=" * 50)
    log.info("UZBEK SPEECH-TO-SUMMARY PIPELINE")
    log.info("=" * 50)
    log.info(f"  Audio:  {audio_path}")
    log.info(f"  STT:    {STT_MODEL_PATH}")
    log.info(f"  LLM:    {LLM_BASE_MODEL} + {LLM_MODEL_PATH}")
    log.info(f"  GPU:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    log.info("")

    # step 1: transcribe
    transcript, audio_dur, trans_time = transcribe(audio_path)

    # step 2: summarize
    summary, summ_time = summarize(transcript)

    total = time.time() - start

    # final output
    log.info("")
    log.info("=" * 50)
    log.info("RESULTS")
    log.info("=" * 50)
    log.info(f"  Original Transcript: {transcript}")
    log.info(f"  Summary: {summary}")
    log.info(f"")
    log.info(f"  Audio duration:      {audio_dur:.1f}s")
    log.info(f"  Transcription time:  {trans_time:.1f}s")
    log.info(f"  Summarization time:  {summ_time:.1f}s")
    log.info(f"  Total pipeline:      {total:.1f}s")
    log.info("=" * 50)

    # save
    result = {
        "audio_file": audio_path,
        "transcript": transcript,
        "summary": summary,
        "audio_duration_s": round(audio_dur, 1),
        "transcription_time_s": round(trans_time, 1),
        "summarization_time_s": round(summ_time, 1),
        "total_time_s": round(total, 1),
        "models": {
            "stt": STT_MODEL_PATH,
            "llm_base": LLM_BASE_MODEL,
            "llm_adapter": LLM_MODEL_PATH,
        }
    }

    out_path = f"{RESULTS_DIR}/pipeline_output_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log.info(f"  Saved to {out_path}")

    # clean stdout output
    print("\n" + "=" * 50)
    print(f"Original Transcript: {transcript}")
    print("=" * 50)
    print(f"Summary: {summary}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uzbek Speech-to-Summary Pipeline")
    parser.add_argument("audio_file", help="Path to .wav or .mp3 file")
    parser.add_argument("--stt-model", default=STT_MODEL_PATH)
    parser.add_argument("--llm-base", default=LLM_BASE_MODEL)
    parser.add_argument("--llm-adapter", default=LLM_MODEL_PATH)
    args = parser.parse_args()

    STT_MODEL_PATH = args.stt_model
    LLM_MODEL_PATH = args.llm_adapter
    LLM_BASE_MODEL = args.llm_base

    main(args.audio_file)