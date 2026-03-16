"""
Phase 1: Whisper-Small Fine-Tuning on Uzbek Speech Corpus
=========================================================
Model:   openai/whisper-small (244M params)
Dataset: murodbek/uzbek-speech-corpus (subset)
GPU:     RTX 2070 Super (8GB VRAM)
"""

import os
import io
import json
import sys
import time
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import soundfile as sf
import datasets
import torch
import evaluate
import numpy as np
from datasets import load_dataset, load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Force soundfile decoder by preventing torchcodec from being used
os.environ["DISABLE_TORCHCODEC"] = "1"

# ============================================================
# CONFIG — Change these if needed
# ============================================================
TRAIN_SAMPLES = 1250          # 50% of 2500
EVAL_SAMPLES = 250            # 50% of 500
BASELINE_EVAL_SAMPLES = 50    # quick baseline check
BATCH_SIZE = 4                # reduce to 2 if OOM
GRAD_ACCUM = 4                # effective batch = BATCH_SIZE * GRAD_ACCUM
MAX_STEPS = 800               # fewer samples = fewer steps needed
EVAL_STEPS = 200
SAVE_STEPS = 200
LEARNING_RATE = 1e-5
MODEL_NAME = "openai/whisper-small"
SEED = 42

# Paths
DATA_DIR = "./data"
MODEL_DIR = "./models"
RESULTS_DIR = "./results"
LOG_DIR = "./logs"
CHECKPOINT_DIR = f"{MODEL_DIR}/whisper-small-uzbek-checkpoints"
FINAL_MODEL_DIR = f"{MODEL_DIR}/whisper-small-uzbek-finetuned"
DATASET_PATH = f"{DATA_DIR}/uzbek_speech_corpus"


def decode_audio(audio_entry):
    """Manually decode audio using soundfile, bypassing torchcodec."""
    if audio_entry.get("bytes") is not None:
        array, sr = sf.read(io.BytesIO(audio_entry["bytes"]))
    else:
        array, sr = sf.read(audio_entry["path"])
    return array, sr


def main():
    # ============================================================
    # SETUP — Folders + Logging
    # ============================================================
    for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR, CHECKPOINT_DIR]:
        os.makedirs(d, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOG_DIR}/training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(stream=open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False))
        ]

    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("PHASE 1: WHISPER STT FINE-TUNING — UZBEK")
    logger.info("=" * 60)
    logger.info(f"Timestamp:       {timestamp}")
    logger.info(f"Model:           {MODEL_NAME}")
    logger.info(f"Train samples:   {TRAIN_SAMPLES}")
    logger.info(f"Eval samples:    {EVAL_SAMPLES}")
    logger.info(f"Batch size:      {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM})")
    logger.info(f"Max steps:       {MAX_STEPS}")
    logger.info(f"Learning rate:   {LEARNING_RATE}")
    logger.info(f"GPU:             {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(
        f"VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A")
    logger.info(f"PyTorch:         {torch.__version__}")
    logger.info(f"CUDA available:  {torch.cuda.is_available()}")
    logger.info("")

    # ============================================================
    # STEP 1: Load Dataset
    # ============================================================
    logger.info("STEP 1: Loading dataset...")
    step1_start = time.time()

    if os.path.exists(DATASET_PATH):
        logger.info(f"Loading cached dataset from {DATASET_PATH}")
        dataset = load_from_disk(DATASET_PATH)
    else:
        logger.info("Downloading dataset from HuggingFace...")
        dataset = load_dataset("murodbek/uzbek-speech-corpus")
        dataset.save_to_disk(DATASET_PATH)
        logger.info(f"Dataset saved to {DATASET_PATH}")

    logger.info(f"Dataset loaded in {time.time() - step1_start:.1f}s")
    logger.info(f"Full dataset: {dataset}")
    logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

    # Cast audio column to decode=False to avoid torchcodec entirely
    dataset = dataset.cast_column("audio", Audio(decode=False))

    # Safe key inspection without triggering audio decoding
    sample_no_audio = {k: v for k, v in dataset['train'][0].items() if k != 'audio'}
    logger.info(f"Sample entry keys: {list(dataset['train'][0].keys())}")
    logger.info(f"Sample sentence: {sample_no_audio['sentence']}")
    logger.info("")

    # ============================================================
    # STEP 2: Create Subsets
    # ============================================================
    logger.info("STEP 2: Creating subsets...")

    train_dataset = dataset["train"].shuffle(seed=SEED).select(range(TRAIN_SAMPLES))
    eval_dataset = dataset["test"].shuffle(seed=SEED).select(range(EVAL_SAMPLES))
    baseline_eval = dataset["test"].shuffle(seed=SEED).select(range(BASELINE_EVAL_SAMPLES))

    logger.info(f"Train subset: {len(train_dataset)} samples")
    logger.info(f"Eval subset:  {len(eval_dataset)} samples")
    logger.info(f"Baseline eval: {len(baseline_eval)} samples")
    logger.info("")

    # ============================================================
    # STEP 3: Load Model + Processor
    # ============================================================
    logger.info("STEP 3: Loading Whisper model + processor...")
    step3_start = time.time()

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="uz", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.generation_config.language = "uz"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model loaded in {time.time() - step3_start:.1f}s")
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size (approx):  {total_params * 4 / 1e9:.2f} GB (fp32)")
    logger.info("")

    # ============================================================
    # STEP 4: Preprocess Audio
    # ============================================================
    logger.info("STEP 4: Preprocessing audio...")
    step4_start = time.time()

    def prepare_dataset(batch):
        array, sr = decode_audio(batch["audio"])
        batch["input_features"] = processor.feature_extractor(
            array,
            sampling_rate=sr
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    train_dataset = train_dataset.map(
        prepare_dataset,
        remove_columns=train_dataset.column_names,
        num_proc=None
    )
    eval_dataset = eval_dataset.map(
        prepare_dataset,
        remove_columns=eval_dataset.column_names,
        num_proc=None
    )

    logger.info(f"Preprocessing done in {time.time() - step4_start:.1f}s")
    logger.info(f"Train features: {len(train_dataset)}")
    logger.info(f"Eval features:  {len(eval_dataset)}")
    logger.info("")

    # ============================================================
    # STEP 5: Data Collator
    # ============================================================
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ============================================================
    # STEP 6: Metrics
    # ============================================================
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # ============================================================
    # STEP 7: Baseline WER (BEFORE fine-tuning)
    # ============================================================
    logger.info("STEP 7: Calculating BASELINE WER (before fine-tuning)...")
    step7_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    predictions_base = []
    references_base = []
    sample_transcriptions = []

    for i, sample in enumerate(baseline_eval):
        array, sr = decode_audio(sample["audio"])
        input_features = processor(
            array,
            sampling_rate=sr,
            return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        reference = sample["sentence"]
        predictions_base.append(transcription)
        references_base.append(reference)

        # Log first 10 examples for qualitative review
        if i < 10:
            sample_transcriptions.append({
                "index": i,
                "reference": reference,
                "prediction": transcription
            })
            logger.info(f"  Sample {i}: REF='{reference[:80]}...' PRED='{transcription[:80]}...'")

    baseline_wer = wer_metric.compute(predictions=predictions_base, references=references_base)

    logger.info(f"")
    logger.info(f"*** BASELINE WER: {baseline_wer:.4f} ***")
    logger.info(f"Baseline eval done in {time.time() - step7_start:.1f}s")
    logger.info("")

    # Save baseline results
    baseline_results = {
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "wer": baseline_wer,
        "num_samples": BASELINE_EVAL_SAMPLES,
        "sample_transcriptions": sample_transcriptions
    }
    with open(f"{RESULTS_DIR}/baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Baseline results saved to {RESULTS_DIR}/baseline_results.json")
    logger.info("")

    # ============================================================
    # STEP 8: Training
    # ============================================================
    logger.info("STEP 8: Starting fine-tuning...")
    logger.info(f"  Batch size:        {BATCH_SIZE}")
    logger.info(f"  Grad accumulation: {GRAD_ACCUM}")
    logger.info(f"  Effective batch:   {BATCH_SIZE * GRAD_ACCUM}")
    logger.info(f"  Max steps:         {MAX_STEPS}")
    logger.info(f"  Eval every:        {EVAL_STEPS} steps")
    logger.info(f"  Save every:        {SAVE_STEPS} steps")
    logger.info("")

    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        max_steps=MAX_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=25,
        fp16=True,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        logging_dir=f"{LOG_DIR}/tensorboard",
        push_to_hub=False,
        remove_unused_columns=False,
        save_total_limit=3,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=0,    # must be 0 on Windows to avoid spawn issues
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    train_start = time.time()
    # train_result = trainer.train(resume_from_checkpoint=True)
    train_result = trainer.train(resume_from_checkpoint="./models/whisper-small-uzbek-checkpoints/checkpoint-400")
    train_time = time.time() - train_start

    logger.info(f"")
    logger.info(f"Training completed in {train_time / 60:.1f} minutes")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    logger.info("")

    # Log training metrics
    train_metrics = train_result.metrics
    train_metrics["train_time_seconds"] = train_time
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    # ============================================================
    # STEP 9: Save Fine-Tuned Model
    # ============================================================
    logger.info("STEP 9: Saving fine-tuned model...")

    model.save_pretrained(FINAL_MODEL_DIR)
    processor.save_pretrained(FINAL_MODEL_DIR)
    logger.info(f"Model saved to {FINAL_MODEL_DIR}")
    logger.info("")

    # ============================================================
    # STEP 10: Final Evaluation (AFTER fine-tuning)
    # ============================================================
    logger.info("STEP 10: Calculating FINE-TUNED WER...")
    step10_start = time.time()

    model.eval()
    predictions_ft = []
    sample_transcriptions_ft = []

    for i, sample in enumerate(baseline_eval):
        array, sr = decode_audio(sample["audio"])
        input_features = processor(
            array,
            sampling_rate=sr,
            return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions_ft.append(transcription)

        if i < 10:
            sample_transcriptions_ft.append({
                "index": i,
                "reference": references_base[i],
                "baseline_prediction": predictions_base[i],
                "finetuned_prediction": transcription
            })
            logger.info(f"  Sample {i}:")
            logger.info(f"    REF:       '{references_base[i][:80]}'")
            logger.info(f"    BASELINE:  '{predictions_base[i][:80]}'")
            logger.info(f"    FINETUNED: '{transcription[:80]}'")

    finetuned_wer = wer_metric.compute(predictions=predictions_ft, references=references_base)
    improvement = (baseline_wer - finetuned_wer) / baseline_wer * 100

    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"  BASELINE WER:    {baseline_wer:.4f}")
    logger.info(f"  FINE-TUNED WER:  {finetuned_wer:.4f}")
    logger.info(f"  IMPROVEMENT:     {improvement:.1f}%")
    logger.info(f"  Eval time:       {time.time() - step10_start:.1f}s")
    logger.info("=" * 60)

    # ============================================================
    # STEP 11: Save All Results
    # ============================================================
    final_results = {
        "timestamp": timestamp,
        "config": {
            "model": MODEL_NAME,
            "train_samples": TRAIN_SAMPLES,
            "eval_samples": EVAL_SAMPLES,
            "batch_size": BATCH_SIZE,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "fp16": True,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "results": {
            "baseline_wer": baseline_wer,
            "finetuned_wer": finetuned_wer,
            "improvement_percent": round(improvement, 2),
            "training_loss": train_result.training_loss,
            "training_time_minutes": round(train_time / 60, 1),
        },
        "sample_comparisons": sample_transcriptions_ft
    }

    results_file = f"{RESULTS_DIR}/final_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    logger.info(f"All results saved to {results_file}")
    logger.info(f"Logs saved to {log_file}")
    logger.info(f"Model saved to {FINAL_MODEL_DIR}")
    logger.info(f"TensorBoard logs: {LOG_DIR}/tensorboard")
    logger.info("")
    logger.info("Phase 1 COMPLETE. Ready for Phase 2.")


if __name__ == '__main__':
    main()
