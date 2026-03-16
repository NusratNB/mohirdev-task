"""
Phase 2: Qwen2.5-1.5B QLoRA Fine-Tuning for Uzbek Summarization
================================================================
Model:   Qwen/Qwen2.5-1.5B + QLoRA (4-bit quantization)
Dataset: csebuetnlp/xlsum (uzbek subset — BBC Uzbek news)
GPU:     Google Colab T4 (15GB VRAM)
Task:    Uzbek article → Uzbek summary
"""

import os
import json
import time
import logging
from datetime import datetime

import torch
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import evaluate

# ============================================================
# CONFIG — Tuned for Colab T4 (15GB VRAM)
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
TRAIN_SAMPLES = 2000
EVAL_SAMPLES = 300
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 4                  # T4 can handle 4 with QLoRA
GRAD_ACCUM = 4                  # effective batch = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SEED = 42

# Paths — Google Drive for persistence
DRIVE_BASE = "/content/drive/MyDrive/uzbek_pipeline"
DATA_DIR = f"{DRIVE_BASE}/data"
MODEL_DIR = f"{DRIVE_BASE}/models"
RESULTS_DIR = f"{DRIVE_BASE}/results"
LOG_DIR = f"{DRIVE_BASE}/logs"
CHECKPOINT_DIR = f"{MODEL_DIR}/qwen-uzbek-summarizer-checkpoints"
FINAL_MODEL_DIR = f"{MODEL_DIR}/qwen-uzbek-summarizer-finetuned"
DATASET_PATH = f"{DATA_DIR}/xlsum_uzbek"

# ============================================================
# SETUP
# ============================================================

# Mount Google Drive first
from google.colab import drive
drive.mount("/content/drive")

for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{LOG_DIR}/phase2_training_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("PHASE 2: QWEN2.5-1.5B QLoRA — UZBEK SUMMARIZATION")
logger.info("=" * 60)
logger.info(f"Timestamp:       {timestamp}")
logger.info(f"Model:           {MODEL_NAME}")
logger.info(f"LoRA rank:       {LORA_R}, alpha: {LORA_ALPHA}")
logger.info(f"Train samples:   {TRAIN_SAMPLES}")
logger.info(f"Eval samples:    {EVAL_SAMPLES}")
logger.info(f"Batch size:      {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM})")
logger.info(f"Epochs:          {NUM_EPOCHS}")
logger.info(f"Learning rate:   {LEARNING_RATE}")
logger.info(f"Max input len:   {MAX_INPUT_LENGTH}")
logger.info(f"Max target len:  {MAX_TARGET_LENGTH}")
logger.info(f"GPU:             {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
logger.info(f"VRAM:            {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A")
logger.info(f"PyTorch:         {torch.__version__}")
logger.info(f"Save location:   {DRIVE_BASE} (Google Drive)")
logger.info("")

# ============================================================
# STEP 1: Load Dataset
# ============================================================
logger.info("STEP 1: Loading XL-Sum Uzbek dataset...")
step1_start = time.time()

if os.path.exists(DATASET_PATH):
    logger.info(f"Loading cached dataset from {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
else:
    logger.info("Downloading from HuggingFace...")
    dataset = load_dataset("csebuetnlp/xlsum", "uzbek")
    dataset.save_to_disk(DATASET_PATH)
    logger.info(f"Dataset saved to {DATASET_PATH}")

logger.info(f"Dataset loaded in {time.time() - step1_start:.1f}s")
logger.info(f"Full dataset: {dataset}")
logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

sample = dataset["train"][0]
logger.info(f"Sample keys: {list(sample.keys())}")
logger.info(f"Sample title: {sample.get('title', 'N/A')[:100]}")
logger.info(f"Sample text (first 200 chars): {sample['text'][:200]}")
logger.info(f"Sample summary: {sample['summary'][:200]}")
logger.info("")

# ============================================================
# STEP 2: Create Subsets
# ============================================================
logger.info("STEP 2: Creating subsets...")

train_dataset = dataset["train"].shuffle(seed=SEED).select(range(TRAIN_SAMPLES))
eval_dataset = dataset["validation"].shuffle(seed=SEED).select(range(min(EVAL_SAMPLES, len(dataset["validation"]))))

train_text_lengths = [len(x["text"].split()) for x in train_dataset]
train_summary_lengths = [len(x["summary"].split()) for x in train_dataset]

logger.info(f"Train subset: {len(train_dataset)} samples")
logger.info(f"Eval subset:  {len(eval_dataset)} samples")
logger.info(f"Avg article length:  {np.mean(train_text_lengths):.0f} words")
logger.info(f"Avg summary length:  {np.mean(train_summary_lengths):.0f} words")
logger.info(f"Max article length:  {max(train_text_lengths)} words")
logger.info(f"Max summary length:  {max(train_summary_lengths)} words")
logger.info("")

# ============================================================
# STEP 3: Load Tokenizer
# ============================================================
logger.info("STEP 3: Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Set pad_token = eos_token: '{tokenizer.eos_token}'")

logger.info(f"Vocab size: {tokenizer.vocab_size}")
logger.info("")

# ============================================================
# STEP 4: Load Model with 4-bit Quantization (QLoRA)
# ============================================================
logger.info("STEP 4: Loading model with 4-bit quantization...")
step4_start = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)

total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Model loaded in {time.time() - step4_start:.1f}s")
logger.info(f"Total parameters: {total_params:,}")
logger.info(f"Quantized to 4-bit — approx {total_params * 0.5 / 1e9:.2f} GB VRAM for weights")
logger.info("")

# ============================================================
# STEP 5: Apply LoRA
# ============================================================
logger.info("STEP 5: Applying LoRA adapters...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
trainable_pct = 100 * trainable_params / all_params

logger.info(f"LoRA rank:           {LORA_R}")
logger.info(f"LoRA alpha:          {LORA_ALPHA}")
logger.info(f"Target modules:      q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj")
logger.info(f"Total parameters:    {all_params:,}")
logger.info(f"Trainable (LoRA):    {trainable_params:,} ({trainable_pct:.2f}%)")
logger.info("")

# ============================================================
# STEP 6: Preprocess — Format as Instruction
# ============================================================
logger.info("STEP 6: Preprocessing dataset...")
step6_start = time.time()

# Uzbek prompt: "Briefly summarize the following text"
PROMPT_TEMPLATE = """Quyidagi matnni qisqacha umumlashtiring.

### Matn:
{text}

### Xulosa:
{summary}"""

def format_and_tokenize(example):
    full_prompt = PROMPT_TEMPLATE.format(
        text=example["text"][:1500],
        summary=example["summary"]
    )

    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        max_length=MAX_INPUT_LENGTH + MAX_TARGET_LENGTH,
        padding=False,
    )

    # Find where summary starts — mask everything before it
    input_prompt = PROMPT_TEMPLATE.split("{summary}")[0].format(
        text=example["text"][:1500]
    )
    input_ids_prompt_only = tokenizer(
        input_prompt,
        truncation=True,
        max_length=MAX_INPUT_LENGTH + MAX_TARGET_LENGTH,
    )["input_ids"]

    # Labels: -100 for input tokens (no loss), actual ids for summary tokens
    labels = tokenized["input_ids"].copy()
    prompt_len = len(input_ids_prompt_only)
    labels[:prompt_len] = [-100] * prompt_len

    tokenized["labels"] = labels
    return tokenized

train_tokenized = train_dataset.map(
    format_and_tokenize,
    remove_columns=train_dataset.column_names,
    num_proc=1,
)
eval_tokenized = eval_dataset.map(
    format_and_tokenize,
    remove_columns=eval_dataset.column_names,
    num_proc=1,
)

logger.info(f"Preprocessing done in {time.time() - step6_start:.1f}s")
logger.info(f"Sample tokenized length: {len(train_tokenized[0]['input_ids'])} tokens")
logger.info(f"Prompt template (Uzbek): 'Quyidagi matnni qisqacha umumlashtiring'")
logger.info("")

decoded_example = tokenizer.decode(train_tokenized[0]["input_ids"], skip_special_tokens=True)
logger.info(f"Decoded training example (first 500 chars):")
logger.info(f"{decoded_example[:500]}")
logger.info("")

# ============================================================
# STEP 7: Baseline — Generate BEFORE Fine-Tuning
# ============================================================
logger.info("STEP 7: Generating baseline summaries (BEFORE fine-tuning)...")
step7_start = time.time()

baseline_samples = dataset["test"].shuffle(seed=SEED).select(range(10))
baseline_results = []

model.eval()
for i, sample in enumerate(baseline_samples):
    input_prompt = PROMPT_TEMPLATE.split("{summary}")[0].format(
        text=sample["text"][:1500]
    )

    inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True,
                        max_length=MAX_INPUT_LENGTH).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            do_sample=False,
            temperature=1.0,
            num_beams=1,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    baseline_results.append({
        "index": i,
        "reference": sample["summary"][:300],
        "baseline_prediction": generated_text[:300],
    })

    logger.info(f"  Sample {i}:")
    logger.info(f"    REF:      '{sample['summary'][:100]}'")
    logger.info(f"    BASELINE: '{generated_text[:100]}'")

logger.info(f"Baseline generation done in {time.time() - step7_start:.1f}s")
logger.info("")

with open(f"{RESULTS_DIR}/phase2_baseline.json", "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, ensure_ascii=False, indent=2)
logger.info(f"Baseline saved to {RESULTS_DIR}/phase2_baseline.json")
logger.info("")

# ============================================================
# STEP 8: Training
# ============================================================
logger.info("STEP 8: Starting QLoRA fine-tuning...")

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=3,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to=["tensorboard"],
    logging_dir=f"{LOG_DIR}/tensorboard_phase2",
    push_to_hub=False,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_checkpointing=True,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

train_start = time.time()
train_result = trainer.train()
train_time = time.time() - train_start

logger.info(f"")
logger.info(f"Training completed in {train_time / 60:.1f} minutes")
logger.info(f"Training loss: {train_result.training_loss:.4f}")
logger.info("")

train_metrics = train_result.metrics
train_metrics["train_time_seconds"] = train_time
trainer.log_metrics("train", train_metrics)
trainer.save_metrics("train", train_metrics)

# ============================================================
# STEP 9: Save Fine-Tuned Model (LoRA adapters)
# ============================================================
logger.info("STEP 9: Saving LoRA adapters to Google Drive...")

model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

adapter_size = sum(
    os.path.getsize(os.path.join(FINAL_MODEL_DIR, f))
    for f in os.listdir(FINAL_MODEL_DIR)
    if os.path.isfile(os.path.join(FINAL_MODEL_DIR, f))
) / 1e6

logger.info(f"LoRA adapters saved to {FINAL_MODEL_DIR}")
logger.info(f"Adapter size: {adapter_size:.1f} MB (vs full model ~3GB)")
logger.info("")

# ============================================================
# STEP 10: Final Evaluation (AFTER fine-tuning)
# ============================================================
logger.info("STEP 10: Generating summaries AFTER fine-tuning...")
step10_start = time.time()

model.eval()
finetuned_results = []

for i, sample in enumerate(baseline_samples):
    input_prompt = PROMPT_TEMPLATE.split("{summary}")[0].format(
        text=sample["text"][:1500]
    )

    inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True,
                        max_length=MAX_INPUT_LENGTH).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            do_sample=False,
            temperature=1.0,
            num_beams=1,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    finetuned_results.append({
        "index": i,
        "reference": sample["summary"][:300],
        "baseline_prediction": baseline_results[i]["baseline_prediction"],
        "finetuned_prediction": generated_text[:300],
    })

    logger.info(f"  Sample {i}:")
    logger.info(f"    REF:       '{sample['summary'][:100]}'")
    logger.info(f"    BASELINE:  '{baseline_results[i]['baseline_prediction'][:100]}'")
    logger.info(f"    FINETUNED: '{generated_text[:100]}'")

logger.info(f"Fine-tuned generation done in {time.time() - step10_start:.1f}s")
logger.info("")

# ============================================================
# STEP 11: Compute ROUGE Scores
# ============================================================
logger.info("STEP 11: Computing ROUGE scores...")

rouge_metric = evaluate.load("rouge")

baseline_preds = [r["baseline_prediction"] for r in finetuned_results]
finetuned_preds = [r["finetuned_prediction"] for r in finetuned_results]
refs = [r["reference"] for r in finetuned_results]

baseline_rouge = rouge_metric.compute(predictions=baseline_preds, references=refs)
finetuned_rouge = rouge_metric.compute(predictions=finetuned_preds, references=refs)

logger.info("")
logger.info("=" * 60)
logger.info("FINAL RESULTS — ROUGE SCORES")
logger.info("=" * 60)
logger.info(f"                BASELINE    FINE-TUNED   IMPROVEMENT")
logger.info(f"  ROUGE-1:      {baseline_rouge['rouge1']:.4f}      {finetuned_rouge['rouge1']:.4f}       {finetuned_rouge['rouge1'] - baseline_rouge['rouge1']:+.4f}")
logger.info(f"  ROUGE-2:      {baseline_rouge['rouge2']:.4f}      {finetuned_rouge['rouge2']:.4f}       {finetuned_rouge['rouge2'] - baseline_rouge['rouge2']:+.4f}")
logger.info(f"  ROUGE-L:      {baseline_rouge['rougeL']:.4f}      {finetuned_rouge['rougeL']:.4f}       {finetuned_rouge['rougeL'] - baseline_rouge['rougeL']:+.4f}")
logger.info("=" * 60)

# ============================================================
# STEP 12: Save Everything to Google Drive
# ============================================================
final_results = {
    "timestamp": timestamp,
    "config": {
        "model": MODEL_NAME,
        "quantization": "4-bit NF4 (QLoRA)",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "train_samples": TRAIN_SAMPLES,
        "eval_samples": EVAL_SAMPLES,
        "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    },
    "results": {
        "training_loss": train_result.training_loss,
        "training_time_minutes": round(train_time / 60, 1),
        "baseline_rouge": {k: round(v, 4) for k, v in baseline_rouge.items()},
        "finetuned_rouge": {k: round(v, 4) for k, v in finetuned_rouge.items()},
        "adapter_size_mb": round(adapter_size, 1),
        "trainable_params": trainable_params,
        "trainable_percent": round(trainable_pct, 2),
    },
    "sample_comparisons": finetuned_results,
}

results_file = f"{RESULTS_DIR}/phase2_final_results.json"
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)

logger.info(f"")
logger.info(f"All results saved to {results_file}")
logger.info(f"LoRA adapters saved to {FINAL_MODEL_DIR}")
logger.info(f"Logs saved to {log_file}")
logger.info(f"TensorBoard: {LOG_DIR}/tensorboard_phase2")
logger.info("")
logger.info("Phase 2 COMPLETE. Ready for Phase 3 (Pipeline).")
logger.info("All files are on Google Drive — safe from Colab disconnects!")