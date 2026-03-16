import os
import json
import time
import logging
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import evaluate


# --- paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "qwen-uz-summarizer-ckpts")
FINAL_MODEL_DIR = os.path.join(MODEL_DIR, "qwen-uz-summarizer-final")
DATASET_CACHE = os.path.join(DATA_DIR, "xlsum_uzbek")

for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# --- logging setup ---
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"phase2_{ts}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# --- hyperparams ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
N_TRAIN = 5000              
N_EVAL = 500                
N_BASELINE = 10
MAX_SRC_LEN = 768           
MAX_TGT_LEN = 192           
BS = 4 
GRAD_ACC = 4
EPOCHS = 8
LR = 1e-4
LORA_R = 16
LORA_ALPHA = 32


def gpu_stats():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{used:.1f}/{total:.1f} GB"
    return "no gpu"


# ===========================================================
log.info(f"Starting phase 2 — Uzbek summarization fine-tuning")
log.info(f"Model: {MODEL_NAME} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
log.info(f"Train: {N_TRAIN} samples | BS: {BS}x{GRAD_ACC}={BS*GRAD_ACC} | Epochs: {EPOCHS} | LR: {LR}")
log.info(f"LoRA r={LORA_R} alpha={LORA_ALPHA}")
log.info(f"VRAM: {gpu_stats()}")


# --- load dataset ---
log.info("Loading XL-Sum uzbek dataset...")
t0 = time.time()

if os.path.exists(DATASET_CACHE):
    ds = load_from_disk(DATASET_CACHE)
    log.info(f"Loaded from cache ({time.time()-t0:.1f}s)")
else:
    # the csebuetnlp/xlsum loader is broken on newer datasets lib,
    # so we grab the parquet files directly from HF hub
    try:
        ds = load_dataset("csebuetnlp/xlsum", "uzbek")
    except (RuntimeError, ValueError) as e:
        log.info(f"Standard load failed ({e.__class__.__name__}), loading parquets directly...")
        base = "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/refs%2Fconvert%2Fparquet/uzbek"
        ds = DatasetDict({
            "train": Dataset.from_pandas(pd.read_parquet(f"{base}/train/0000.parquet")),
            "validation": Dataset.from_pandas(pd.read_parquet(f"{base}/validation/0000.parquet")),
            "test": Dataset.from_pandas(pd.read_parquet(f"{base}/test/0000.parquet")),
        })
    ds.save_to_disk(DATASET_CACHE)
    log.info(f"Downloaded and cached ({time.time()-t0:.1f}s)")

log.info(f"Dataset sizes — train: {len(ds['train'])} | val: {len(ds['validation'])} | test: {len(ds['test'])}")

# quick look at the data
for i in range(2):
    s = ds["train"][i]
    log.info(f"  example {i}: text={len(s['text'].split())}w, summary={len(s['summary'].split())}w")
    log.info(f"    text: {s['text'][:150]}...")
    log.info(f"    summary: {s['summary'][:150]}")


# Uzbek Cyrillic → Latin transliteration
# Official mapping since Uzbekistan switched to Latin in 1993
CYRILLIC_TO_LATIN = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
    'е': 'e', 'ё': 'yo', 'ж': 'j', 'з': 'z', 'и': 'i',
    'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
    'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
    'у': 'u', 'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'ch',
    'ш': 'sh', 'щ': 'sh', 'ъ': "'", 'ы': 'i', 'ь': '',
    'э': 'e', 'ю': 'yu', 'я': 'ya',
    'ў': "o'", 'қ': 'q', 'ғ': "g'", 'ҳ': 'h',
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D',
    'Е': 'E', 'Ё': 'Yo', 'Ж': 'J', 'З': 'Z', 'И': 'I',
    'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N',
    'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T',
    'У': 'U', 'Ф': 'F', 'Х': 'X', 'Ц': 'Ts', 'Ч': 'Ch',
    'Ш': 'Sh', 'Щ': 'Sh', 'Ъ': "'", 'Ы': 'I', 'Ь': '',
    'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
    'Ў': "O'", 'Қ': 'Q', 'Ғ': "G'", 'Ҳ': 'H',
}

def cyr_to_lat(text):
    result = []
    for ch in text:
        result.append(CYRILLIC_TO_LATIN.get(ch, ch))
    return ''.join(result)

# convert the dataset to latin script
def convert_to_latin(example):
    example["text"] = cyr_to_lat(example["text"])
    example["summary"] = cyr_to_lat(example["summary"])
    if "title" in example:
        example["title"] = cyr_to_lat(example["title"])
    return example

log.info("Converting dataset from Cyrillic to Latin script...")
ds = ds.map(convert_to_latin, num_proc=4, desc="cyr→lat")

# verify
sample = ds["train"][0]
log.info(f"  Converted sample: {sample['text'][:150]}")
log.info(f"  Summary: {sample['summary'][:150]}")

# --- subsets ---
train_ds = ds["train"].shuffle(seed=42).select(range(min(N_TRAIN, len(ds["train"]))))
eval_ds = ds["validation"].shuffle(seed=42).select(range(min(N_EVAL, len(ds["validation"]))))
test_samples = ds["test"].shuffle(seed=42).select(range(min(N_BASELINE, len(ds["test"]))))

steps_per_epoch = N_TRAIN // (BS * GRAD_ACC)
total_steps = steps_per_epoch * EPOCHS
log.info(f"Subsets ready — train:{len(train_ds)} eval:{len(eval_ds)} test:{len(test_samples)}")
log.info(f"Estimated {steps_per_epoch} steps/epoch, {total_steps} total")


# --- tokenizer ---
log.info("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# --- model + quantization ---
log.info("Loading model with 4-bit quantization...")
t0 = time.time()

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_cfg,
    device_map="auto", trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
log.info(f"Model loaded ({time.time()-t0:.1f}s) | params: {sum(p.numel() for p in model.parameters()):,}")
log.info(f"VRAM after load: {gpu_stats()}")


# --- lora ---
log.info("Applying LoRA adapters...")
lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    bias="none", task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_cfg)

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
log.info(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")


# --- preprocessing ---
# prompt is in uzbek: "Briefly summarize the following text"
PROMPT = """Quyidagi matnni qisqacha umumlashtiring.

### Matn:
{text}

### Xulosa:
{summary}"""

# same thing but without the summary (used for inference)
PROMPT_NOSUMMARY = """Quyidagi matnni qisqacha umumlashtiring.

### Matn:
{text}

### Xulosa:
"""

def tokenize_example(ex):
    """tokenize a single example, masking the prompt tokens in labels"""
    full = PROMPT.format(text=ex["text"][:2500], summary=ex["summary"])
    toks = tok(full, truncation=True, max_length=MAX_SRC_LEN + MAX_TGT_LEN, padding=False)

    # figure out where the summary starts so we only compute loss on it
    prompt_part = PROMPT_NOSUMMARY.format(text=ex["text"][:2500])
    prompt_ids = tok(prompt_part, truncation=True, max_length=MAX_SRC_LEN + MAX_TGT_LEN)["input_ids"]

    labels = toks["input_ids"].copy()
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
    toks["labels"] = labels
    return toks


log.info("Tokenizing datasets...")
t0 = time.time()
train_tok = train_ds.map(tokenize_example, remove_columns=train_ds.column_names, num_proc=4, desc="train")
eval_tok = eval_ds.map(tokenize_example, remove_columns=eval_ds.column_names, num_proc=4, desc="eval")

# sanity check — how many examples lost their summary to truncation?
empty = sum(1 for ex in train_tok if all(l == -100 for l in ex["labels"]))
avg_len = np.mean([len(ex["input_ids"]) for ex in train_tok])
log.info(f"Tokenized in {time.time()-t0:.1f}s | avg seq len: {avg_len:.0f} | empty labels: {empty}/{len(train_tok)}")

# decode one example to make sure it looks right
sample_decoded = tok.decode(train_tok[0]["input_ids"], skip_special_tokens=True)
log.info(f"Sample (first 300 chars): {sample_decoded[:300]}")


# --- baseline eval (before training) ---
log.info("Running baseline inference (before fine-tuning)...")
model.eval()
baseline = []
rouge = evaluate.load("rouge")

for i, s in enumerate(test_samples):
    prompt = PROMPT_NOSUMMARY.format(text=s["text"][:2500])
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_SRC_LEN).to(model.device)

    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_TGT_LEN, do_sample=False)

    pred = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    baseline.append({"ref": s["summary"], "pred": pred})
    log.info(f"  [{i}] ref: {s['summary'][:100]}")
    log.info(f"       pred: {pred[:100]}")

baseline_rouge = rouge.compute(
    predictions=[b["pred"] for b in baseline],
    references=[b["ref"] for b in baseline]
)
log.info(f"Baseline ROUGE — R1: {baseline_rouge['rouge1']:.4f} | R2: {baseline_rouge['rouge2']:.4f} | RL: {baseline_rouge['rougeL']:.4f}")

with open(os.path.join(RESULTS_DIR, "phase2_baseline.json"), "w", encoding="utf-8") as f:
    json.dump({"rouge": baseline_rouge, "samples": baseline}, f, ensure_ascii=False, indent=2)


# --- custom callback for nicer logs ---
class ProgressLog(TrainerCallback):
    def __init__(self):
        self.times = []
        self.t = None

    def on_step_begin(self, args, state, control, **kw):
        self.t = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        if self.t:
            self.times.append(time.time() - self.t)

        step = state.global_step
        loss = logs.get("loss", logs.get("eval_loss"))
        lr = logs.get("learning_rate")

        msg = f"step {step}/{total_steps}"
        if loss is not None:
            msg += f" | loss: {loss:.4f}"
        if lr is not None:
            msg += f" | lr: {lr:.2e}"
        msg += f" | gpu: {gpu_stats()}"

        if len(self.times) > 3:
            eta = np.mean(self.times[-10:]) * (total_steps - step) / 60
            msg += f" | eta: {eta:.0f}m"

        log.info(msg)

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if metrics:
            log.info(f"  eval @ step {state.global_step} — loss: {metrics.get('eval_loss', '?')}")


# --- training ---
log.info("Starting training...")
log.info(f"VRAM before training: {gpu_stats()}")

args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BS,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=150,
    save_steps=150,
    save_total_limit=3,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to=["tensorboard"],
    logging_dir=os.path.join(LOG_DIR, "tb_phase2"),
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_checkpointing=True,
)

collator = DataCollatorForSeq2Seq(tokenizer=tok, padding=True, pad_to_multiple_of=8)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_tok, eval_dataset=eval_tok,
    data_collator=collator, processing_class=tok,
    callbacks=[ProgressLog()],
)

t0 = time.time()
result = trainer.train()
train_time = time.time() - t0

log.info(f"Training done — {train_time/60:.1f} min, final loss: {result.training_loss:.4f}")

metrics = result.metrics
metrics["train_time_seconds"] = train_time
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)


# --- save lora adapters ---
log.info(f"Saving model to {FINAL_MODEL_DIR}")
model.save_pretrained(FINAL_MODEL_DIR)
tok.save_pretrained(FINAL_MODEL_DIR)

adapter_mb = sum(
    os.path.getsize(os.path.join(FINAL_MODEL_DIR, f)) / 1e6
    for f in os.listdir(FINAL_MODEL_DIR)
    if os.path.isfile(os.path.join(FINAL_MODEL_DIR, f))
)
log.info(f"Adapter size: {adapter_mb:.1f} MB (vs ~3GB for full model)")


# --- final eval (after training) ---
log.info("Running fine-tuned inference...")
model.eval()
ft_preds = []

for i, s in enumerate(test_samples):
    prompt = PROMPT_NOSUMMARY.format(text=s["text"][:2500])
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_SRC_LEN).to(model.device)

    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_TGT_LEN, do_sample=False)

    pred = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    ft_preds.append(pred)

    log.info(f"  [{i}] ref:      {baseline[i]['ref'][:100]}")
    log.info(f"       baseline: {baseline[i]['pred'][:100]}")
    log.info(f"       tuned:    {pred[:100]}")

ft_rouge = rouge.compute(predictions=ft_preds, references=[b["ref"] for b in baseline])


# --- results ---
log.info("")
log.info("=" * 50)
log.info("RESULTS")
log.info("=" * 50)
log.info(f"            Baseline → Fine-tuned")
log.info(f"  ROUGE-1:  {baseline_rouge['rouge1']:.4f}  →  {ft_rouge['rouge1']:.4f}  ({ft_rouge['rouge1']-baseline_rouge['rouge1']:+.4f})")
log.info(f"  ROUGE-2:  {baseline_rouge['rouge2']:.4f}  →  {ft_rouge['rouge2']:.4f}  ({ft_rouge['rouge2']-baseline_rouge['rouge2']:+.4f})")
log.info(f"  ROUGE-L:  {baseline_rouge['rougeL']:.4f}  →  {ft_rouge['rougeL']:.4f}  ({ft_rouge['rougeL']-baseline_rouge['rougeL']:+.4f})")
log.info(f"  Training:  {train_time/60:.1f} min | Loss: {result.training_loss:.4f}")
log.info(f"  Adapter:   {adapter_mb:.1f} MB")
log.info("=" * 50)

# save everything
comparisons = []
for i in range(len(baseline)):
    comparisons.append({
        "reference": baseline[i]["ref"],
        "baseline": baseline[i]["pred"],
        "finetuned": ft_preds[i],
    })

with open(os.path.join(RESULTS_DIR, "phase2_results.json"), "w", encoding="utf-8") as f:
    json.dump({
        "model": MODEL_NAME,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA},
        "training": {
            "samples": N_TRAIN, "epochs": EPOCHS,
            "batch_size": BS * GRAD_ACC,
            "lr": LR, "time_min": round(train_time/60, 1),
            "final_loss": round(result.training_loss, 4),
        },
        "baseline_rouge": {k: round(v, 4) for k, v in baseline_rouge.items()},
        "finetuned_rouge": {k: round(v, 4) for k, v in ft_rouge.items()},
        "adapter_size_mb": round(adapter_mb, 1),
        "comparisons": comparisons,
    }, f, ensure_ascii=False, indent=2)

log.info(f"Results saved to {RESULTS_DIR}/phase2_results.json")
log.info("Done! Ready for pipeline integration.")