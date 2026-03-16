## Disclaimer: This task was completed with the assistance of AI.

# Uzbek Speech-to-Summary Pipeline

Takes Uzbek audio, transcribes it, and generates a summary — all in Uzbek.

```
Audio (.wav/.mp3) → Whisper (fine-tuned) → Uzbek text → Qwen2.5 (QLoRA) → Uzbek summary
```

## Models & Datasets

**STT:** `openai/whisper-small` fine-tuned on [`murodbek/uzbek-speech-corpus`](https://huggingface.co/datasets/murodbek/uzbek-speech-corpus) (108K utterances, 105 hours, 958 speakers). Used 1250 samples for training.

**Summarization:** `Qwen/Qwen2.5-1.5B` with QLoRA (4-bit) fine-tuned on [`csebuetnlp/xlsum`](https://huggingface.co/datasets/csebuetnlp/xlsum) Uzbek subset (4728 BBC Uzbek news articles with summaries). Dataset was transliterated from Cyrillic to Latin to match Whisper's output.

## Results

### STT (Whisper)

| | WER |
|---|---|
| Base model | 0.583 |
| Fine-tuned | 0.462 |
| Improvement | -20.8% |

### Summarization (Qwen + QLoRA)

| | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Base model | 0.049 | 0.009 | 0.043 |
| Fine-tuned | 0.107 | 0.026 | 0.091 |
| Improvement | +121% | +197% | +112% |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Fine-tune STT
```bash
python phase1_stt_finetune.py
```

### Fine-tune LLM
```bash
python phase2_llm_finetune.py
```

### Run pipeline
```bash
python pipeline.py test_data/test_audio.mp3
```

Output:
```
Original Transcript: [Uzbek text]
Summary: [Uzbek summary]
```

## Project Structure

```
├── phase1_stt_finetune.py       # whisper fine-tuning
├── phase2_llm_finetune.py       # qwen qlora fine-tuning
├── pipeline.py                  # end-to-end inference
├── requirements.txt
├── test_data/
│   └── test_audio.mp3
├── models/
│   ├── whisper-small-uzbek-checkpoints/
│   └── qwen-uz-summarizer-ckpts/
├── results/
│   └── *.json                   # metrics and pipeline outputs
└── logs/
```

## Challenges

- XL-Sum dataset is in Cyrillic but Whisper outputs Latin — implemented transliteration using Uzbek's official mapping
- ~70% of training samples had summaries truncated due to long articles — future fix: filter these out or reduce max input length
- Overfitting after epoch 3 on summarization — used `load_best_model_at_end` to pick best checkpoint (step 750)
- `datasets` library broke script-based loading for xlsum — fell back to loading parquet files directly

## Hardware

| Phase | GPU | Time |
|---|---|---|
| STT | RTX 2070 Super (8GB) | ~45 min |
| LLM | RTX 3090 (24GB) | ~233 min |

