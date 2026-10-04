<div align="center">

# 🌱 FarmifAI — Core Development

**Notebooks and scripts used to build, train and evaluate the AI behind FarmifAI**

An offline agricultural assistant for rural Colombia: a fine-tuned small language model + hybrid RAG.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Base model" src="https://img.shields.io/badge/Base%20model-Qwen3.5--0.8B-6f42c1">
  <img alt="Fine-tuning" src="https://img.shields.io/badge/LoRA-Unsloth-00a67e">
  <img alt="Inference" src="https://img.shields.io/badge/Inference-llama.cpp%20(GGUF)-orange">
  <a href="https://huggingface.co/FarmifAI"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FarmifAI-FFD21E"></a>
</p>

</div>

---

## Overview

FarmifAI answers farmers' questions **without an internet connection**. It combines a Qwen3.5-0.8B model fine-tuned with LoRA and a hybrid retrieval pipeline (BM25 + dense embeddings + cross-encoder reranker) over Colombian agricultural technical documents.

This repository holds the **research and AI development work**: dataset generation, dataset quality evaluation, fine-tuning, GGUF export, RAG prototyping, and model / system evaluation.

> [!NOTE]
> The Android application lives in a separate repository: [FarmifAI app](https://github.com/Mikhaerys/FarmifAI).


## Repository structure

```text
├── Notebooks/
│   ├── Colab/        # All 9 notebooks, written for Google Colab
│   └── Local/        # 6 notebooks adapted for local / VPS runs + install scripts
├── Scripts/          # CLI tools to re-compute metrics and sync the knowledge base
├── .env.example      # API key template
└── README.md
```

## Notebooks

| Stage | Notebook | What it does | Run |
|---|---|---|---|
| Data | `dataset_generator` | Turns knowledge chunks into ChatML conversations (plain, farmer-style Spanish) with DeepSeek. Resumable; tracks per-crop stats. | [Colab](https://colab.research.google.com/github/Mikhaerys/FarmifAI_Core_Development/blob/main/Notebooks/Colab/dataset_generator.ipynb) |
| Data | `evaluacion_dataset_SLM` | Scores dataset quality across 9 metric groups (structure, deduplication, lexical diversity, crop balance, format, …) with a green/yellow/red scorecard; exports CSV and HTML reports. | [Colab](https://colab.research.google.com/github/Mikhaerys/FarmifAI_Core_Development/blob/main/Notebooks/Colab/evaluacion_dataset_SLM.ipynb) |
| Training | `FarmifAI_LoRA_Training` | LoRA fine-tuning of `unsloth/Qwen3.5-0.8B` (T4-friendly, fp16) with built-in ROUGE / BLEU / NLI / SelfCheckGPT checks. | [Colab](https://colab.research.google.com/github/Mikhaerys/FarmifAI_Core_Development/blob/main/Notebooks/Colab/FarmifAI_LoRA_Training.ipynb) |
| Training | `Qwen3.5-0.8B_Merge_GGUF_Export` | Merges the LoRA adapter into the base model and exports quantized GGUF files; optional upload to Hugging Face. | [Colab](https://colab.research.google.com/github/Mikhaerys/FarmifAI_Core_Development/blob/main/Notebooks/Colab/Qwen3.5-0.8B_Merge_GGUF_Export.ipynb) |
| RAG | `rag_hybrid_reranker` | Hybrid retrieval: BM25 + `multilingual-e5-small`, fused with Reciprocal Rank Fusion and reranked by a cross-encoder (top 3 chunks). | [Colab](https://colab.research.google.com/github/Mikhaerys/FarmifAI_Core_Development/blob/main/Notebooks/Colab/rag_hybrid_reranker.ipynb) |
| RAG | `rag_slm_colab` | Gradio chat that connects the RAG pipeline to the GGUF model through `llama.cpp`. | [Colab](https://colab.research.google.com/github/Mikhaerys/FarmifAI_Core_Development/blob/main/Notebooks/Colab/rag_slm_colab.ipynb) |
| Evaluation | `evaluacion_rag_retrieval` | Retrieval benchmark: Hit Rate @20 / @3 / @1 and which retriever (BM25, semantic, both) found the right chunk. | [Colab](https://colab.research.google.com/github/Mikhaerys/FarmifAI_Core_Development/blob/main/Notebooks/Colab/evaluacion_rag_retrieval.ipynb) |
| Evaluation | `FarmifAI_Model_Evaluation` | Evaluates the GGUF model with a given context: format adherence, ROUGE, BLEU, Flesch (Spanish), cosine similarity, BERTScore, NLI, SelfCheckGPT and LLM-as-a-Judge. Checkpointed. | [Colab](https://colab.research.google.com/github/Mikhaerys/FarmifAI_Core_Development/blob/main/Notebooks/Colab/FarmifAI_Model_Evaluation.ipynb) |
| Evaluation | `FarmifAI_RAG_System_Evaluation` | End-to-end evaluation (retriever + reranker + SLM) with dynamically retrieved context: RAG triad, BERTScore, G-Eval and per-component latency. | [Colab](https://colab.research.google.com/github/Mikhaerys/FarmifAI_Core_Development/blob/main/Notebooks/Colab/FarmifAI_RAG_System_Evaluation.ipynb) |

## Scripts

Command-line helpers in [`Scripts/`](Scripts/) to repair or recompute metrics without repeating expensive evaluation runs.

| Script | Purpose |
|---|---|
| `reeval_metrics.py` | Recomputes local metrics (format adherence, ROUGE, BLEU, Flesch, cosine similarity, NLI) while keeping LLM-judge scores. |
| `reeval_bertscore.py` | Adds BERTScore (precision, recall, F1) to an existing results file. |
| `reeval_llm_judge.py` | Repairs or completes LLM-judge metrics after API failures, with retries and `--fix_zeros`. |
| `sync_knowledge_base.py` | Adds any `<knowledge>` chunk found in the dataset but missing from `knowledge_base.json` (`--dry-run`, `--backup`). |

```bash
python Scripts/reeval_llm_judge.py --input eval_progress.jsonl --provider openrouter --fix_zeros
python Scripts/sync_knowledge_base.py --dry-run
```

Run any script with `--help` for the full list of options.

## Getting started

**Google Colab:** open a notebook from the table above. A GPU runtime (a T4 is enough) is recommended.

**Local machine or VPS:**

```bash
git clone https://github.com/Mikhaerys/FarmifAI_Core_Development.git
cd FarmifAI_Core_Development/Notebooks/Local

./install_deps.sh            # Linux / VPS (CUDA)
# .\install_deps.ps1         # Windows (PowerShell)

source venv/bin/activate     # Windows: .\venv\Scripts\Activate.ps1
jupyter notebook
```

The installer creates a virtual environment and installs PyTorch, `llama-cpp-python` (with CUDA when available) and Unsloth (NVIDIA GPU only). See [`Notebooks/Local/README.md`](Notebooks/Local/README.md) for remote-server setup and implementation notes.

## Configuration

LLM-as-a-Judge and dataset generation call external APIs. Provide your key as an environment variable or in the notebook's configuration cell (see [`.env.example`](.env.example)):

| Variable | Used by |
|---|---|
| `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` | `reeval_llm_judge.py` (depending on `--provider`) |

## Data format

Training and evaluation data is JSONL in ChatML format, with the retrieved context inside `<knowledge>` tags and the answer split into reasoning and final answer:

```json
{
  "messages": [
    {"role": "system",    "content": "Format and behavior instructions"},
    {"role": "user",      "content": "<knowledge>\n{chunk}\n</knowledge>\n\n{farmer's question}"},
    {"role": "assistant", "content": "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"}
  ]
}
```

Datasets, `knowledge_base.json` and embedding files are not stored in this repository; place them next to the notebook you are running (or in `output/JSON/`).

## Models and datasets

All releases are published under the [FarmifAI organization on Hugging Face](https://huggingface.co/FarmifAI), including the GGUF-quantized model and the training dataset.

<!-- TODO: add a License section once one is chosen. -->
