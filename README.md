# 🤖 LLM-Powered Code Review Assistant

> Automates first-pass pull request reviews using a fine-tuned CodeLlama model trained on 80,000 PR diffs — classifying bug patterns, anti-patterns, and security vulnerabilities in real time.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Model](https://img.shields.io/badge/Model-CodeLlama--7B-green.svg)](https://huggingface.co/codellama)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Actions](https://img.shields.io/badge/CI-GitHub_Actions-black.svg)](https://github.com/features/actions)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Anti-pattern Detection Rate | **73%** |
| False Positive Rate | **< 8%** |
| Training Data | **80,000 PR diffs** |
| F1 Score (weighted) | **0.71** |
| p95 Inference Latency | **< 4s** |

---

## 🏗️ Architecture

```
Developer opens PR
        ↓
GitHub Webhook fires
        ↓
FastAPI Server receives event
        ↓
tree-sitter parses diff (Python / JS / Go)
        ↓
CodeLlama-7B (LoRA fine-tuned) runs inference
        ↓
Annotated inline comments posted to PR
```

---

## 📁 Project Structure

```
llm-code-review-assistant/
├── src/
│   ├── parser/          # Diff parsing with tree-sitter
│   ├── model/           # Model loading, inference, fine-tuning
│   ├── api/             # FastAPI server
│   └── bot/             # GitHub bot / webhook handler
├── training/            # Fine-tuning scripts
├── evaluation/          # Eval scripts & metrics
├── data/                # Data collection & preprocessing
├── deployment/          # Docker, docker-compose
├── scripts/             # Utility scripts
├── tests/               # Unit & integration tests
├── notebooks/           # EDA & analysis
└── .github/workflows/   # CI/CD pipelines
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/llm-code-review-assistant.git
cd llm-code-review-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Setup Environment

```bash
cp .env.example .env
# Fill in your GitHub token, HuggingFace token, etc.
```

### 3. Run the API Server

```bash
uvicorn src.api.server:app --reload --port 8000
```

### 4. Setup GitHub Webhook

- Go to your repo → Settings → Webhooks → Add webhook
- Payload URL: `https://your-server.com/webhook`
- Content type: `application/json`
- Events: `Pull requests`

---

## 🧠 Fine-tuning

```bash
# Collect training data
python scripts/collect_data.py --limit 80000 --output data/raw/

# Preprocess
python scripts/preprocess.py --input data/raw/ --output data/processed/

# Fine-tune CodeLlama
python training/finetune.py \
  --model codellama/CodeLlama-7b-hf \
  --data data/processed/train.jsonl \
  --output models/codellama-review-lora \
  --epochs 3 \
  --lora-rank 16
```

---

## 🧪 Evaluation

```bash
python evaluation/evaluate.py \
  --model models/codellama-review-lora \
  --test-data data/processed/test.jsonl \
  --output evaluation/results/
```

---

## 🐳 Docker

```bash
docker-compose up --build
```

---

## 📄 License

MIT — see [LICENSE](LICENSE)
