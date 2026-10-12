# Day 2 — Fine-Tuned BERT News Classifier

**AI Skills Bootcamp | Tsegai Yhdego, PhD**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-orange)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-89%25-brightgreen)](https://github.com/Tsegai20)

---

## What I Built

Fine-tuned BERT (bert-base-uncased) on the AG News dataset to classify news headlines into 4 categories — World, Sports, Business, and Sci/Tech — achieving 89% accuracy on unseen test data.

---

## Results

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| World | 96% | 80% | 87% |
| Sports | 95% | 99% | **97%** |
| Business | 86% | 86% | 86% |
| Sci/Tech | 81% | 91% | 86% |
| **Overall** | **90%** | **89%** | **89%** |

**Training:** 2000 samples · 3 epochs · bert-base-uncased · CPU

**Key finding:** Sports achieves 97% F1 — sports language is highly distinctive. World news is hardest to recall at 80% due to vocabulary overlap with other categories.

---

## What I Built — Two Deliverables

### 1. Fine-Tuned BERT Model
- Loaded pre-trained bert-base-uncased
- Added 4-class classification head
- Full fine-tuning — all layers updated
- Evaluated with precision, recall, F1 per class
- Saved model weights to disk

### 2. Streamlit Inference App
- Upload any news headline
- Get predicted category with confidence score
- Probability bar chart across all 4 classes
- Batch classification — paste multiple headlines at once
- Category distribution pie chart for batch results

---

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Tokenization | AutoTokenizer with truncation and padding to 128 tokens |
| Transfer learning | Pre-trained BERT weights adapted to classification |
| Full fine-tuning | All 110M parameters updated during training |
| Trainer API | HuggingFace Trainer handles training loop automatically |
| Evaluation metrics | Precision, recall, F1 per class via scikit-learn |
| Batch inference | Multiple headlines classified in one forward pass |
| Model persistence | Saved with save_pretrained for reuse |

---

## Why Full Fine-Tuning vs Frozen Layers

| Approach | When to use | Accuracy |
|----------|-------------|----------|
| Full fine-tuning | 1000+ samples, domain close to pre-training | Higher |
| Frozen BERT + train head | Very little data, fast training needed | Lower |
| Partial freeze | Domain far from pre-training data | Middle |

**This project uses full fine-tuning** — AG News is close to BERT's Wikipedia pre-training domain and we have 2000 samples.

---

## Project Structure
day2/
├── app_day2.py # Streamlit inference app
├── requirements_day2.txt # Dependencies
├── my_news_classifier/ # Saved fine-tuned model
│ ├── config.json
│ ├── model.safetensors
│ └── tokenizer files
└── README.md


---

## Run Locally

```bash
git clone https://github.com/Tsegai20/ai-skills-bootcamp-day2.git
cd ai-skills-bootcamp-day2
pip install -r requirements_day2.txt

# Run inference app
streamlit run app_day2.py
```

**Note:** Model weights (~440MB) are required. Run the training notebook first or download from the releases tab.

---

## Training Code

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

training_args = TrainingArguments(
    output_dir="./news_classifier",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

trainer.train()
```

---

## Skills Added to Resume

- HuggingFace Transformers fine-tuning
- BERT text classification
- Transfer learning
- HuggingFace Trainer API
- Model evaluation — precision, recall, F1
- Batch inference pipelines
- Streamlit ML app deployment

---

## Part of AI Skills Bootcamp

| Day | Project | Repo |
|-----|---------|------|
| 1 | Streamlit AI Apps | [ai-skills-bootcamp-day1](https://github.com/Tsegai20/ai-skills-bootcamp) |
| **2** | **Fine-Tuned BERT Classifier** | **You are here** |
| 3 | FastAPI ML Serving | Coming |
| 4 | RAG Pipeline | Coming |
| 5 | MLflow + Federated Learning | Coming |
| 6 | Docker Federated System | [ai-skills-bootcamp6](https://github.com/Tsegai20/ai-skills-bootcamp6) |

---

## Author

**Tsegai Yhdego, PhD**  
Industrial Engineering — Machine Learning and Optimization  
Assistant Director, R-SEAT Center, FAMU-FSU  
USDOT-Funded Research — Intelligent Transportation and Autonomous Systems  
[GitHub](https://github.com/Tsegai20) · [LinkedIn](https://www.linkedin.com/in/tsegai-yhdego-ph-d-b6209b118/)
