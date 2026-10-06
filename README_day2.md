# Day 2 — Fine-tuned BERT News Classifier

## What I built
Fine-tuned a BERT model on AG News dataset to classify news headlines into
4 categories: World, Sports, Business, and Sci/Tech.

## Results
- Overall Accuracy: 89%
- Sports F1: 97%
- World F1: 87%
- Business F1: 86%
- Sci/Tech F1: 86%

## Tech Stack
- HuggingFace Transformers
- BERT (bert-base-uncased)
- PyTorch
- Streamlit
- Plotly
- Scikit-learn

## What I learned
- Tokenization and dataset preparation
- Full fine-tuning vs frozen layers
- HuggingFace Trainer API
- Evaluation metrics — precision, recall, F1
- Batch inference pipelines
- Wrapping ML models in Streamlit apps

## How to run
pip install -r requirements_day2.txt
streamlit run app_day2.py

## Model Performance
| Category | Precision | Recall | F1 |
|---|---|---|---|
| World | 96% | 80% | 87% |
| Sports | 95% | 99% | 97% |
| Business | 86% | 86% | 86% |
| Sci/Tech | 81% | 91% | 86% |

## Author
Tsegai Yhdego
PhD Industrial Engineering — FAMU-FSU
AI/ML Researcher — R-SEAT Center
