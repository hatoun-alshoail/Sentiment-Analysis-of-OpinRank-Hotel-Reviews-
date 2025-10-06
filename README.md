🏨 Sentiment Analysis of OpinRank Hotel Reviews

This project implements a Transformer-based sentiment analysis pipeline to classify hotel reviews into Positive, Negative, or Neutral categories using the OpinRank dataset. The goal is to develop and fine-tune high-performance NLP models for real-world text classification tasks.

🚀 Project Overview

- Built with Python, PyTorch, and Hugging Face Transformers.

- Fine-tuned two transformer models — BERT and DistilRoBERTa — on a curated and balanced hotel review dataset.

- Integrated Optuna for automated hyperparameter optimization.

- Deployed the best-performing model using Gradio and Hugging Face Spaces for live inference.

🧩 Key Components

Data Preprocessing

- Extracted and cleaned hotel reviews from the OpinRank dataset.

- Applied VADER for initial sentiment labeling.

- Balanced the dataset using synthetic data generation with GPT-3.5-Turbo.

Model Training

- Tokenized text using BertTokenizerFast.

- Fine-tuned models with the Hugging Face Trainer API.

- Used Optuna to optimize learning rate, epochs, and batch size.

- Implemented AdamW optimizer and linear learning rate scheduler.

Evaluation

- Metrics: Accuracy, Precision, Recall, F1-Score.

- Analyzed validation and test results to detect overfitting and assess generalization.

Deployment

- Built an interactive demo using Gradio.

- Hosted the fine-tuned model on Hugging Face Spaces for public access.

🛠️ Tech Stack
- Python 3.10+

- PyTorch

- Transformers (Hugging Face)

- Optuna

- Pandas, NumPy, Matplotlib

- Gradio / Hugging Face Spaces
