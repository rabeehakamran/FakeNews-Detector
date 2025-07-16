readme = """
# Fake News Detector using Machine Learning

This project is a machine learning-based text classification system that predicts whether a news article is **real** or **fake**.

I explored different models and preprocessing techniques, and selected **Naive Bayes** as the final model based on its performance on unseen test samples.

---

## Problem Statement

In an era of misinformation, it's important to develop tools that can help detect fake news articles. This project aims to classify news articles as either fake or real using natural language processing (NLP) and machine learning.

---

## Dataset

The dataset used is a combination of:

- `Fake.csv`
- `True.csv`

Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Each news article is labeled:
- `0` → Fake
- `1` → Real

---

## Tools & Technologies

- Python
- NLTK (for text preprocessing)
- Scikit-learn
  - TfidfVectorizer
  - Naive Bayes Classifier
- Matplotlib & Seaborn (for data visualization)

---

## Text Preprocessing

The following preprocessing steps were applied:

- Lowercasing text
- Removing punctuation and numbers
- Removing stopwords
- Stemming using `PorterStemmer`
- TF-IDF vectorization (top 5000 features)

---

## Model Selection

I initially tested both:

- Logistic Regression  
- Multinomial Naive Bayes (Final choice)

Although Logistic Regression gave good accuracy, Naive Bayes gave better generalization on real-world samples.

---

## Results

- **Accuracy**: ~98.8%
- **Evaluation**:
  - Precision, Recall, F1-score
  - Confusion Matrix
- Successfully tested on real and fake news examples

---

## Sample Prediction

```python
test_news = """In a press briefing held at the National Press Club, the finance minister
announced a 12% increase in education budget for the fiscal year 2025."""
print(predict_news(test_news))  
# 🚨 Real News
```
"""
---

## How to Run

1. Clone the repo or download the `.ipynb` notebook  
2. Install dependencies (NLTK, scikit-learn, pandas)  
3. Download stopwords via NLTK  
4. Run all cells in Jupyter/Colab

---

## 👩‍💻 Author

**Rabeeha Kamran**  
Undergraduate | Machine Learning Enthusiast  
✨ Dreaming big, building small ✨

---

## ⭐️ If you found this project useful, feel free to give it a star!

