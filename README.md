# Customer Reviews Intelligence System using NLP

> AI4001 – Fundamentals of Natural Language Processing | NLP Project

---

## Overview

This project builds an end-to-end **Customer Reviews Intelligence System** applied to the **Amazon Fine Food Reviews** dataset. It automates three tasks that would be infeasible to perform manually at scale:

| Task | Technique | Best Result |
|---|---|---|
| **Sentiment Analysis** | VADER / Logistic Regression + TF-IDF | ~93% accuracy |
| **Intent Classification** | Weak supervision + Logistic Regression | ~80–85% accuracy |
| **Topic Modeling** | NMF (Non-Negative Matrix Factorization) | 5 coherent topics |

A **Gradio web interface** ties all three components together into a live customer service dashboard.

---

## Dataset

**Amazon Fine Food Reviews** — [Kaggle (snap)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

| Column | Description |
|---|---|
| `Score` | Star rating 1–5 (used to derive sentiment label) |
| `Summary` | Short review headline |
| `Text` | Full review body — primary input feature |

**Label construction:**
- Score ≥ 4 → **Positive (1)**
- Score ≤ 2 → **Negative (0)**
- Score = 3 → **Excluded** (ambiguous — often mixed sentiment)

**Working sample:** 50,000 randomly sampled reviews (`random_state=42`) for reasonable runtime.

---

## Project Structure

```
AI4001_NLP_Project.ipynb     # Main notebook (all tasks)
Reviews.csv                  # Dataset — download from Kaggle
README_NLP_Project.md        # This file
```

---

## Setup & Requirements

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn gradio
```

### Download NLTK Data (run once)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

### Dataset

Download `Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and place it in the same directory as the notebook.

---

## Pipeline

```
Reviews.csv (568k rows)
        │
        ▼
  50,000 row sample  →  Drop Score=3  →  Binary Label (0/1)
        │
        ▼
  Text Preprocessing
  ├── Lowercase
  ├── Strip HTML tags (<br />, etc.)
  ├── Remove URLs
  ├── Remove non-alpha characters & digits
  ├── Tokenize (NLTK word_tokenize)
  ├── Remove stopwords
  └── Lemmatize (WordNetLemmatizer)
        │
        ├─────────────────────────────────────────────────┐
        ▼                                                 ▼
  Task 1: Sentiment Analysis              Task 2: Intent Classification
  ├── VADER (rule-based, on raw text)     ├── Keyword weak supervision
  ├── LR + BoW (CountVectorizer)          │   (Refund/Delivery/Complaint/General)
  └── LR + TF-IDF ← Best                 └── LR + TF-IDF (class_weight='balanced')
        │
        ▼
  Task 3: Topic Modeling (NMF, 5 topics)
        │
        ▼
  Gradio Interface (live demo)
```

---

## Task 1 — Sentiment Analysis

### Methods

**VADER (Rule-Based)**
- No training required — uses a curated word-polarity dictionary
- Applied to **original raw text** (capitalisation and punctuation matter to VADER)
- Compound score ≥ 0.05 → Positive; ≤ −0.05 → Negative

**Logistic Regression + BoW**
- `CountVectorizer(max_features=10_000, ngram_range=(1,2))`
- Unigrams + bigrams, top 10k features

**Logistic Regression + TF-IDF** ← Best
- `TfidfVectorizer(max_features=10_000, ngram_range=(1,2), sublinear_tf=True)`
- Log-normalised TF, downweights common uninformative words

### Results

| Model | Accuracy |
|---|---|
| VADER (rule-based) | ~86% |
| LR + BoW | ~91% |
| **LR + TF-IDF** | **~93%** |

TF-IDF outperforms BoW because it penalises high-frequency but uninformative words (e.g. "product", "amazon") and upweights rare but discriminative words (e.g. "rancid", "delicious").

### Interpretability

Top positive indicators (LR coefficients): `great`, `love`, `delicious`, `perfect`, `highly recommend`  
Top negative indicators: `terrible`, `waste`, `disgusting`, `returned`, `awful`

---

## Task 2 — Intent Classification

### Labelling Strategy (Weak Supervision)

Since the dataset has no intent labels, keyword-based rules generate approximate training labels with priority ordering:

| Priority | Intent | Trigger Keywords |
|---|---|---|
| 1 (highest) | **Refund Request** | refund, return, money back, reimburse, cancel, chargeback |
| 2 | **Delivery Issue** | late, delayed, not arrived, missing, wrong item, shipping, package |
| 3 | **Complaint** | broken, damaged, disgusting, expired, terrible, stale, rotten |
| 4 (default) | **General Query** | Everything else |

### Classifier

```
TF-IDF (10k features, bigrams)
    → LogisticRegression(class_weight='balanced')
```

`class_weight='balanced'` corrects for the majority-class imbalance (General Query dominates).

### Results

Accuracy: ~80–85% on keyword-derived labels. Performance is bounded by label quality — weak supervision introduces noise but is practical when no ground-truth intent labels exist.

---

## Task 3 — Topic Modeling (NMF)

### How NMF Works

NMF decomposes the document-term matrix **V** into:
- **W** (documents × topics): how strongly each document belongs to each topic
- **H** (topics × words): how strongly each word is associated with each topic

All values are non-negative, making the decomposition **additive** and naturally interpretable.

### Configuration

```python
TfidfVectorizer(max_features=5_000, ngram_range=(1,1), min_df=5, max_df=0.85)
NMF(n_components=5, random_state=42, max_iter=400)
```

### Discovered Topics

| Topic | Label | Example Top Words |
|---|---|---|
| 0 | Product Quality & Taste | good, great, taste, love, flavor, delicious |
| 1 | Coffee | coffee, cup, brew, espresso, roast, pod |
| 2 | Health & Nutrition | organic, natural, gluten, healthy, ingredient |
| 3 | Pet Food | dog, cat, food, treat, feed, pet |
| 4 | Snacks & Packaging | bag, chip, box, package, snack, cracker |

---

## Gradio Interface

A live web dashboard wraps all three models:

```python
import gradio as gr
demo.launch(share=True)   # generates a temporary public HTTPS URL
```

**Input:** Any review text  
**Output:**
- Predicted sentiment + confidence score
- Predicted customer intent (for routing)
- Dominant NMF topic + top keywords

This simulates a real-world customer service triage system where NLP models route reviews to the correct support team automatically.

---

## Evaluation

### Metrics Used

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision (weighted) | Of predicted class, how many are correct? |
| Recall (weighted) | Of actual class, how many were caught? |
| F1-Score (weighted) | Balance of Precision and Recall |
| Confusion Matrix | Breakdown of TP / FP / TN / FN per class |

### Business Priority

For complaint and refund detection, **Recall is the critical metric** — a missed complaint means an unaddressed dissatisfied customer (churn risk), whereas a false positive creates only minor operational overhead.

---

## Key Findings

- **LR + TF-IDF** is the best sentiment model (~93%) — TF-IDF's IDF weighting gives it a consistent edge over plain BoW
- **VADER** (~86%) is a strong zero-shot baseline requiring no training data — useful when labelled data is unavailable
- **Weak supervision** enables intent classification without manual labelling — practical but limited by keyword coverage and label noise
- **NMF** produced five coherent, human-interpretable topics aligning with real Amazon Fine Food categories
- The **Gradio interface** demonstrates how these models would be deployed in a production support system

---

## Limitations & Future Work

**Limitations:**
- Sentiment labels are derived from star ratings, not human annotation — noisy for borderline reviews
- Intent labels are keyword-generated — ceiling limited by keyword coverage
- Linear models (LR) have no understanding of context, negation, or sarcasm

**Future Improvements:**
- Fine-tune a transformer (BERT / RoBERTa) for sentiment and intent
- Collect human-annotated intent labels to replace weak supervision
- Increase `n_components` in NMF for finer-grained topic discovery
- Deploy the Gradio app persistently using Hugging Face Spaces

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading and manipulation |
| `nltk` | Tokenization, stopwords, lemmatization, VADER |
| `scikit-learn` | Vectorizers, Logistic Regression, NMF, metrics |
| `matplotlib`, `seaborn` | Visualizations, heatmaps |
| `gradio` | Live web demo interface |

---

## References

- Amazon Fine Food Reviews Dataset — https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- Hutto, C. & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*. ICWSM.
- Lee, D. & Seung, S. (1999). *Learning the parts of objects by non-negative matrix factorization*. Nature.
- Scikit-learn Documentation — https://scikit-learn.org
- Gradio Documentation — https://gradio.app
