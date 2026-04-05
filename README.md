# 🧠 Comparing Psycholinguistic, Lexicon-Based, and Transformer-Derived Markers of Suicide-Related Language in a Reddit Proxy Corpus

> **Mila – Quebec AI Institute | HEC Montréal**  
> Yanis Bencheikh, Nelly Bozorgzad, Tegan Maharaj

> ⚠️ **988 Suicide & Crisis Lifeline** — If you or someone you know is in crisis, call or text **988**.

---

## 📌 Overview

AI systems are increasingly deployed in emotionally sensitive settings, yet current chatbots remain **unreliable in suicidal-crisis scenarios**. This project frames a responsible-AI audit question:

> *Which linguistic and emotional markers best distinguish suicide-labeled posts from non-suicide-labeled posts in a Reddit proxy corpus — and do transformer-derived features recover stronger signal than classical lexicon-based methods?*

We compare three families of features across a **balanced corpus of 232,074 Reddit posts**, reporting effect sizes (Cohen's d) alongside statistical significance — emphasizing that findings are **risk-related language markers, not clinical diagnostic tools**.

---

## ⚠️ Ethical Framing

This project is explicitly **not** a deployable crisis detector. It is a **responsible-AI audit of feature salience**:

- **Proxy data:** Reddit posts may not reflect clinical populations or real-world care settings
- **Misclassification risk:** Errors in detecting suicidal language could cause harm if deployed without clinical validation
- **Cognitive offloading:** Findings must not be used for clinical decision-making without proper validation and human oversight
- **Context awareness:** Single-post analysis ignores multi-turn and longitudinal context needed for reliable assessment

---

## 📦 Dataset

| Property | Value |
|---|---|
| **Source** | Reddit Proxy Corpus (Kaggle: `nikhileswarkomati/suicide-watch`) |
| **Subreddits** | r/SuicideWatch (suicide-labeled), r/teenagers (non-suicide-labeled) |
| **Total posts** | 232,074 |
| **Balanced split** | 116,037 suicide / 116,037 non-suicide |
| **Task** | Binary classification proxy (not clinical diagnosis) |

---

## 🔬 Methods

### A. Text Preprocessing
- URL removal, camelCase splitting artifact correction (`"dieI"` → `"die I"`)
- Lowercasing and whitespace normalization

### B. Lexicon-Based Markers

| Feature | Description |
|---|---|
| **N-grams** | Bigrams & trigrams ranked by Log-Odds ratio with effect size labels |
| **Self-reference density** | First-person singular pronoun rate (`I`, `me`, `my`, `myself`) |
| **Social-reference density** | Second/third-person pronoun rate |
| **Absolutist word density** | Rate of all-or-nothing terms (`always`, `nothing`, `completely`) linked to cognitive distortions |
| **NRC Emotion Scores** | Lexicon-based emotion distribution across 8 dimensions (anger, fear, sadness, joy, etc.) |

### C. Temporal Orientation
- SpaCy morphological analysis of verb tenses (Past / Present / Future)
- Future tense approximated via modal `will`

### D. Transformer-Derived Emotion Features
- **Model:** `SamLowe/roberta-base-go_emotions` (fine-tuned on GoEmotions)
- **Inference:** FP16, batch size 4096 on A100 GPU
- **Output:** 28-dimensional emotion probability vector per post
- Compared directly against NRC lexicon scores on the same emotion axes

### E. Topic Modeling
- **LDA** with k=5 topics on TF-IDF vectors (max 1,000 features)
- Qualitative inspection of topic dominance per class

### F. Statistical Framework
- **Welch's t-test** (unequal variance) for all group comparisons
- **Cohen's d** as primary effect size metric
- p-values treated as exploratory across multiple comparisons (not corrected)

---

## 📊 Key Findings

- **Self-focus language** shows the **largest and clearest between-class separation** — suicide-labeled posts use significantly more first-person singular language, consistent with crisis states and the need to be seen
- **Absolutist wording** is elevated in suicide-labeled posts, reflecting cognitive distortion patterns (rigid, polarized thinking)
- **Temporal orientation:** Suicide-labeled posts use more **past tense** and less **future tense** — consistent with rumination and reduced future-projection
- **Transformer emotions** yield stronger and more interpretable signal than NRC lexicon for self-focus and social-reference emotion dimensions
- **NRC lexicon** still produces statistically significant results for emotions correlated with suicidal ideation, though effect sizes are weaker and context-blind
- **Top discriminating n-grams** include crisis-related phrases (`"battle pills"`, `"chronic pain"`, `"writing suicide note"`) with small but reliable effect sizes (d ≈ 0.06–0.13)

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/suicide-language-markers.git
cd suicide-language-markers

pip install pandas numpy matplotlib seaborn scipy scikit-learn \
            nrclex textblob nltk spacy transformers datasets torch tqdm kagglehub
python -m spacy download en_core_web_sm
python -m textblob.download_corpora lite
```

### Data Access
```python
import kagglehub
cache_path = kagglehub.dataset_download("nikhileswarkomati/suicide-watch")
```
> Requires a Kaggle account and API token.

---

## 🚀 Pipeline

```
1. Data Loading & Caching          → Google Drive persistence (kagglehub)
2. EDA                             → Class distribution, word count, text length
3. Text Cleaning                   → quick_clean() — URL removal, artifact fix
4. N-gram Analysis                 → Log-odds + Cohen's d + Welch's t-test
5. Topic Modeling                  → LDA (k=5), qualitative topic inspection
6. Pronoun & Absolutist Analysis   → Density features + group comparison
7. Temporal Analysis               → SpaCy verb morphology (past/present/future)
8. NRC Emotion Analysis            → Lexicon-based 8-dim emotion scores
9. Transformer Emotion Analysis    → RoBERTa-GoEmotions (FP16, A100 batch=4096)
10. Comparative Visualization      → Effect size plots, radar charts, bar charts
```

---

## 📁 Repository Structure

```
.
├── data_exploration_suicide_watch.ipynb   # Main Colab notebook (end-to-end)
├── README.md
└── assets/
    └── poster.png                         # RAI conference poster (Mila / HEC Montréal)
```

---

## 📖 References

1. Fernandez et al. *Suicide and Depression Detection.* Kaggle Dataset.
2. Demszky et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* ACL 2020.
3. Mohammad & Turney (2013). *Crowdsourcing a Word-Emotion Association Lexicon.* Computational Intelligence.
4. Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.
5. Coppersmith et al. (2018). *Natural Language Processing of Social Media as Screening for Suicide Risk.* Biomedical Informatics Insights.
6. Rude et al. (2004). *Language use of depressed and depression-vulnerable college students.* Cognition & Emotion.
7. Ophir et al. (2020). *Linguistic Features of Suicidal and Nonsuicidal Notes.* Clinical Psychology Review.
8. Benton et al. (2017). *Ethical Research Protocols for Social Media Health Studies.* ACL Workshop.

---

## 👥 Authors & Affiliations

| Author | Affiliation |
|---|---|
| **Yanis Bencheikh** | HEC Montréal, Mila – Quebec AI Institute |
| Nelly Bozorgzad | HEC Montréal, Mila – Quebec AI Institute |
| Tegan Maharaj | HEC Montréal, Mila – Quebec AI Institute |

---

## 📜 License & Usage

This repository is released for **academic and research purposes only**. The findings must not be used to build or deploy clinical decision-support tools without appropriate clinical validation and ethical oversight. See `LICENSE` for details.
