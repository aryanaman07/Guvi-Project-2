# 🎬 Movie Review Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b.svg)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

## 📌 Project Overview

**Movie Review Sentiment Analyzer** is a **Natural Language Processing (NLP) project** that classifies movie reviews as **Positive** or **Negative**.
Manually reading and classifying hundreds of reviews is time-consuming and inconsistent. This project provides an automated solution using **TF-IDF vectorization** and **Logistic Regression** for sentiment classification.

---

## 📝 Problem Statement

It is difficult to manually read and classify hundreds of movie reviews as positive or negative.
➡️ **Question:** How can we develop a machine learning model that classifies a movie review as positive or negative based on its text?

---

## ✅ Summary

* Built a **text classification model** using **NLP techniques**.
* Dataset: **IMDB Movie Reviews** (labeled as Positive / Negative).
* Preprocessing: Cleaning text, removing stopwords, and punctuation.
* Feature Extraction: **TF-IDF Vectorizer**.
* Model: **Logistic Regression**.
* Achieved **\~89% accuracy** on test data.

---

## 📂 Project Structure

```
Guvi-Project-2/
│── Movie_Review_Sentiment_Analyzer_Notebook.ipynb   # Jupyter Notebook (training & analysis)
│── app.py                                           # Streamlit app for deployment
│── IMDB Dataset                                     # Dataset folder
│── logreg_sentiment_model.joblib                    # Trained logistic regression model
│── tfidf_vectorizer.joblib                          # Saved TF-IDF vectorizer
│── requirements.txt                                 # Dependencies
│── README.md                                        # Project documentation
```

---

## ⚙️ Functional Components

* **Data Preprocessing**: Cleaning reviews (lowercasing, stopword removal, punctuation removal).
* **Feature Extraction**: Convert text into numerical features using **TF-IDF**.
* **Model Training**: Train **Logistic Regression classifier**.
* **Evaluation**: Accuracy, Confusion Matrix, and F1-score.
* **Prediction**: Take **custom user input** to predict sentiment.

---

## 📊 Sample Dataset

| Review                                 | Sentiment |
| -------------------------------------- | --------- |
| "The movie was absolutely fantastic!"  | Positive  |
| "It was a complete waste of time."     | Negative  |
| "I loved the plot and the characters." | Positive  |

---

## 🎯 Expected Output

**Input Review:**

```
"The story was dull and disappointing."
```

**Model Prediction:**

```
Predicted Sentiment: Negative
Confidence - Positive: 0.13%, Negative: 99.87%
```

---

## 🚀 How to Run the Project

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/aryanaman07/Guvi-Project-2.git
cd Guvi-Project-2
```

### 🔹 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 3. Run Jupyter Notebook (for training & analysis)

```bash
jupyter notebook Movie_Review_Sentiment_Analyzer_Notebook.ipynb
```

### 🔹 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📦 Requirements

* Python 3.x
* scikit-learn
* pandas
* numpy
* joblib
* streamlit

*(All dependencies are listed in `requirements.txt`)*

---

## 📈 Results

* Achieved **~89% accuracy** using **TF-IDF + Logistic Regression**.
* Model generalizes well to unseen reviews.

---

## 💡 Future Improvements

* Use advanced models like **Naive Bayes, SVM, or Deep Learning (LSTM/BERT)**.
* Add **multiclass classification** (positive, neutral, negative).
* Deploy on **Heroku / Hugging Face Spaces**.

---

## 👨‍💻 Author

**Aryan Aman**
B.Tech CSE (Data Science) | ML & AI Enthusiast

---
