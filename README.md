# Women Safety Analysis Using Machine Learning

A Machine Learning-based sentiment analysis system that analyzes tweets related to women's safety in Indian cities. The project classifies tweets into **Positive**, **Negative**, or **Neutral** sentiments and generates insights about public perception of women's safety using Natural Language Processing (NLP).

---

## Overview

Women's safety is one of the major concerns in modern society. Social media platforms like Twitter (X) contain valuable public opinions regarding safety, crime, law enforcement, and women's issues.

This project uses Machine Learning algorithms to classify the sentiment of tweets and estimate the overall safety perception of different Indian cities.

---

## Features

- Tweet preprocessing using NLP
- Text cleaning and normalization
- Stopword removal
- Lemmatization
- TF-IDF Vectorization
- Sentiment Classification
- Hyperparameter tuning using GridSearchCV
- Comparison of multiple Machine Learning algorithms
- Save trained models using Joblib
- Flask web interface for prediction
- Women's Safety Index calculation (Optional)

---

## Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- NLTK
- TF-IDF Vectorizer
- Logistic Regression
- Naïve Bayes
- Decision Tree
- K-Nearest Neighbors (KNN)


---

## Machine Learning Models

The project compares the performance of the following models:

- Logistic Regression
- Multinomial Naïve Bayes
- Decision Tree
- K-Nearest Neighbors (KNN)

Hyperparameter tuning is performed using **GridSearchCV** to improve classification accuracy.

---

## Dataset

Dataset contains tweets related to women's safety.

Example format:

| Tweet | Label |
|--------|--------|
| Delhi is becoming unsafe for women at night | Negative |
| Mumbai police responded quickly | Positive |
| Street lighting has improved | Positive |
| Harassment cases increasing | Negative |
| The city is okay | Neutral |

Labels:

- Positive
- Negative
- Neutral

---

## Project Structure

```
Women-Safety-Analysis/
│
├── dataset/
│   └── women_safety_tweets.csv
│
├── models/
│   ├── logistic.pkl
│   ├── naive_bayes.pkl
│   ├── decision_tree.pkl
│   └── knn.pkl
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── screenshots/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Women-Safety-Analysis.git
```

Move into the project directory

```bash
cd Women-Safety-Analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train.py
```

This will:

- Preprocess the tweets
- Train all ML models
- Perform hyperparameter tuning
- Save the best trained models inside the `models/` directory

---

Example

```
Enter Tweet:

Women feel unsafe while travelling late at night.

Prediction:

Negative
```

---

## Run Flask Application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

Enter any tweet to predict its sentiment.

---

## NLP Pipeline

```
Raw Tweet
      │
      ▼
Text Cleaning
      │
      ▼
Lowercase Conversion
      │
      ▼
Remove URLs
      │
      ▼
Remove Mentions
      │
      ▼
Remove Hashtags
      │
      ▼
Remove Stopwords
      │
      ▼
Lemmatization
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Machine Learning Model
      │
      ▼
Sentiment Prediction
```

---

## Women's Safety Index (Optional)

The project can estimate a Women's Safety Index for different cities.

Formula

```
Safety Index = ((Positive Tweets − Negative Tweets)
/ Total Tweets) × 100
```

Higher scores indicate a more positive public perception of women's safety.

---

## Evaluation Metrics

The following metrics are used for model evaluation:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Expected Accuracy

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 88–92% |
| Naïve Bayes | 84–89% |
| Decision Tree | 75–82% |
| KNN | 72–80% |

*(Accuracy may vary depending on dataset size and quality.)*

---

## Future Enhancements

- Real-time Twitter API integration
- Deep Learning using LSTM
- BERT-based sentiment classification
- RoBERTa / DistilBERT implementation
- Interactive dashboard using Streamlit
- City-wise safety heatmap
- Real-time sentiment monitoring
- Geolocation-based safety visualization

---

## Applications

- Smart City Analytics
- Crime Trend Analysis
- Public Sentiment Monitoring
- Women's Safety Awareness
- Government Decision Support
- Urban Safety Research

---

## Author

**Krish Gupta**

B.Tech Artificial Intelligence & Machine Learning

University School of Automation and Robotics (USAR)

Guru Gobind Singh Indraprastha University

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- Scikit-learn
- NLTK
- Pandas
- NumPy
- Flask
- Python Community
