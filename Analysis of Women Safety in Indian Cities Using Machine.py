import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("dataset/women_safety_tweets.csv")

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

def clean(text):

    text=text.lower()

    text=re.sub(r"http\S+","",text)
    text=re.sub(r"@\w+","",text)
    text=re.sub(r"#","",text)
    text=re.sub(r"[^a-zA-Z ]","",text)

    words=text.split()

    words=[lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)

df["clean"]=df["tweet"].apply(clean)

X=df["clean"]
y=df["label"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipe=Pipeline([
    ("tfidf",TfidfVectorizer()),
    ("model",LogisticRegression())
])

params={
    "tfidf__max_features":[3000,5000],
    "tfidf__ngram_range":[(1,1),(1,2)],
    "model__C":[0.1,1,10]
}

grid=GridSearchCV(pipe,
                  params,
                  cv=5,
                  scoring="accuracy")

grid.fit(X_train,y_train)

pred=grid.predict(X_test)

print("Logistic Accuracy:",accuracy_score(y_test,pred))

joblib.dump(grid.best_estimator_,"models/logistic.pkl")


pipe=Pipeline([
    ("tfidf",TfidfVectorizer()),
    ("model",MultinomialNB())
])

params={
    "tfidf__max_features":[3000,5000],
    "tfidf__ngram_range":[(1,1),(1,2)],
    "model__alpha":[0.1,0.5,1]
}

grid=GridSearchCV(pipe,
                  params,
                  cv=5,
                  scoring="accuracy")

grid.fit(X_train,y_train)

pred=grid.predict(X_test)

print("Naive Bayes Accuracy:",accuracy_score(y_test,pred))

joblib.dump(grid.best_estimator_,"models/naive_bayes.pkl")


pipe=Pipeline([
    ("tfidf",TfidfVectorizer()),
    ("model",DecisionTreeClassifier())
])

params={
    "model__max_depth":[5,10,20],
    "model__criterion":["gini","entropy"]
}

grid=GridSearchCV(pipe,
                  params,
                  cv=5)

grid.fit(X_train,y_train)

pred=grid.predict(X_test)

print("Decision Tree Accuracy:",accuracy_score(y_test,pred))

joblib.dump(grid.best_estimator_,"models/decision_tree.pkl")

pipe=Pipeline([
    ("tfidf",TfidfVectorizer()),
    ("model",KNeighborsClassifier())
])

params={
    "model__n_neighbors":[3,5,7]
}

grid=GridSearchCV(pipe,
                  params,
                  cv=5)

grid.fit(X_train,y_train)

pred=grid.predict(X_test)

print("KNN Accuracy:",accuracy_score(y_test,pred))

joblib.dump(grid.best_estimator_,"models/knn.pkl")
