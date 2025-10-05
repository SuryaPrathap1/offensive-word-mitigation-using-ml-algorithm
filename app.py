from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.corpus import stopwords
import string

app = Flask(__name__)

# Download stopwords
nltk.download('stopwords')

# Load CSVs
d = pd.read_csv(r"C:\Users\91701\Desktop\projects\bad-words.csv")
data = pd.read_csv(r"C:\Users\91701\Desktop\projects\twitter.csv")

# Map labels
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive", 2: "No Hate and Offensive"})
data = data[["tweet", "labels"]]

# Preprocessing
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split()]
    text = " ".join(text)
    return text

data["tweet"] = data["tweet"].apply(clean)

# Train model
x = np.array(data["tweet"])
y = np.array(data["labels"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Prediction + censor function
def predict_and_censor(user_input):
    vector = cv.transform([user_input]).toarray()
    category = clf.predict(vector)[0]

    bad_words = d.iloc[:, 0].astype(str).str.lower().tolist()
    words = user_input.split()
    censored_text = ""
    for word in words:
        if word.lower() in bad_words:
            censored_text += "#" * len(word) + " "
        else:
            censored_text += word + " "
    return category, censored_text.strip()

# Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    category = None
    filtered_text = None
    user_input = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        category, filtered_text = predict_and_censor(user_input)
    return render_template("index.html", category=category, filtered_text=filtered_text, user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)
