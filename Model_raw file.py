import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
a=pd.read_csv("data.csv")
l=a["label"]
content=a["text"]
vec=CountVectorizer()
x=vec.fit_transform(content)
#Now convert spam labels to 1 and ham to 0
y=l.map({
    "ham":0,
    "spam":1
})
model=MultinomialNB()
model.fit(x,y)
print(y.value_counts())