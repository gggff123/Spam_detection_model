# Spam_detection_model
Spam detection model (pre-made) for **absoloute begginers** along with raw code.

## This is a fun hobby project and a tool for begginers to learn Data science & ML to build real world applications that will actually help them land a job and also get a litle knowledge about how ai works

# File structure
```
[Main]--
     Model_raw file.py #contains the raw source code of the model
     README.MD #readme
     use model.py #contains how to actually run the pre made model
```

# Where to download the model?
You can download the model from the releases section 

## IMPORTANT Download both vectorizer.pkl and spam_detect.pkl for it to work

## How to run the model (pre_made)
```python
#SPAM DETECTION MODEL, USE CASE , HOW TO USE
import joblib
#Loads models
model=joblib.load("spam_detect.pkl")
vec=joblib.load("vectorizer.pkl")
#Takes input
message=input("Enter your message: ")
x=vec.transform([message])
#Prediction
prediction=model.predict(x)[0]
if prediction==1:
    print("Spam")
else:
    print("Ham")
#Probability of prediction
probs = model.predict_proba(x)[0]
print("Ham probability :", probs[0])
print("Spam probability:", probs[1])
```

# ACKNOWLEDGEMENT
- mshenoda/spam-messages for the dataset
- Hugginface
- sklearn,joblib,numpy,pandas

# MIT OPEN-SOURCE LICENSE 
