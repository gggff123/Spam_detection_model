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