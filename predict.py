import joblib

model=joblib.load("models/logistic.pkl")

while True:

    text=input("Enter Tweet : ")

    pred=model.predict([text])[0]

    print("Prediction :",pred)
