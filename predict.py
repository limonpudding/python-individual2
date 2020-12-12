import SignaturesSearcher
from joblib import load
import pandas as pd
import os

for entry in os.listdir("models"):
    file = f"models\\{entry}"
    if os.path.isfile(file):
        print("Модель: " + file)
        testing = pd.read_json(SignaturesSearcher.searchAllInFolder("test_progs", False))
        model = load(file)
        predicted = model.predict(testing)
        print(predicted)