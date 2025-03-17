import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if not os.path.exists("model"):
    os.makedirs("model")

url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
data = pd.read_csv(url)

features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
X = data[features]
y = data['variety']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

with open("model/IrisClassifier.pkl", "wb") as f:
    pickle.dump(model, f)

