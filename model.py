import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Données fictives
data = {
    "age": [25, 40, 30, 50, 22, 45],
    "revenu": [150000, 300000, 200000, 500000, 120000, 400000],
    "niveau_etude": ["Secondaire", "Supérieur", "Supérieur", "Aucun", "Secondaire", "Supérieur"],
    "chome": ["Oui", "Non", "Non", "Oui", "Oui", "Non"]
}

df = pd.DataFrame(data)

le = LabelEncoder()
df["niveau_etude"] = le.fit_transform(df["niveau_etude"])
df["chome"] = le.fit_transform(df["chome"])

X = df[["age", "revenu", "niveau_etude"]]
y = df["chome"]

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))