import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("equipment_anomaly_data_india_balanced.csv")

le_equipment = LabelEncoder()
le_location = LabelEncoder()
df['equipment_enc'] = le_equipment.fit_transform(df['equipment'])
df['location_enc'] = le_location.fit_transform(df['location'])

X = df[['temperature', 'pressure', 'vibration', 'humidity', 'equipment_enc', 'location_enc']]
y = df['faulty'].astype(int)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump((clf, le_equipment, le_location), f)
