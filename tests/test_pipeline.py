import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
import os

# Cargar pipeline
current_dir = os.getcwd()
print(current_dir)
pipeline_path = os.path.join(current_dir, 'models', 'pipeline.pkl')
pipeline = load(pipeline_path)

# Cargar dataset
current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'data', 'heart.csv')
data = pd.read_csv(data_path)

# Definimos las variables predictoras y valor a predecir
X = data.drop(columns=['HeartDisease'])
y = data['HeartDisease']

# Creamos un conjunto para pruebas unitarias
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)


def test_pipeline_training():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    assert len(y_pred) == len(y_test)


def test_pipeline_prediction():
    y_pred = pipeline.predict(X_test)
    assert y_pred is not None
    assert len(y_pred) == len(y_test)
