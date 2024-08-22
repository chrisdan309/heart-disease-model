import pytest
import sys
import os
from flask import Flask, json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_hello_world(client):
    response = client.get('/')
    assert response.data == b'Hello World 2!'


def test_predict(client):
    input_data = {
        "Age": 40,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 172,
        "ExerciseAngina": "N",
        "Oldpeak": 0,
        "ST_Slope": "Up"
    }

    response = client.post('/predict', json=input_data)
    data = json.loads(response.data)

    assert response.status_code == 200
    assert 'probability' in data
    assert 0.04570 - 10**-5 <= data['probability'] <= 0.04570 + 10**-5
