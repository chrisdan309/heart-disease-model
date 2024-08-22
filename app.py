import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


app = Flask(__name__)

model_dir = 'models'

pipeline_path = os.path.join(model_dir, 'pipeline.pkl')

try:
	pipeline = joblib.load(pipeline_path)
except FileNotFoundError:
	logging.error(f"Pipeline not found at {pipeline_path}")
	raise
except Exception as e:
	logging.error(f"Error loading pipeline: {e}")
	raise


@app.route('/')
def hello_world():
	return 'Hello World 2!'


@app.route('/predict', methods=['POST'])
def predict():
	try:
		data = request.json

		required_keys = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR',
						'ExerciseAngina', 'Oldpeak', 'ST_Slope']
		if not all(key in data for key in required_keys):
			return jsonify({'error': 'Invalid input data'}), 400

		input_data = pd.DataFrame([data])
		proba = pipeline.predict_proba(input_data)
		result = {'probability': float(proba[0][1])}
		logging.info(f"Input data: {data}")
		logging.info(f"Prediction: {result}")
		return jsonify(result)
	except Exception as e:
		logging.error(e)
		return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
	port = int(os.environ.get('PORT', 8080))
	app.run(host='0.0.0.0', port=port)
