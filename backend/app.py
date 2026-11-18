import joblib
import pandas as pd
from flask import Flask, jsonify, request
import io

app = Flask(__name__)

# Load the serialized model pipeline
model = joblib.load("superkart_model.joblib")

@app.route("/v1/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Convert single record to DataFrame to match preprocessing pipeline requirements
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": float(prediction)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/v1/predictbatch", methods=["POST"])
def predict_batch():
    try:
        # Retrieve the sent CSV file
        file = request.files["file"]
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
            
        # Read file contents and reconstruct DataFrame
        file_contents = file.read().decode("utf-8")
        input_df = pd.read_csv(io.StringIO(file_contents))
        
        # Generate predictions
        predictions = model.predict(input_df)
        
        # Convert predictions list into a indexed dictionary
        results = {str(i): float(pred) for i, pred in enumerate(predictions)}
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
