from flask import Flask, request, jsonify
from model.train import load_trained_components
from model.inference import predict_single
from utils.config import load_taxonomy
from utils.explain import explain_prediction

app = Flask(__name__)

model, tfidf = load_trained_components()
taxonomy = load_taxonomy()


@app.route("/predict", methods=["POST"])
def api_predict():
    """
    Predicts a category for a single transaction string.
    Request:
    {
        "merchant_name": "Starbucks Coffee"
    }
    """
    data = request.get_json()
    text = data.get("merchant_name", "")

    category, confidence = predict_single(text, model, tfidf)

    return jsonify({
        "merchant_name": text,
        "predicted_category": category,
        "confidence": confidence
    })


@app.route("/explain", methods=["POST"])
def api_explain():
    """
    Returns SHAP explanation values for a model prediction.
    Request:
    {
        "merchant_name": "Starbucks Coffee"
    }
    """
    data = request.get_json()
    text = data.get("merchant_name", "")

    shap_values = explain_prediction(text, model, tfidf)

    return jsonify({
        "merchant_name": text,
        "shap_values": shap_values
    })


@app.route("/feedback", methods=["POST"])
def api_feedback():
    """
    Saves a corrected label to feedback.csv for future retraining.
    Request:
    {
        "merchant_name": "Amazn Shop",
        "correct_category": "Shopping"
    }
    """
    data = request.get_json()
    merchant_text = data.get("merchant_name", "")
    correct_category = data.get("correct_category", "")

    with open("feedback.csv", "a") as f:
        f.write(f"{merchant_text},{correct_category}\n")

    return jsonify({"status": "saved"})


if __name__ == "__main__":
    print("Starting InHouseTX API on http://localhost:5000")
    app.run(debug=True)
