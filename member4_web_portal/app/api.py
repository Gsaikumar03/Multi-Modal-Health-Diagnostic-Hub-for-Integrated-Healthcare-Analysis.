from flask import Blueprint, request, jsonify
from app.services.heart_service import predict_heart
from app.services.xray_service import predict_xray
from app.services.chat_service import ChatService

api = Blueprint("api", __name__)

chat_service = ChatService()


# -----------------------------
# HEART DISEASE PREDICTION API
# -----------------------------
@api.route("/predict-heart", methods=["POST"])
def heart():

    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON input"}), 400

    try:
        # Send dictionary directly to model service
        result = predict_heart(data)

        probability = result["risk_probability"]

        # Convert probability to label
        prediction_label = "High Risk" if probability > 0.5 else "Low Risk"

        return jsonify({
            "prediction": prediction_label,
            "probability": probability
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# XRAY PNEUMONIA PREDICTION API
# -----------------------------
@api.route("/predict-xray", methods=["POST"])
def xray():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        result = predict_xray(file)

        probability = result["pneumonia_probability"]

        prediction_label = "Pneumonia Detected" if probability > 0.5 else "Normal"

        return jsonify({
            "prediction": prediction_label,
            "confidence": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# MEDICAL CHATBOT API
# -----------------------------
@api.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Question is required"}), 400

    question = data["question"]

    try:
        response = chat_service.ask_question(question)

        return jsonify({
            "question": response.get("question"),
            "answer": response.get("answer"),
            "confidence": response.get("confidence"),
            "context": response.get("context")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
