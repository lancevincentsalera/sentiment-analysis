from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@app.route("/api/sentiment", methods=["POST"])
def sentiment_analysis():
    data = request.get_json()

    #############
    #pailis nya ko aning "text" herns, wala koy access sa swagger gud
    #############
    if "text" not in data:
        return jsonify({"error": "Missing 'text' key in request body"}), 400
    
    text = data["text"] 
    result = sentiment_pipeline(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)