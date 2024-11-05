from flask import Flask, jsonify, request
import joblib
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained models and TF-IDF vectorizer
logistic_model = joblib.load('Predictors/logistic_regression_model.pkl')
tweet_emotions_model = joblib.load('Predictors/tweet_emotions_predictor.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer/tfidf_vectorizer.pkl')
tweet_tfidf_vectorizer = joblib.load('tfidf_vectorizer/tweet_emotions-tfidf_vectorizer.pkl')
financial_emotions_model = joblib.load('Predictors/financial_emotions_predictor.pkl')
financial_tfidf_vectorizer = joblib.load('tfidf_vectorizer/financial_emotions-tfidf_vectorizer.pkl')
mental_health_model = joblib.load('Predictors/mental_health_model.pkl')
mental_health_tfidf_vectorizer = joblib.load('tfidf_vectorizer/mental_health_tfidf_vectorizer.pkl')
toxicLang_tfidf_vectorizer = joblib.load('tfidf_vectorizer/toxic_lang.pkl')
toxicLang_model = joblib.load('Predictors/toxic_lang.pkl')

# Define prediction route with text in the request body
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json
        mytext = data.get('text', '')

        if not mytext:
            return jsonify({'error': 'No text provided'}), 400

        # Transform the text using the loaded TF-IDF vectorizer
        transformed_text_logistic = tfidf_vectorizer.transform([mytext])
        transformed_text_tweet_emotions = tweet_tfidf_vectorizer.transform([mytext])
        transformed_text_financial_emotions = financial_tfidf_vectorizer.transform([mytext]) 
        transformed_mental_health_emotions = mental_health_tfidf_vectorizer.transform([mytext]) 
        transformed_toxic_lang = toxicLang_tfidf_vectorizer.transform([mytext]) 

        # Make predictions
        logistic_prediction = logistic_model.predict(transformed_text_logistic)
        tweet_emotions_prediction = tweet_emotions_model.predict(transformed_text_tweet_emotions)
        financial_emotions_prediction = financial_emotions_model.predict(transformed_text_financial_emotions)
        mental_health_prediction = mental_health_model.predict(transformed_mental_health_emotions)
        toxic_lang_prediction = toxicLang_model.predict(transformed_toxic_lang)

        # Return predictions
        return jsonify({
            'emotion': int(logistic_prediction[0]),
            'tweet_emotion_prediction': str(tweet_emotions_prediction[0]),
            'financial_emotion_prediction': str(financial_emotions_prediction[0]),
            'mental_health_prediction': str(mental_health_prediction[0]),
            'toxic_language': str(toxic_lang_prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define a health check route
@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'ML Model API is up and running!'}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001)
