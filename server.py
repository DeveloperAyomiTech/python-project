from flask import Flask, request, jsonify
import os
from app.emotion_detector import detect_emotions, format_emotion_output

app = Flask(__name__)

# Environment variables for Watson API credentials
api_key = os.getenv('WATSON_API_KEY')
url = os.getenv('WATSON_URL')

if not api_key or not url:
    raise ValueError("Please set the WATSON_API_KEY and WATSON_URL environment variables.")

@app.route('/detect_emotions', methods=['POST'])
def detect_emotions_endpoint():
    # Get JSON data from the request
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    
    # Detect emotions
    emotions = detect_emotions(api_key, url, text)
    
    # Format the response
    result = format_emotion_output(emotions)
    
    return jsonify({'result': result})

@app.route('/')
def index():
    return "Welcome to the Emotion Detection Service! Use the /detect_emotions endpoint to detect emotions."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
