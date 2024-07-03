from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors

import requests
import whisper
from transformers import pipeline
import os
import pandas as pd

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

app = Flask(__name__)
CORS(app)  # Use CORS with your Flask app

session = requests.Session()
session.verify = False
model = whisper.load_model("small")
sentiment_analyzer = pipeline("sentiment-analysis")

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result

@app.route('/')
def index():
    return "Server is running fine."

@app.route('/transcribe_and_analyze', methods=['POST'])
def transcribe_and_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.wav'):
        file_path = os.path.join('/tmp', file.filename)
        file.save(file_path)
        
        try:
            transcription = transcribe_audio(file_path)
            sentiment = analyze_sentiment(transcription)
            
            result_dict = {
                'file_name': file.filename,
                'text_infile': transcription,
                'sentiment_infile': sentiment
            }
            return jsonify(result_dict), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return jsonify({'error': 'Invalid file type, only .wav files are accepted'}), 400
    from app import app
