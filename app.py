from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import whisper
from transformers import pipeline
import os
import ssl
import pandas as pd
import io
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

app = Flask(__name__)
CORS(app)
os.makedirs('static', exist_ok=True)
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
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files')
    results = []
    results1 = []
    for file in files:
        if file and file.filename.endswith('.wav'):
            file_path = os.path.join('/tmp', file.filename)
            file.save(file_path)

            try:
                transcription = transcribe_audio(file_path)
                sentiment = analyze_sentiment(transcription)
                print(sentiment)
                result_dict = {
                    'file_name': file.filename,
                    'transcription': transcription,
                    'sentiment': sentiment[0]  # Assuming the first result is the most relevant
                }
                result_dict1 = {
                    'file_name': file.filename,
                    'transcription': transcription,
                    'sentiment': sentiment[0]['label'],
                    #   'score':   sentiment[0].score# Assuming the first result is the most relevant
                    # 'sentiment' : sentiment.label,
                    'score' : sentiment[0]['score']
                }
                results.append(result_dict)
                results1.append(result_dict1)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:
            return jsonify({'error': 'Invalid file type, only .wav files are accepted'}), 400
    df_all = pd.DataFrame(results1)
    df_all.to_csv('transcription.csv')
    csv_file_path = os.path.join('static', 'transcription.csv')
    df_all.to_csv(csv_file_path, index=True)

    return jsonify({'results': results, 'csv_link': f'http://localhost:5000/static/transcription.csv'}), 200
    return jsonify({'results': results}), 200
if __name__ == "__main__":
    app.run(debug=True)
