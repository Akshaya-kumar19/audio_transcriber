import whisper_timestamped as whisper
import json
from flask import Flask, request, render_template, send_file
from io import BytesIO

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_path = request.form['audio_path']
    
    # Load the audio file
    try:
        audio = whisper.load_audio(audio_path)
    except Exception as e:
        return f"Error loading audio: {e}", 400

    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the audio
    try:
        result = whisper.transcribe(model, audio, language="fr")
    except Exception as e:
        return f"Error transcribing audio: {e}", 500

    # Convert the result to a JSON string with indentation and non-ASCII characters
    result_json = json.dumps(result, indent=2, ensure_ascii=False)

    # Prepare the file for download
    result_txt = BytesIO()
    result_txt.write(result_json.encode('utf-8'))
    result_txt.seek(0)

    # Send the file as a download
    return send_file(result_txt, as_attachment=True, download_name="transcription.txt", mimetype="text/plain")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
