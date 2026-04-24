#app.py
# 
# Copyright © 2025. All Rights Reserved.
# 
# PROPRIETARY AND CONFIDENTIAL
# This software is the proprietary information of the copyright holder.
# Unauthorized copying, distribution, or use is strictly prohibited.
# See LICENSE file in the root directory for terms and conditions.
#
import os
import base64
import json
from flask import Flask, request, jsonify, Response
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from gtts import gTTS
import sounddevice as sd
import numpy as np
from google.cloud import speech
import requests
import tempfile
import wave
from google.cloud import texttospeech
import uuid
import sqlite3
from datetime import datetime

app = Flask(__name__)
LATEST_INPUT_WAV = None
LATEST_INPUT_META = {}
AUDIO_CACHE = {}
AUDIO_CACHE_MAX_ITEMS = 20

# Speech cleanup tuning (override via Cloud Run env vars if needed)
AUDIO_HP_CUTOFF_HZ = int(os.environ.get("AUDIO_HP_CUTOFF_HZ", "220"))
AUDIO_LP_CUTOFF_HZ = int(os.environ.get("AUDIO_LP_CUTOFF_HZ", "3400"))
AUDIO_TARGET_DBFS = float(os.environ.get("AUDIO_TARGET_DBFS", "-24.0"))
AUDIO_MAX_GAIN_DB = float(os.environ.get("AUDIO_MAX_GAIN_DB", "10.0"))

# Inisialisasi Vertex AI
#key_path = os.path.join(os.path.dirname(__file__), "key.json")
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

# Get project ID and location from environment variables for security
YOUR_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", os.environ.get("GCP_PROJECT_ID"))
YOUR_VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "us-central1")

if not YOUR_PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT_ID or GCP_PROJECT_ID environment variable must be set")

vertexai.init(project=YOUR_PROJECT_ID, location=YOUR_VERTEX_AI_LOCATION)
model = GenerativeModel("gemini-2.5-flash")

# In-memory session store.
# WARNING: This is not suitable for production on Cloud Run with multiple
# instances, as each instance will have its own memory. Use a shared
# database like Redis or Firestore for session management in production.
SESSIONS = {}

DB_PATH = os.path.join(os.path.dirname(__file__), 'session_history.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_message(session_id, role, message):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO chat_history (session_id, role, message, timestamp) VALUES (?, ?, ?, ?)',
              (session_id, role, message, datetime.now()))
    conn.commit()
    conn.close()

def get_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT role, message FROM chat_history WHERE session_id = ? ORDER BY id ASC', (session_id,))
    rows = c.fetchall()
    conn.close()
    # Format sesuai dengan yang diharapkan VertexAI: [{'role': 'user', 'parts': [msg]}, ...]
    return [{'role': row[0], 'parts': [row[1]]} for row in rows]

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Voice Discussion</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #micBtn {
                background: #f2f2f2;
                border: none;
                border-radius: 50%;
                width: 80px;
                height: 80px;
                font-size: 40px;
                cursor: pointer;
                box-shadow: 0 2px 8px #aaa;
                transition: background 0.2s;
                margin-top: 40px;
            }
            #micBtn.recording {
                background: #ff5252;
                color: #fff;
            }
            #response { margin-top: 40px; }
            #log { color: #888; font-size: 12px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>AI Voice Discussion</h1>
        <button id="micBtn" title="Tap to record"><span id="micIcon">🎤</span></button>
        <div id="response"></div>
        <div id="log"></div>
        <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        const micBtn = document.getElementById('micBtn');
        const micIcon = document.getElementById('micIcon');
        const responseDiv = document.getElementById('response');
        const logDiv = document.getElementById('log');

        function log(msg) {
            console.log('[AI DISCUSSION]', msg);
            logDiv.innerHTML += msg + '<br>';
        }

        async function startRecording() {
            log('Start recording...');
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                alert('Browser tidak mendukung perekaman audio.');
                log('getUserMedia not supported');
                return;
            }
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                log('Microphone access granted');
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) audioChunks.push(event.data);
                    log('Data available, size: ' + event.data.size);
                };
                mediaRecorder.onstop = async () => {
                    log('Recording stopped');
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    log('Audio blob created, size: ' + audioBlob.size);
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = async () => {
                        const base64Audio = reader.result.split(',')[1];
                        log('Audio converted to base64, length: ' + base64Audio.length);
                        responseDiv.innerHTML = '<p>Processing...</p>';
                        try {
                            log('Sending audio to backend...');
                            const response = await fetch('/api/process-audio', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ audio: base64Audio })
                            });
                            log('Response received from backend');
                            const result = await response.json();
                            log('Response JSON: ' + JSON.stringify(result));
                            if (result.answer && result.audio_base64) {
                                responseDiv.innerHTML =
                                    `<p><b>Pertanyaan:</b> ${result.question}</p>` +
                                    `<p><b>Jawaban:</b> ${result.answer}</p>` +
                                    `<audio controls src="data:audio/mp3;base64,${result.audio_base64}"></audio>`;
                                log('Answer and audio displayed');
                            } else {
                                responseDiv.innerHTML = `<p style='color:red;'>${result.error || 'Terjadi kesalahan.'}</p>`;
                                log('Error from backend: ' + (result.error || 'Unknown error'));
                            }
                        } catch (err) {
                            responseDiv.innerHTML = `<p style='color:red;'>Gagal mengirim audio: ${err}</p>`;
                            log('Fetch error: ' + err);
                        }
                    };
                };
                mediaRecorder.start();
                isRecording = true;
                micBtn.classList.add('recording');
                micIcon.textContent = '⏹️';
                log('MediaRecorder started');
            } catch (err) {
                alert('Tidak bisa mengakses mikrofon: ' + err);
                log('Microphone access error: ' + err);
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                micBtn.classList.remove('recording');
                micIcon.textContent = '🎤';
                log('Stop recording requested');
            }
        }

        micBtn.onclick = () => {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
        };
        </script>
    </body>
    </html>
    '''

# Endpoint untuk menerima audio dari robot
from pydub import AudioSegment

@app.route('/api/process-audio', methods=['POST'])
def api_process_audio():
    global LATEST_INPUT_WAV, LATEST_INPUT_META
    input_path = None
    converted_path = None
    diagnostics = {
        'content_type': request.content_type,
        'content_length': request.content_length,
        'is_json': request.is_json,
        'user_agent': request.user_agent.string if request.user_agent else None,
    }
    try:
        print('[API] Request received:', diagnostics)
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type harus application/json',
                'stage': 'validate_content_type',
                'diagnostics': diagnostics
            }), 400

        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({
                'error': 'Body JSON tidak valid',
                'stage': 'parse_json',
                'diagnostics': diagnostics
            }), 400

        audio_data = data.get('audio')
        session_id = data.get('session_id')
        diagnostics['audio_b64_len'] = len(audio_data) if isinstance(audio_data, str) else 0

        if not audio_data:
            return jsonify({
                'error': 'Audio data tidak ditemukan',
                'stage': 'validate_audio_field',
                'diagnostics': diagnostics
            }), 400

        if not session_id:
            session_id = uuid.uuid4().hex
            print(f'[API] New session created: {session_id}')
        diagnostics['session_id'] = session_id

        # Ambil history dari database
        history = get_history(session_id)
        chat = model.start_chat(history=history)
        diagnostics['history_len'] = len(history)

        # Simpan file sementara
        try:
            audio_bytes = base64.b64decode(audio_data, validate=True)
        except Exception as e:
            return jsonify({
                'error': f'Audio base64 tidak valid: {str(e)}',
                'stage': 'decode_base64',
                'diagnostics': diagnostics
            }), 400
        diagnostics['audio_bytes_len'] = len(audio_bytes)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_input:
            temp_input.write(audio_bytes)
            input_path = temp_input.name

        # Konversi ke WAV 16kHz mono dengan pydub
        sound = AudioSegment.from_file(input_path)
        diagnostics['decoded_ms'] = len(sound)
        diagnostics['decoded_channels'] = sound.channels
        diagnostics['decoded_frame_rate'] = sound.frame_rate
        diagnostics['decoded_sample_width'] = sound.sample_width
        diagnostics['decoded_dbfs'] = float(sound.dBFS) if sound.dBFS != float("-inf") else -999.0
        diagnostics['decoded_rms'] = int(sound.rms)

        sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        if AUDIO_HP_CUTOFF_HZ > 0:
            sound = sound.high_pass_filter(AUDIO_HP_CUTOFF_HZ)
            # Second pass to reduce low-end rumble from noisy power/ground.
            sound = sound.high_pass_filter(AUDIO_HP_CUTOFF_HZ)
        if AUDIO_LP_CUTOFF_HZ > 0:
            sound = sound.low_pass_filter(AUDIO_LP_CUTOFF_HZ)
        diagnostics['hp_cutoff_hz'] = AUDIO_HP_CUTOFF_HZ
        diagnostics['lp_cutoff_hz'] = AUDIO_LP_CUTOFF_HZ

        # ESP32 mic input is often quiet; apply controlled gain before STT.
        if sound.rms > 0:
            gain_db = AUDIO_TARGET_DBFS - sound.dBFS
            if gain_db > AUDIO_MAX_GAIN_DB:
                gain_db = AUDIO_MAX_GAIN_DB
            if gain_db > 1.0:
                sound = sound.apply_gain(gain_db)
            diagnostics['applied_gain_db'] = round(float(gain_db), 2)
        else:
            diagnostics['applied_gain_db'] = 0.0

        diagnostics['post_gain_dbfs'] = float(sound.dBFS) if sound.dBFS != float("-inf") else -999.0
        diagnostics['post_gain_rms'] = int(sound.rms)
        temp_converted = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sound.export(temp_converted.name, format="wav")
        converted_path = temp_converted.name
        print('[API] Audio converted to 16kHz mono WAV')
        with open(converted_path, 'rb') as f:
            converted_bytes = f.read()
        diagnostics['converted_bytes_len'] = len(converted_bytes)
        diagnostics['converted_ms'] = len(sound)
        LATEST_INPUT_WAV = converted_bytes
        LATEST_INPUT_META = {
            'session_id': session_id,
            'bytes': len(converted_bytes),
            'duration_ms': len(sound),
            'sample_rate': 16000,
            'channels': 1,
            'sample_width': 2,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }

        # Kirim ke Google STT
        speech_client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=converted_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="id-ID",
            enable_automatic_punctuation=True
        )
        print('[API] Sending to Google STT...')
        response = speech_client.recognize(config=config, audio=audio)
        diagnostics['stt_result_count'] = len(response.results)

        question = ""
        for result in response.results:
            question += result.alternatives[0].transcript
        question = question.strip()
        diagnostics['stt_question_len'] = len(question)
        print('[API] STT result:', question)
        if not question:
            return jsonify({
                'error': 'Tidak ada teks yang terdeteksi',
                'stage': 'stt_empty_result',
                'session_id': session_id,
                'diagnostics': diagnostics
            }), 400

        # Simpan pertanyaan user ke DB
        save_message(session_id, 'user', question)

        # Gemini
        print(f'[API] Sending message to Gemini for session {session_id}. History length: {len(history)}')
        prompt = (
            u"Jawab pertanyaan berikut secara singkat, jelas, dan tidak lebih dari 3 kalimat. "
            u"Jangan mengulang pertanyaan. "
            u"Pertanyaan: %s" % question
        )
        gemini_response = chat.send_message(prompt)
        answer = gemini_response.text
        print('[API] Gemini answer:', answer)

        # Simpan jawaban ke DB
        save_message(session_id, 'assistant', answer)

        user_agent = (request.user_agent.string or "") if request.user_agent else ""
        is_esp_client = ("ESP32HTTPClient" in user_agent) or (request.headers.get("X-Client", "").lower() == "esp32")

        response_payload = {
            'question': question,
            'answer': answer,
            'session_id': session_id,
            'stage': 'ok',
            'diagnostics': diagnostics
        }

        include_audio_req = data.get('include_audio')
        # Backward compatibility:
        # - browser/web: include audio by default
        # - ESP: default no inline audio
        include_audio = (not is_esp_client) if include_audio_req is None else bool(include_audio_req)
        audio_delivery = (data.get('audio_delivery') or "auto").lower()  # auto | inline | url | none

        if audio_delivery == "auto":
            if not include_audio:
                audio_delivery = "none"
            else:
                audio_delivery = "url" if is_esp_client else "inline"

        diagnostics['include_audio'] = include_audio
        diagnostics['audio_delivery'] = audio_delivery

        if include_audio and audio_delivery in ("inline", "url"):
            synthesize_speech(answer, "temp_response.mp3")
            with open("temp_response.mp3", "rb") as audio_file:
                audio_bytes = audio_file.read()
            os.remove("temp_response.mp3")

            if audio_delivery == "inline":
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                response_payload['audio_base64'] = audio_base64
            elif audio_delivery == "url":
                audio_id = uuid.uuid4().hex
                AUDIO_CACHE[audio_id] = {
                    'bytes': audio_bytes,
                    'mime': 'audio/mpeg',
                    'created_at': datetime.utcnow().isoformat() + 'Z',
                }
                # Simple bounded cache
                while len(AUDIO_CACHE) > AUDIO_CACHE_MAX_ITEMS:
                    oldest_key = next(iter(AUDIO_CACHE))
                    del AUDIO_CACHE[oldest_key]
                response_payload['audio_id'] = audio_id
                response_payload['audio_url'] = f"/api/audio/{audio_id}"
                response_payload['audio_mime'] = "audio/mpeg"

        diagnostics['is_esp_client'] = is_esp_client
        # Keep ESP responses minimal to avoid JSON parsing/memory issues on device.
        if is_esp_client:
            minimal_payload = {
                'question': question,
                'answer': answer,
                'session_id': session_id,
                'stage': 'ok'
            }
            if audio_delivery == "url" and 'audio_url' in response_payload:
                minimal_payload['audio_url'] = response_payload['audio_url']
                minimal_payload['audio_id'] = response_payload.get('audio_id')
                minimal_payload['audio_mime'] = response_payload.get('audio_mime')
            return jsonify(minimal_payload)
        return jsonify(response_payload)

    except Exception as e:
        print('[API] ERROR:', str(e))
        return jsonify({
            'error': str(e),
            'stage': 'internal_error',
            'diagnostics': diagnostics
        }), 500
    finally:
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
        except Exception as cleanup_err:
            print('[API] cleanup input error:', cleanup_err)
        try:
            if converted_path and os.path.exists(converted_path):
                os.remove(converted_path)
        except Exception as cleanup_err:
            print('[API] cleanup converted error:', cleanup_err)

@app.route('/api/audio/<audio_id>', methods=['GET'])
def get_cached_audio(audio_id):
    audio_item = AUDIO_CACHE.get(audio_id)
    if not audio_item:
        return jsonify({'error': 'Audio tidak ditemukan atau kadaluarsa'}), 404
    return Response(
        audio_item['bytes'],
        mimetype=audio_item.get('mime', 'audio/mpeg'),
        headers={'Content-Disposition': f'inline; filename={audio_id}.mp3'}
    )

@app.route('/api/debug/latest-input-meta', methods=['GET'])
def debug_latest_input_meta():
    if not LATEST_INPUT_META:
        return jsonify({'error': 'Belum ada audio input tersimpan'}), 404
    return jsonify({
        'status': 'ok',
        'meta': LATEST_INPUT_META
    })

@app.route('/api/debug/latest-input-wav', methods=['GET'])
def debug_latest_input_wav():
    if not LATEST_INPUT_WAV:
        return jsonify({'error': 'Belum ada audio input tersimpan'}), 404
    return Response(
        LATEST_INPUT_WAV,
        mimetype='audio/wav',
        headers={
            'Content-Disposition': 'inline; filename=latest-input.wav'
        }
    )

# Endpoint untuk testing API
@app.route('/api/test', methods=['POST'])
def test_api():
    try:
        test_text = "Halo, ini adalah tes API."
        # Generate response dari Gemini
        gemini_response = model.generate_content(test_text)
        answer = gemini_response.text

        # Convert ke audio
        synthesize_speech(answer, "temp_test_response.mp3")

        with open("temp_test_response.mp3", "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

        os.remove("temp_test_response.mp3")

        return jsonify({
            'status': 'success',
            'data': {
                'test_text': test_text,
                'answer': answer,
                'audio': audio_base64
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# Fungsi untuk merekam audio dari mic (format: WAV)
def record_audio(duration=5, fs=16000):
    print(f"Rekam suara selama {duration} detik...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio, fs

# Simpan audio ke file WAV sementara
def save_wav(audio, fs):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    with wave.open(temp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    return temp.name

# Konversi file audio ke base64
def audio_to_base64(filepath):
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Kirim audio ke server dan terima jawaban
def send_audio_to_server(audio_base64, server_url):
    payload = {'audio': audio_base64}
    r = requests.post(server_url, json=payload)
    r.raise_for_status()
    return r.json()

# Main loop
def main():
    SERVER_URL = 'http://localhost:5000/api/process-audio'  # Ganti jika server beda
    while True:
        input("Tekan ENTER untuk mulai rekam (atau Ctrl+C untuk keluar)...")
        audio, fs = record_audio(duration=5)
        wav_path = save_wav(audio, fs)
        audio_b64 = audio_to_base64(wav_path)
        print("Mengirim audio ke server...")
        try:
            response = send_audio_to_server(audio_b64, SERVER_URL)
            print("Jawaban dari AI:", response.get('answer'))
            # Jika ingin Pepper mengucapkan:
            # pepper_say(response.get('answer'))
            # Jika ingin play audio TTS:
            if response.get('audio_base64'):
                tts_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
                with open(tts_path, 'wb') as f:
                    f.write(base64.b64decode(response['audio_base64']))
                print(f"Audio jawaban disimpan di: {tts_path}")
                # Play audio jika ingin (gunakan mpg123/vlc atau library lain)
        except Exception as e:
            print("Gagal:", e)

def synthesize_speech(text, output_path):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    # Pilih voice yang natural
    voice = texttospeech.VoiceSelectionParams(
        language_code="id-ID",
        name="id-ID-Wavenet-A",  # Coba juga id-ID-Wavenet-B, dst
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0
    )
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    with open(output_path, "wb") as out:
        out.write(response.audio_content)

if __name__ == '__main__':
    if os.environ.get("RUN_LOCAL", "false").lower() == "true":
        main()  # jalankan hanya saat lokal
    else:
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
