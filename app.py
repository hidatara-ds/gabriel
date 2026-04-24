"""
GABRIEL — Cloud Voice API
Dead simple: receive audio → Gemini → return text
"""

import os
import struct
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

# ── Config ──────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "")
LANGUAGE = os.environ.get("LANGUAGE", "id")
MODEL = "gemini-2.0-flash"

# ── GenAI Client ────────────────────────────────────────────
client = None
if API_KEY:
    client = genai.Client(api_key=API_KEY)
    print(f"[OK] Gemini connected with API key, model={MODEL}")
else:
    print("[ERROR] GEMINI_API_KEY not set!")

# ── App ─────────────────────────────────────────────────────
app = FastAPI(title="Gabriel API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def wav_header(data_len: int) -> bytes:
    """Create a WAV header for 16kHz 16-bit mono PCM."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_len, b"WAVE", b"fmt ", 16,
        1, 1, 16000, 32000, 2, 16, b"data", data_len,
    )


def ask_gemini_audio(audio_bytes: bytes, mime: str) -> str:
    """Send audio to Gemini, get text reply."""
    prompt = (
        "Listen to this audio and reply to the user. "
        "Keep it short (max 12 words), engaging, casual English."
        if LANGUAGE == "en" else
        "Dengarkan suara ini dan beri balasan langsung ke user. "
        "Maksimal 12 kata, santai, bahasa Indonesia."
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=audio_bytes, mime_type=mime),
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=80,
            temperature=0.9,
        ),
    )
    return (response.text or "").strip() or "Maaf, tidak bisa memproses."


def ask_gemini_text(prompt: str) -> str:
    """Send text to Gemini, get text reply."""
    system = (
        "You are Gabriel, a friendly AI on a tiny OLED screen. "
        "Keep responses under 15 words."
        if LANGUAGE == "en" else
        "Kamu Gabriel, AI ramah di layar OLED kecil. Jawab maks 15 kata."
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=[f"{system}\n\nUser: {prompt}\nGabriel:"],
        config=types.GenerateContentConfig(
            max_output_tokens=80,
            temperature=0.9,
        ),
    )
    return (response.text or "").strip() or "Maaf, tidak bisa menjawab."


# ═════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════

@app.get("/")
async def health():
    return {
        "status": "online",
        "model": MODEL,
        "api_key_set": bool(API_KEY),
        "language": LANGUAGE,
    }


@app.post("/generate")
async def generate(request: Request):
    """Text prompt → AI response."""
    if not client:
        raise HTTPException(503, "GEMINI_API_KEY not set")

    body = await request.json()
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "Missing 'prompt'")

    try:
        text = ask_gemini_text(prompt)
        return {
            "message": text,
            "category": "chat",
            "emoji": "💬",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "gemini",
        }
    except Exception as e:
        print(f"[ERROR] generate: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/chat/audio")
async def audio_chat(request: Request):
    """Raw PCM from ESP32 → WAV → Gemini → text response."""
    if not client:
        raise HTTPException(503, "GEMINI_API_KEY not set")

    pcm = await request.body()
    if not pcm or len(pcm) < 100:
        raise HTTPException(400, "No audio data")

    print(f"[AUDIO] Received {len(pcm)} bytes PCM")

    try:
        wav = wav_header(len(pcm)) + pcm
        start = time.time()
        text = ask_gemini_audio(wav, "audio/wav")
        ms = int((time.time() - start) * 1000)
        print(f'[OK] Audio response ({ms}ms): "{text}"')

        return {
            "message": text,
            "category": "chat",
            "emoji": "🗣️",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "gemini",
        }
    except Exception as e:
        print(f"[ERROR] audio: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/chat/audio/web")
async def audio_chat_web(request: Request):
    """WebM audio from browser → Gemini → text response."""
    if not client:
        raise HTTPException(503, "GEMINI_API_KEY not set")

    audio = await request.body()
    if not audio or len(audio) < 100:
        raise HTTPException(400, "No audio data")

    mime = request.headers.get("content-type", "audio/webm")
    print(f"[WEB AUDIO] Received {len(audio)} bytes ({mime})")

    try:
        start = time.time()
        text = ask_gemini_audio(audio, mime)
        ms = int((time.time() - start) * 1000)
        print(f'[OK] Web audio response ({ms}ms): "{text}"')

        return {
            "message": text,
            "category": "chat",
            "emoji": "🗣️",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "gemini",
        }
    except Exception as e:
        print(f"[ERROR] web audio: {e}")
        raise HTTPException(500, str(e))


@app.get("/web-test", response_class=HTMLResponse)
async def web_test():
    """Test page — record audio or type text."""
    return """<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gabriel Test</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui;background:#111;color:#eee;padding:2rem;max-width:600px;margin:0 auto}
h1{margin-bottom:.5rem;color:#a78bfa}
.sub{color:#666;margin-bottom:2rem;font-size:.9rem}
button{padding:.6rem 1.2rem;border:none;border-radius:8px;font-weight:600;cursor:pointer;margin:.3rem}
button:disabled{opacity:.3;cursor:not-allowed}
.rec{background:#ef4444;color:#fff}
.stop{background:#f59e0b;color:#000}
.send{background:#6366f1;color:#fff}
.text-send{background:#10b981;color:#fff}
input{width:100%;padding:.6rem;border:1px solid #333;border-radius:8px;background:#1a1a2e;color:#eee;margin:.5rem 0}
#status{margin:1rem 0;font-weight:600;color:#a78bfa}
#out{background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:1rem;white-space:pre-wrap;font-family:monospace;font-size:.85rem;color:#c4b5fd;min-height:60px}
hr{border:none;border-top:1px solid #333;margin:1.5rem 0}
</style></head>
<body>
<h1>Gabriel Test</h1>
<p class="sub">Audio + Text test</p>
<div>
<button id="rec" class="rec">Record</button>
<button id="stop" class="stop" disabled>Stop</button>
<button id="send" class="send" disabled>Send Audio</button>
</div>
<hr>
<input id="txt" placeholder="Or type a message...">
<button id="tsend" class="text-send">Send Text</button>
<div id="status">Ready</div>
<div id="out">-</div>
<script>
let mr,chunks=[],blob;
rec.onclick=async()=>{
  try{
    const s=await navigator.mediaDevices.getUserMedia({audio:true});
    mr=new MediaRecorder(s,{mimeType:"audio/webm"});
    chunks=[];blob=null;
    mr.ondataavailable=e=>{if(e.data.size>0)chunks.push(e.data)};
    mr.onstop=()=>{blob=new Blob(chunks,{type:"audio/webm"});status.textContent="Recorded. Send it.";send.disabled=false;s.getTracks().forEach(t=>t.stop())};
    mr.start();status.textContent="Recording...";rec.disabled=true;stop.disabled=false;send.disabled=true;
  }catch(e){status.textContent="Mic error: "+e}
};
stop.onclick=()=>{if(mr&&mr.state==="recording"){mr.stop();rec.disabled=false;stop.disabled=true}};
send.onclick=async()=>{
  if(!blob)return;
  status.textContent="Sending...";out.textContent="...";
  try{
    const r=await fetch("/api/chat/audio/web",{method:"POST",headers:{"Content-Type":"audio/webm"},body:await blob.arrayBuffer()});
    const d=await r.json();
    status.textContent=r.ok?"Done":"Error "+r.status;
    out.textContent=JSON.stringify(d,null,2);
  }catch(e){status.textContent="Failed";out.textContent=e+""}
};
tsend.onclick=async()=>{
  const p=txt.value.trim();if(!p)return;
  status.textContent="Sending...";out.textContent="...";
  try{
    const r=await fetch("/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:p})});
    const d=await r.json();
    status.textContent=r.ok?"Done":"Error "+r.status;
    out.textContent=JSON.stringify(d,null,2);
  }catch(e){status.textContent="Failed";out.textContent=e+""}
};
txt.addEventListener("keydown",e=>{if(e.key==="Enter")tsend.click()});
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
