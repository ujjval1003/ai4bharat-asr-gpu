"""
IndicConformer Web UI — Gradio v8.1
=====================================
Browser interface for all 22 Indian languages.
Live transcription • Committed history • Export .txt

Fixes over v8:
  - History now commits reliably using VAD silence detection (same logic as v7.2)
  - Stop() commits remaining live_text before clearing — no race condition
  - Export works correctly (history is always populated after speaking)
  - Separate speech_buf (VAD utterance) from rolling_buf (live partial)
  - SILENCE_RMS_THRESH calibrated at startup

Install:
    pip install nemo_toolkit[asr] sounddevice soundfile numpy torch gradio

Run:
    python indic_web_asr.py
    Then open: http://127.0.0.1:7860
"""

import os
import torch
import numpy as np
import sounddevice as sd
import queue
import threading
import time
import tempfile
import soundfile as sf
import nemo.collections.asr as nemo_asr
import gradio as gr
import logging
from datetime import datetime
import warnings

# ── Suppress spam ──────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="pin_memory")
warnings.filterwarnings("ignore", message="amp")
warnings.filterwarnings("ignore", message="custom_fwd")
warnings.filterwarnings("ignore", message="custom_bwd")
warnings.filterwarnings("ignore", message="_register_pytree_node")
logging.getLogger("nemo").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

torch.set_flush_denormal(True)
torch.set_num_threads(os.cpu_count())

# ====================== 22 LANGUAGES ======================
LANGUAGES = {
    "Assamese":  ("as",  "ai4bharat/indicconformer_stt_as_hybrid_rnnt_large"),
    "Bengali":   ("bn",  "ai4bharat/indicconformer_stt_bn_hybrid_rnnt_large"),
    "Bodo":      ("brx", "ai4bharat/indicconformer_stt_brx_hybrid_rnnt_large"),
    "Dogri":     ("doi", "ai4bharat/indicconformer_stt_doi_hybrid_rnnt_large"),
    "Gujarati":  ("gu",  "ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large"),
    "Hindi":     ("hi",  "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large"),
    "Kannada":   ("kn",  "ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large"),
    "Konkani":   ("kok", "ai4bharat/indicconformer_stt_kok_hybrid_rnnt_large"),
    "Kashmiri":  ("ks",  "ai4bharat/indicconformer_stt_ks_hybrid_rnnt_large"),
    "Maithili":  ("mai", "ai4bharat/indicconformer_stt_mai_hybrid_rnnt_large"),
    "Malayalam": ("ml",  "ai4bharat/indicconformer_stt_ml_hybrid_rnnt_large"),
    "Manipuri":  ("mni", "ai4bharat/indicconformer_stt_mni_hybrid_rnnt_large"),
    "Marathi":   ("mr",  "ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large"),
    "Nepali":    ("ne",  "ai4bharat/indicconformer_stt_ne_hybrid_rnnt_large"),
    "Odia":      ("or",  "ai4bharat/indicconformer_stt_or_hybrid_rnnt_large"),
    "Punjabi":   ("pa",  "ai4bharat/indicconformer_stt_pa_hybrid_rnnt_large"),
    "Sanskrit":  ("sa",  "ai4bharat/indicconformer_stt_sa_hybrid_rnnt_large"),
    "Santali":   ("sat", "ai4bharat/indicconformer_stt_sat_hybrid_rnnt_large"),
    "Sindhi":    ("sd",  "ai4bharat/indicconformer_stt_sd_hybrid_rnnt_large"),
    "Tamil":     ("ta",  "ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large"),
    "Telugu":    ("te",  "ai4bharat/indicconformer_stt_te_hybrid_rnnt_large"),
    "Urdu":      ("ur",  "ai4bharat/indicconformer_stt_ur_hybrid_rnnt_large"),
}

# ====================== CONFIG ======================
SAMPLE_RATE          = 16000
CHUNK_SAMPLES        = int(SAMPLE_RATE * 0.1)   # 100 ms chunks
MAX_BUFFER_SEC       = 12
MIN_SPEECH_SEC       = 0.4
PARTIAL_INTERVAL_GPU = 1.25   # seconds between live partial decodes on GPU
PARTIAL_INTERVAL_CPU = 2.0    # seconds between live partial decodes on CPU
SILENCE_RMS_THRESH   = 0.012  # calibrated at startup
SILENCE_CHUNKS       = 32     # 32 × 100 ms = 3.2 s silence → commit utterance


# ====================== TRANSCRIPTION HELPER ======================

def _transcribe(model, lang_id: str, decoder: str, audio: np.ndarray) -> str:
    if len(audio) / SAMPLE_RATE < MIN_SPEECH_SEC:
        return ""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, audio, SAMPLE_RATE, subtype="FLOAT")
    try:
        model.cur_decoder = decoder
        with torch.inference_mode():
            results = model.transcribe(
                [tmp.name],
                batch_size=1,
                language_id=lang_id,
                return_hypotheses=False,
                logprobs=False,
                verbose=False,
            )
        r = results[0]
        return (r.text if hasattr(r, "text") else str(r)).strip()
    except Exception as e:
        print(f"[transcribe error] {e}")
        return ""
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ====================== ASR ENGINE ======================

class WebASR:
    """
    Dual-buffer decode loop — same proven architecture as v7.2 terminal version:

    rolling_buf  → capped MAX_BUFFER_SEC rolling window
                   → decoded every partial_interval for LIVE TEXT display
    speech_buf   → accumulates only while voice is active (VAD)
                   → decoded + COMMITTED TO HISTORY on 3.2 s of silence
                   → reset after each commit (clean utterance segments)

    This is exactly how v7.1/v7.2 worked, now ported to the web UI.
    """

    def __init__(self):
        self.model        = None
        self.lang_id      = None
        self.lang_name    = None
        self.decoder      = None
        self.device       = None
        self.partial_interval = PARTIAL_INTERVAL_CPU

        self.audio_q      = queue.Queue()
        self.running      = False
        self._decode_thread = None
        self._sd_stream   = None
        self._lock        = threading.Lock()   # protects history list

        # UI-readable state (written by decode thread, read by Gradio timer)
        self.live_text    = ""
        self.history      = []          # committed utterances as "[HH:MM:SS] text"
        self.status_msg   = "Ready — load a language to begin."

    # ── Model management ──────────────────────────────────────────────────

    def load(self, lang_name: str) -> str:
        if self.running:
            return "⚠ Stop listening before switching languages."
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None

        lang_id, model_id = LANGUAGES[lang_name]
        self.lang_name = lang_name
        self.lang_id   = lang_id
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder   = "rnnt" if self.device.type == "cuda" else "ctc"
        self.partial_interval = (
            PARTIAL_INTERVAL_GPU if self.device.type == "cuda" else PARTIAL_INTERVAL_CPU
        )
        self.status_msg = f"Loading {lang_name}... (first run downloads ~1.8 GB)"

        try:
            model = nemo_asr.models.ASRModel.from_pretrained(model_id)
            model.freeze()
            self.model = model.to(self.device)
            self.status_msg = (
                f"✅ {lang_name} ready on {str(self.device).upper()} "
                f"| {self.decoder.upper()} decoder | Click 'Start Listening'"
            )
        except Exception as e:
            self.status_msg = f"❌ Load failed: {e}"
        return self.status_msg

    # ── Stream control ────────────────────────────────────────────────────

    def start(self) -> str:
        if self.model is None:
            return "⚠ Load a language first."
        if self.running:
            return "Already listening."
        self.running   = True
        self.live_text = ""
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

        self._decode_thread = threading.Thread(
            target=self._decode_loop, daemon=True
        )
        self._decode_thread.start()

        self._sd_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=self._audio_callback,
        )
        self._sd_stream.start()
        self.status_msg = "🎤 Listening... speak now."
        return self.status_msg

    def stop(self) -> str:
        self.running = False

        # Stop audio first so no new chunks arrive
        if self._sd_stream:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None

        # Wait for decode thread to finish its current inference
        if self._decode_thread:
            self._decode_thread.join(timeout=5)
            self._decode_thread = None

        # Commit any remaining live text to history
        if self.live_text.strip():
            self._commit(self.live_text)
            self.live_text = ""

        self.status_msg = "⏹ Stopped. See committed transcript below."
        return self.status_msg

    def clear(self):
        with self._lock:
            self.history   = []
        self.live_text = ""

    def export(self):
        with self._lock:
            lines = list(self.history)
        if not lines:
            return None
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"transcript_{(self.lang_name or 'indic').lower()}_{ts}.txt"
        path = os.path.join(tempfile.gettempdir(), name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    # ── Internal helpers ──────────────────────────────────────────────────

    def _commit(self, text: str):
        """Append a finalised utterance to history with timestamp."""
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self.history.append(f"[{ts}]  {text}")

    def _audio_callback(self, indata, frames, time_info, status):
        self.audio_q.put(indata[:, 0].copy())

    def get_history_str(self) -> str:
        with self._lock:
            return "\n".join(self.history)

    # ── Decode loop (background thread) ───────────────────────────────────

    def _decode_loop(self):
        """
        Dual-buffer strategy — proven in v7.2:

        rolling_buf  → always accumulating, capped, decoded every partial_interval
                        → updates self.live_text (cyan line in UI)

        speech_buf   → accumulates only during detected speech
                        → on SILENCE_CHUNKS consecutive silent chunks:
                           transcribe speech_buf → commit to self.history → reset
                        → this is what v7.1/7.2 used and it worked perfectly
        """
        rolling_buf  = np.array([], dtype=np.float32)
        speech_buf   = np.array([], dtype=np.float32)
        in_speech    = False
        silence_cnt  = 0
        last_decode  = 0.0

        while self.running:
            try:
                chunk = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            now = time.time()

            # ── Rolling buffer for live partial ───────────────────────────
            rolling_buf = np.concatenate([rolling_buf, chunk])
            if len(rolling_buf) > MAX_BUFFER_SEC * SAMPLE_RATE:
                rolling_buf = rolling_buf[-int(MAX_BUFFER_SEC * SAMPLE_RATE):]

            # ── VAD: energy-based speech detection ────────────────────────
            chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))
            is_speech = chunk_rms > SILENCE_RMS_THRESH

            if is_speech:
                in_speech   = True
                silence_cnt = 0
                speech_buf  = np.concatenate([speech_buf, chunk])

                # Cap speech buffer too — prevents runaway long utterances
                if len(speech_buf) / SAMPLE_RATE > MAX_BUFFER_SEC:
                    # Force-commit on very long utterance (like v7.2)
                    text = _transcribe(
                        self.model, self.lang_id, self.decoder, speech_buf
                    )
                    if text:
                        self._commit(text)
                    speech_buf = np.array([], dtype=np.float32)
                    in_speech  = False

            else:
                if in_speech:
                    silence_cnt += 1
                    speech_buf = np.concatenate([speech_buf, chunk])  # keep tail

                    if silence_cnt >= SILENCE_CHUNKS:
                        # ── COMMIT UTTERANCE TO HISTORY ───────────────────
                        text = _transcribe(
                            self.model, self.lang_id, self.decoder, speech_buf
                        )
                        if text:
                            self._commit(text)
                            # Clear live text so UI shows the committed text
                            # clearly rather than leaving stale partial
                            self.live_text = ""

                        # Reset VAD state for next utterance
                        speech_buf  = np.array([], dtype=np.float32)
                        in_speech   = False
                        silence_cnt = 0

            # ── Live partial update (rolling buffer, rate-limited) ────────
            if now - last_decode >= self.partial_interval:
                last_decode = now
                if len(rolling_buf) / SAMPLE_RATE >= MIN_SPEECH_SEC:
                    partial = _transcribe(
                        self.model, self.lang_id, self.decoder, rolling_buf
                    )
                    if partial:
                        self.live_text = partial

            time.sleep(0.01)


# ====================== GLOBAL INSTANCE ======================
asr = WebASR()

# Calibrate silence threshold from ambient noise at startup
try:
    print("Calibrating silence threshold...", end="", flush=True)
    noise = sd.rec(int(SAMPLE_RATE * 1.0), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    SILENCE_RMS_THRESH = max(0.008, min(0.05, noise_rms * 3.0))
    print(f" {SILENCE_RMS_THRESH:.4f}")
except Exception:
    print(" using default")


# ====================== GRADIO UI ======================

CSS = """
#live-box textarea {
    font-size: 1.15em;
    color: #0ea5e9;
    font-weight: 600;
    min-height: 80px;
}
#history-box textarea {
    font-size: 0.95em;
    min-height: 220px;
}
"""

with gr.Blocks(title="IndicConformer ASR", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.Markdown(
        "# 🇮🇳 IndicConformer Real-Time ASR\n"
        "Google Voice Typing for all 22 Indian languages — AI4Bharat"
    )

    # ── Language + Load ───────────────────────────────────────────────────
    with gr.Row():
        lang_dd   = gr.Dropdown(
            choices=list(LANGUAGES.keys()), value="Gujarati",
            label="Language", scale=3
        )
        load_btn  = gr.Button("⬇ Load Model", variant="primary", scale=1)

    status_box = gr.Textbox(
        value=asr.status_msg, label="Status",
        interactive=False, lines=1
    )

    # ── Controls ──────────────────────────────────────────────────────────
    with gr.Row():
        start_btn  = gr.Button("🎤 Start Listening", variant="primary",   scale=2)
        stop_btn   = gr.Button("⏹ Stop",             variant="stop",      scale=1)
        clear_btn  = gr.Button("🗑 Clear",            variant="secondary", scale=1)
        export_btn = gr.Button("💾 Export .txt",      variant="secondary", scale=1)

    # ── Live text ─────────────────────────────────────────────────────────
    live_box = gr.Textbox(
        label="🎙 Live (updates every 0.5 s while speaking)",
        value="", interactive=False, lines=3, elem_id="live-box"
    )

    # ── Committed history ─────────────────────────────────────────────────
    history_box = gr.Textbox(
        label="📜 Committed Transcript (saved after each pause)",
        value="", interactive=False, lines=12, elem_id="history-box"
    )

    export_file = gr.File(label="Download transcript", visible=False)

    # ── Event handlers ────────────────────────────────────────────────────

    def do_load(lang):
        return asr.load(lang)

    def do_start():
        return asr.start()

    def do_stop():
        msg = asr.stop()
        return msg, asr.live_text, asr.get_history_str()

    def do_clear():
        asr.clear()
        return "", ""

    def do_export():
        path = asr.export()
        if path:
            return gr.File(value=path, visible=True)
        return gr.File(visible=False)

    def poll():
        """Called every 0.5 s by gr.Timer — returns live partial + full history."""
        return asr.live_text, asr.get_history_str()

    load_btn.click(do_load,  inputs=lang_dd, outputs=status_box)
    start_btn.click(do_start, outputs=status_box)
    stop_btn.click(do_stop,   outputs=[status_box, live_box, history_box])
    clear_btn.click(do_clear, outputs=[live_box, history_box])
    export_btn.click(do_export, outputs=export_file)

    # gr.Timer polls every 0.5 s — correct Gradio 4.x pattern
    timer = gr.Timer(value=0.5)
    timer.tick(poll, outputs=[live_box, history_box])


# ====================== ENTRY POINT ======================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,      # set True for a public gradio.live link
        show_error=True,
    )