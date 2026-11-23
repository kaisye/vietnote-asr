"""Live captioning demo (the one at https://asr.vietnote.vn): the browser streams microphone audio over a
WebSocket, this server runs the streaming transducer (sherpa-onnx) and sends back partial and final text. Final
segments go through ITN (numbers to digits), the punctuation/casing restorer and the hotword list.

    python demo/server.py --model-dir models/streaming/fp32 --punct-dir models/punct \
        --vad models/silero_vad.onnx --hotwords data/hotwords.txt --bpe-model models/bpe.model
    then open http://localhost:8765

Protocol: the client sends binary messages of float32 PCM at 16 kHz mono. The server sends JSON:
  {"type": "partial", "text": ...}                              while a segment is in progress
  {"type": "final", "text": ..., "raw": ..., "start": s, "end": s, "punct_ms": ms}
  {"type": "stats", "audio_s": ..., "compute_s": ..., "chunk_ms": ...}
"""

import argparse
import asyncio
import concurrent.futures
import json
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnasr import hotwords as hotwords_mod  # noqa: E402
from vnasr.itn import _NUMBER_WORDS, spoken_to_digits  # noqa: E402

# A segment starts where the previous one ended, so its length includes the pause before it. A number segment that
# short right after another number segment is the same reading split by pauses ("086" ... "222" ... "3252").
MERGE_MAX_S = 5.0

ROOT = Path(__file__).resolve().parent


def make_recognizer(model_dir: Path, threads: int, method: str, hot: dict | None = None):
    import sherpa_onnx

    def pick(prefix):
        files = sorted(model_dir.glob(f"{prefix}*.onnx"))
        int8 = [f for f in files if ".int8." in f.name]
        return str((int8 or files)[0])

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        **(hot or {}), encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"),
        tokens=str(model_dir / "tokens.txt"), num_threads=threads, sample_rate=16000, feature_dim=80,
        decoding_method=method, max_active_paths=4, enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.0, rule2_min_trailing_silence=0.8, rule3_min_utterance_length=15)


class TTS:
    """Optional voice-swap TTS worker (tts_worker.py, not included); off unless --tts-python and --tts-model are given."""

    def __init__(self, python: str, model: str):
        self.dir = Path(tempfile.mkdtemp(prefix="tts-"))
        self.proc = subprocess.Popen([python, str(ROOT / "tts_worker.py"), "--model", model],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        head = json.loads(self.proc.stdout.readline())
        self.voices = head["voices"]
        self.lock = threading.Lock()

    def __call__(self, text: str, voice: str):
        name = f"{uuid.uuid4().hex}.wav"
        with self.lock:
            self.proc.stdin.write(json.dumps({"text": text, "voice": voice, "out": str(self.dir / name)}) + "\n")
            self.proc.stdin.flush()
            res = json.loads(self.proc.stdout.readline())
        res["name"] = name
        return res


class SpeechGate:
    """Silero VAD in front of the recogniser: frames without speech are replaced by zeros, so
    room noise boosted by the browser's gain control is not turned into words. Frames are delayed
    by `lookahead` so the VAD (which reacts late) does not cut the first syllable, and speech is
    kept for `hangover` frames after the VAD says it ended."""

    def __init__(self, model: str, lookahead: int = 3, hangover: int = 5, min_db: float = -35.0):
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.25
        cfg.silero_vad.min_speech_duration = 0.1
        cfg.sample_rate = 16000
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
        self.frames, self.flags, self.levels = [], [], []
        self.lookahead, self.hangover = lookahead, hangover
        self.min_db = min_db  # near-field gate: quieter speech (other people, a TV) is ignored
        self.last_speech = 10**9  # frames since speech was last seen; starts as "long ago"

    def push(self, audio: np.ndarray) -> list:
        self.vad.accept_waveform(audio)
        while not self.vad.empty():
            self.vad.pop()
        self.frames.append(audio)
        self.flags.append(self.vad.is_speech_detected())
        self.levels.append(20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-9))
        out = []
        while len(self.frames) > self.lookahead:
            w = self.lookahead + 1
            if any(self.flags[:w]) and max(self.levels[:w]) >= self.min_db:
                self.last_speech = 0
            keep = self.last_speech <= self.hangover
            frame = self.frames.pop(0)
            self.flags.pop(0)
            self.levels.pop(0)
            self.last_speech += 1
            out.append(frame if keep else np.zeros_like(frame))
        return out

    def flush(self) -> list:
        out = self.frames if any(self.flags) or self.last_speech <= self.hangover else \
            [np.zeros_like(f) for f in self.frames]
        self.frames, self.flags, self.levels = [], [], []
        return out


class Speakers:
    """Speaker enrolment and identification with a sherpa-onnx speaker embedding model (CAM++).
    Profiles are unit vectors averaged over enrolment clips, kept in memory and in a local JSON file."""

    def __init__(self, model: str, store: Path, threshold: float = 0.5):
        import sherpa_onnx
        self.ext = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=2))
        self.store, self.threshold = store, threshold
        self.profiles = {}
        if store.exists():
            self.profiles = {k: np.array(v) for k, v in json.loads(store.read_text()).items()}

    def embed(self, audio: np.ndarray) -> np.ndarray:
        st = self.ext.create_stream()
        st.accept_waveform(16000, audio)
        st.input_finished()
        e = np.array(self.ext.compute(st))
        return e / (np.linalg.norm(e) + 1e-9)

    def enroll(self, name: str, audio: np.ndarray) -> dict:
        e = self.embed(audio)
        old = self.profiles.get(name)
        e = e if old is None else (old + e) / np.linalg.norm(old + e)
        self.profiles[name] = e
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self.store.write_text(json.dumps({k: v.tolist() for k, v in self.profiles.items()}))
        return {"names": sorted(self.profiles)}

    def identify(self, audio: np.ndarray) -> dict:
        if not self.profiles or len(audio) < 16000:  # under 1 s: too short to tell
            return {"speaker": "", "score": None}
        e = self.embed(audio)
        scores = {k: float(v @ e) for k, v in self.profiles.items()}
        best = max(scores, key=scores.get)
        name = best if scores[best] >= self.threshold else "?"
        return {"speaker": name, "score": round(scores[best], 2), "best": best}


async def tts_file(request):
    path = request.app["tts"].dir / request.match_info["name"]
    if not path.exists() or path.parent != request.app["tts"].dir:
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Cache-Control": "no-store"})


async def index(request):
    return web.FileResponse(ROOT / "static/index.html")


async def info(request):
    return web.json_response(request.app["info"])


async def ws_handler(request):
    app = request.app
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    rec = app["rec"]
    stream = rec.create_stream()
    loop = asyncio.get_running_loop()
    samples = 0
    seg_start = 0.0
    compute = 0.0
    chunk_times = []
    last_partial = ""

    state = {"voice": None}
    received = []  # raw microphone audio of this session, saved on close for debugging
    pending = set()  # finalize() tasks still running; awaited before "stopped" so no segment is lost
    gate = SpeechGate(app["vad"]) if app["vad"] else None

    def step(audio):
        for f in (gate.push(audio) if gate else [audio]):
            stream.accept_waveform(16000, f)
        dts = []
        while rec.is_ready(stream):
            t = time.perf_counter()
            rec.decode_stream(stream)
            dts.append(time.perf_counter() - t)
        text = rec.get_result(stream).strip().lower()
        endpoint = rec.is_endpoint(stream)
        if endpoint:
            rec.reset(stream)
        return text, endpoint, dts

    def finish():
        for f in (gate.flush() if gate else []):
            stream.accept_waveform(16000, f)
        stream.accept_waveform(16000, np.zeros(int(0.66 * 16000), dtype=np.float32))
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        return rec.get_result(stream).strip().lower()

    merge_lock = asyncio.Lock()
    last = {}  # previous final segment: raw, start, end; used to join a number read in pauses

    async def finalize(raw: str, start: float, end: float):
        async with merge_lock:
            replace = bool(last) and end - start < MERGE_MAX_S and raw.split()[0] in _NUMBER_WORDS \
                and last["raw"].split()[-1] in _NUMBER_WORDS
            if replace:
                raw, start = f"{last['raw']} {raw}", last["start"]
            last.update(raw=raw, start=start, end=end)
            await _finalize(raw, start, end, replace)

    async def _finalize(raw: str, start: float, end: float, replace: bool):
        # Numbers to digits first ("ba phẩy bốn triệu" -> "3,4 triệu"); the restorer handles digits fine.
        text, ms = (spoken_to_digits(raw) if raw else raw), 0.0
        if app["punct"] is not None and raw:
            t = time.perf_counter()
            text = await loop.run_in_executor(None, app["punct"], text)
            ms = 1000 * (time.perf_counter() - t)
        if app["hot"] is not None and text:
            text = app["hot"].apply(text)
        spk = {}
        if app["spk"] is not None and received:
            audio = np.concatenate(received)[int(start * 16000):int(end * 16000)]
            spk = await loop.run_in_executor(None, app["spk"].identify, audio)
        await ws.send_json({"type": "final", "replace": replace, "text": text, "raw": raw, "start": start, "end": end,
                            "punct_ms": round(ms, 1), **spk})
        if app["tts"] is not None and state["voice"] and text:
            res = await loop.run_in_executor(None, app["tts"], text, state["voice"])
            if res.get("ok"):
                await ws.send_json({"type": "tts", "url": f"/tts/{res['name']}", "seconds": res["seconds"],
                                    "ms": round(res["ms"])})

    async for msg in ws:
        if msg.type == WSMsgType.TEXT and msg.data.startswith("enroll:") and app["spk"] is not None:
            state["enroll"] = (msg.data[7:].strip(), samples)
            continue
        if msg.type == WSMsgType.TEXT and msg.data == "enroll_done" and state.get("enroll"):
            name, s0 = state.pop("enroll")
            audio = np.concatenate(received)[s0:] if received else np.zeros(0, np.float32)
            if name and len(audio) >= 3 * 16000:
                res = await loop.run_in_executor(None, app["spk"].enroll, name, audio)
                await ws.send_json({"type": "enrolled", "name": name, "seconds": round(len(audio) / 16000, 1), **res})
            else:
                await ws.send_json({"type": "enrolled", "error": "cần tên và ít nhất 3 giây giọng nói"})
            continue
        if msg.type == WSMsgType.TEXT and msg.data.startswith("gate:"):
            if gate:
                gate.min_db = float(msg.data[5:])
            continue
        if msg.type == WSMsgType.TEXT and msg.data.startswith("voice:"):
            state["voice"] = msg.data[6:] or None
            continue
        if msg.type == WSMsgType.TEXT and msg.data == "stop":
            raw = await loop.run_in_executor(app["pool"], finish)
            if raw:
                await finalize(raw, seg_start, samples / 16000)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await ws.send_json({"type": "stopped"})
            break
        if msg.type != WSMsgType.BINARY:
            continue
        audio = np.frombuffer(msg.data, dtype=np.float32).copy()
        if app["dump"] or app["spk"] is not None:  # kept only when something needs it (privacy, memory)
            received.append(audio)
        samples += len(audio)
        # Decoding runs in the worker pool so one session does not block the others; a session's own messages
        # are handled in order, so its stream is never touched by two threads at once.
        text, endpoint, dts = await loop.run_in_executor(app["pool"], step, audio)
        compute += sum(dts)
        chunk_times.extend(dts)
        if endpoint:
            end = samples / 16000
            if text:
                task = asyncio.ensure_future(finalize(text, seg_start, end))
                pending.add(task)
                task.add_done_callback(pending.discard)
            seg_start = end
            last_partial = ""
            await ws.send_json({"type": "partial", "text": ""})
        elif text != last_partial:
            last_partial = text
            await ws.send_json({"type": "partial", "text": text})
        if chunk_times and len(chunk_times) % 5 == 0:
            recent = chunk_times[-20:]
            await ws.send_json({"type": "stats", "audio_s": samples / 16000, "compute_s": compute,
                                "chunk_ms": 1000 * sum(recent) / len(recent)})
    if received and app["dump"]:
        import soundfile as sf
        app["dump"].mkdir(parents=True, exist_ok=True)
        path = app["dump"] / time.strftime("%Y%m%d-%H%M%S.wav")
        sf.write(path, np.concatenate(received), 16000, subtype="PCM_16")
        print("saved", path, flush=True)
    return ws


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--punct-dir", type=Path, help="exp dir of train_punct.py; omit to disable")
    p.add_argument("--threads", type=int, default=2, help="ONNX threads per decode call")
    p.add_argument("--workers", type=int, default=4, help="sessions decoded in parallel (about the vCPU count)")
    p.add_argument("--punct-onnx", action="store_true", help="use the ONNX export even when best.pt exists")
    p.add_argument("--method", default="modified_beam_search")
    p.add_argument("--dump", type=Path, help="save each session's received audio here")
    p.add_argument("--speaker-model", type=Path, help="sherpa-onnx speaker embedding model; enables names")
    p.add_argument("--speaker-store", type=Path, default=Path("data/speakers.json"))
    p.add_argument("--vad", type=Path, help="silero_vad.onnx; gates non-speech to silence")
    p.add_argument("--tts-python", help="python with zerotts installed; enables the voice-swap mode")
    p.add_argument("--tts-model", help="ZeroTTS model dir")
    p.add_argument("--hotwords", type=Path, default=Path("data/hotwords.txt"),
                   help="names to boost (see src/vnasr/hotwords.py); a missing file disables it")
    p.add_argument("--hotwords-score", type=float, default=2.0)
    p.add_argument("--bpe-model", type=Path, default=Path("models/bpe.model"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()

    app = web.Application()
    app["hot"], hot = None, None
    if a.hotwords and a.hotwords.exists() and a.method == "modified_beam_search":
        app["hot"] = hotwords_mod.load(a.hotwords)
        words, vocab = hotwords_mod.write_sherpa_files(app["hot"], a.bpe_model, Path(tempfile.mkdtemp(prefix="hotwords-")))
        hot = dict(hotwords_file=str(words), hotwords_score=a.hotwords_score, modeling_unit="bpe", bpe_vocab=str(vocab))
    app["rec"] = make_recognizer(a.model_dir, a.threads, a.method, hot)
    app["pool"] = concurrent.futures.ThreadPoolExecutor(max_workers=a.workers)
    app["punct"] = None
    if a.punct_dir and any((a.punct_dir / f).exists() for f in ("best.pt", "model.onnx", "onnx/model.onnx")):
        sys.path.insert(0, str(ROOT.parent / "punct"))
        from punctuate import Punctuator
        app["punct"] = Punctuator(a.punct_dir, onnx=a.punct_onnx or None)
        app["punct"]("xin chào")  # warm up
    app["vad"] = str(a.vad) if a.vad else None
    app["dump"] = a.dump
    app["spk"] = Speakers(str(a.speaker_model), a.speaker_store) if a.speaker_model else None
    app["tts"] = TTS(a.tts_python, a.tts_model) if a.tts_python and a.tts_model else None
    app["info"] = {"model": a.model_dir.name, "method": a.method, "threads": a.threads,
                   "hotwords": sorted(app["hot"].names.values()) if app["hot"] else [],
                   "punct": app["punct"] is not None, "vad": bool(a.vad),
                   "stores_audio": bool(a.dump or a.speaker_model), "speakers": sorted(app["spk"].profiles) if app["spk"] else None, "voices": app["tts"].voices if app["tts"] else []}
    app.router.add_get("/", index)
    app.router.add_get("/info", info)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/tts/{name}", tts_file)
    app.router.add_static("/samples", ROOT / "static/samples")
    print(f"open http://{a.host}:{a.port}  model={a.model_dir.name} punct={app['punct'] is not None}", flush=True)
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
