# WhisperX Subtitle Generator

A GUI tool for batch-generating subtitles for large media libraries using your NVIDIA GPU. Transcribes audio with WhisperX and translates using Argos Translate (offline) or Google Translate.

Built mainly for Plex libraries where existing subtitles from tools like Bazarr don't match properly.

---

## Requirements

- Windows 10/11
- Python 3.11 — https://python.org/downloads/release/python-3110/
  - Check **"Add Python to PATH"** during install
- NVIDIA GPU (GTX 1000 series or newer)
- Internet connection for first-time model downloads

---

## Installation

1. Clone or download this repo
2. Double-click **`install.bat`**
3. Wait for it to finish — PyTorch is large (~2.5GB) so this takes a while
4. Run with **`run.bat`**

---

## Usage

1. Run `run.bat`
2. Add folders to the queue using **Browse** + **Add to Queue**
3. Use ▲/▼ to set processing priority
4. Click **Scan Queue** to count files
5. Set your options and click **▶ Start Processing**

Subtitles are saved next to each video file in Plex format: `Show.S01E01.no.srt`

---

## Settings

**Whisper model** — bigger = more accurate but slower

| Model | VRAM | Speed (RTX 3090) | Quality |
|-------|------|-----------------|---------|
| tiny | ~1GB | ~32x realtime | poor |
| base | ~1.5GB | ~16x realtime | okay |
| small | ~2.5GB | ~8x realtime | good |
| medium | ~5GB | ~4x realtime | very good |
| large-v3 | ~10GB | ~2x realtime | best |

**Compute type** — `float16` for 8GB+ VRAM, `int8_float16` if you have less

**Translation engine**
- `Argos (local, offline)` — runs fully offline, no rate limits, good for large batches. Downloads a ~100MB language pack on first use.
- `Google Translate (online)` — better quality but requires internet and may rate-limit on large batches

**Batch size** — how many audio chunks to process at once. Higher = faster but more VRAM. Start at 4-8 and go up if stable.

---

## Options

- **Skip files that already have a .en.srt / .no.srt** — avoid reprocessing what's already done
- **Also save original-language .srt** — saves the raw transcription before translation
- **Delete existing .en.srt / .no.srt** — removes old subtitle files before generating new ones, useful for replacing mismatched Bazarr subs
- **Dry run** — lists files without processing them, useful for checking what will be affected

---

## Tips

- Close Chrome and other GPU-heavy apps before running
- The first run downloads AI models — subsequent runs are much faster
- If the GPU crashes mid-batch, close and reopen the GUI (resets CUDA state), then run again — already-completed files will be skipped
- For a large backlog, `small` model is a reasonable tradeoff between speed and quality

---

## Troubleshooting

**"python is not recognized"** — Python not in PATH. Reinstall and check "Add to PATH".

**"No CUDA GPU found"** — PyTorch was installed without CUDA. Run `install.bat` again.

**CUDA crashes** — Close background GPU apps, lower batch size.

**Translation not working (Google)** — Check internet connection, or switch to Argos.

**Subtitles out of sync** — Alignment occasionally fails and falls back to proportional timing. Re-running the specific file usually fixes it.

---

## License

MIT
