# WhisperX Subtitle Generator

Batch subtitle generator for media libraries. Uses your NVIDIA GPU to transcribe audio from video files and translate subtitles to your language of choice. Outputs Plex-compatible `.srt` files.

## Features

- GPU-accelerated transcription via WhisperX
- Automatic language detection
- Google Translate integration (no API key needed)
- Batch processing of entire TV show seasons/libraries
- Plex-compatible subtitle naming (`Show.S01E01.no.srt`)
- Skips files that already have subtitles
- Dark GUI with progress logging

## Requirements

- Windows 10/11
- Python 3.11 — https://python.org/downloads/release/python-3110/
  - ⚠️ During install, check **"Add Python to PATH"**
- NVIDIA GPU with CUDA support (GTX 1000 series or newer)
- Internet connection (for downloading models and translation)

## Installation

1. Clone or download this repo
2. Double-click **`install.bat`**
3. Wait for it to finish (PyTorch alone is ~2.5GB so this takes a while)
4. Done — run with **`run.bat`**

## Usage

1. Run `run.bat`
2. Click **Browse** and select your target folder (e.g. `Z:\media\tv\Show Name\Season 1`)
3. Click **Scan** to see how many files will be processed
4. Choose your settings:
   - **Whisper model**: `small` for speed, `medium` for quality (recommended), `large-v3` for best
   - **Compute type**: `float16` if you have 8GB+ VRAM, `int8_float16` for less
   - **Source language**: set if you know it, otherwise leave on auto-detect
   - **Translate to**: your target language
   - **Batch size**: 8 is a good default for most GPUs
5. Click **▶ Start Processing**

Subtitle files are saved next to each video file automatically.

## Model Size Guide

| Model    | VRAM   | Speed (RTX 3090) | Quality        |
|----------|--------|------------------|----------------|
| tiny     | ~1 GB  | ~32× realtime    | Poor           |
| base     | ~1.5GB | ~16× realtime    | Okay           |
| small    | ~2.5GB | ~8× realtime     | Good           |
| medium   | ~5 GB  | ~4× realtime     | Very good ★    |
| large-v2 | ~10GB  | ~2× realtime     | Excellent      |
| large-v3 | ~10GB  | ~2× realtime     | Best           |

## Tips

- Close Chrome, games, and other GPU-heavy apps before running
- The first run will download AI models (~1-5GB depending on model size) — subsequent runs are instant
- Use **Dry run** to preview which files will be processed without actually processing them
- Enable **Skip files that already have a .no.srt** to safely re-run without duplicating work
- If the GPU crashes mid-batch, close the GUI fully, restart it, and it will skip already-completed files

## Output Format

Subtitles are saved as `VideoFilename.LANG.srt` next to the video file:
```
Show - S01E01 - Episode Title.mkv
Show - S01E01 - Episode Title.en.srt   (original language, if "Also save original" is checked)
Show - S01E01 - Episode Title.no.srt   (translated)
```

Plex picks these up automatically — no configuration needed.

## Troubleshooting

**"python is not recognized"** — Python is not in PATH. Reinstall Python and check "Add to PATH".

**"No CUDA GPU found"** — PyTorch was installed without CUDA support. Run `install.bat` again, or manually run:
```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**CUDA crashes mid-batch** — Close all GPU-heavy background apps (Chrome with hardware acceleration, games, OBS). Reduce batch size to 4.

**Translation not working** — Check your internet connection. Google Translate is used via HTTP requests.

**Subtitles out of sync** — This can happen if alignment fails. The tool falls back to proportional timing which is close but not perfect. Try re-running that file with a larger model.

## License

MIT — do whatever you want with it.
