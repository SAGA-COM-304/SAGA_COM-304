
## 1. Python Dependencies

Add to the `requirements.txt` file at the root of the project the following content:

```text
# Deep learning & data
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0

# Data processing
pandas>=1.5.0
numpy>=1.23.0

# Computer vision & audio I/O
opencv-python>=4.6.0
Pillow>=9.0.0

# CLI parsing (included in the Python standard library)
argparse
```

## 2. System Dependency

### Linux (Debian / Ubuntu)

```bash
sudo apt update
sudo apt install -y \
  yt-dlp \
  ffmpeg \
  parallel \
  gawk
```

### macOS (Homebrew)
```bash
brew update
brew install \
  yt-dlp \
  ffmpeg \
  parallel \
  gawk
```