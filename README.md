# Setup
## Install libraries
### Using conda
```
conda create -n saga304 python=3.10 -y
conda activate saga304
pip install --upgrade pip
pip install -e .
pip install git+https://github.com/NVIDIA/Cosmos-Tokenizer.git --no-dependencies
python -m ipykernel install --user --name saga304 --display-name "nano4M kernel (saga304)"
```

## Install ffmpeg (required by pydub) 

### from conda-forge
```
conda install -c conda-forge ffmpeg --yes
```
### On macOS, you can alternatively install ffmpeg via Homebrew
```
brew install ffmpeg
```

# Useful resources
[Foundation Models repository](https://github.com/EPFL-VILAB/com-304-FM-project)

[Overleaf Project](https://www.overleaf.com/read/brbpqrkfsnmn#35fa19)

[Conventional Commits rules](https://www.conventionalcommits.org/en/v1.0.0/)
