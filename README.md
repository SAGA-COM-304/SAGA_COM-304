# Setup
## Install libraries
### Using conda (Local)
```
conda create -n saga304 python=3.10 -y
conda activate saga304
pip install --upgrade pip
pip install -e . --upgrade
pip install git+https://github.com/NVIDIA/Cosmos-Tokenizer.git --no-dependencies
python -m ipykernel install --user --name saga304 --display-name "nano4M kernel (saga304)"
```

### Using conda (SCITAS)
The /work/com-304/SAGA/ directory is designed to be our shared directory on the SCITAS cluster.
Create a new search place for conda environments:
```
conda config --append envs_dirs /work/com-304/SAGA/.envs
```
Activate the conda environment:
```
conda activate saga304
```

### Install ffmpeg (required by pydub) 

#### from conda-forge
```
conda install -c conda-forge ffmpeg --yes
```
#### On macOS, you can alternatively install ffmpeg via Homebrew
```
brew install ffmpeg
```

## Training model
TODO

### On SCITAS
```
sbatch submit_job_multi_node_scitas.sh <config_file> <your_wandb_key>
```
# Useful resources
[Foundation Models repository](https://github.com/EPFL-VILAB/com-304-FM-project)

[Overleaf Project](https://www.overleaf.com/read/brbpqrkfsnmn#35fa19)

[Conventional Commits rules](https://www.conventionalcommits.org/en/v1.0.0/)
