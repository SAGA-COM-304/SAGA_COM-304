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

#### If needed to create it back
```
conda create --prefix /work/com-304/SAGA/.envs/saga304 python=3.10 -y
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


## Run notebook on SCITAS
```
srun -t 120 -A com-304 --qos=com-304 --gres=gpu:1 --mem=16G --pty bash
```
```
jupyter lab --no-browser --port=8888 --ip=$(hostname -i)
```
On your local machine, run:
```
ssh -L 8756:<IP+Port> -l bousquie izar.epfl.ch -f -N
```

### Kill process
```
lsof -ti:<port_number>
kill <pid>
```

### Launch training on a node :

```
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py --config cfgs/nano4M/SAGAnano4M.yaml
```
```
sbatch submit_job_multi_node_scitas.sh cfgs/nano4M/SAGAnano4M.yaml
```

# Useful resources
[Foundation Models repository](https://github.com/EPFL-VILAB/com-304-FM-project)

[Overleaf Project](https://www.overleaf.com/read/brbpqrkfsnmn#35fa19)

[Conventional Commits rules](https://www.conventionalcommits.org/en/v1.0.0/)

