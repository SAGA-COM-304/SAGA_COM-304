# README
## Setup (In SCITAS)
### Install libraries
In the root of the project execute those commands to install the environments with the libraries. If needed, all the libraries are listed in the `pyproject.toml` file
```bash
conda create -n saga304 python=3.10 -y
conda activate saga304
pip install --upgrade pip
pip install -e . --upgrade
pip install git+https://github.com/NVIDIA/Cosmos-Tokenizer.git --no-dependencies
python -m ipykernel install --user --name saga304 --display-name "nano4M kernel (saga304)"
conda install -c conda-forge ffmpeg --yes
```

### Train model
Make sure to change the wandb attributes in the `.yaml` files
```yaml
wandb_project: ...
wandb_entity: ...
wandb_run_name: ...
```
And in the root of the project run this command to run any of the configuration files in the `./cfgs/` folder. 
```bash
sbatch submit_job_multi_node_scitas.sh <config_file> <your_wandb_key>
```

### Run the notebooks
```bash
srun -t 120 -A com-304 --qos=com-304 --gres=gpu:1 --mem=16G --pty bash
```
```bash
jupyter lab --no-browser --port=8888 --ip=$(hostname -i)
```
On your local machine, run:
```bash
ssh -L 8888:<IP+Port> -l <SCITAS_username> izar.epfl.ch -f -N
```

## Useful resources
[Foundation Models repository](https://github.com/EPFL-VILAB/com-304-FM-project)

[Overleaf Project](https://www.overleaf.com/read/brbpqrkfsnmn#35fa19)
