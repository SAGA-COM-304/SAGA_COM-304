# SAGA_COM-304 - Foundation Model Project

This repository contains the code, demonstration notebook, and setup instructions for the SAGA group's project in the COM-304 course at EPFL. The project focuses on training and evaluating a version of Nano4m extended with video and audio modalities.

You will find:
- All source code in the `nanofm/` directory
- A demonstration notebook (`demo.ipynb`) to showcase and test the models
- Environment setup instructions for SCITAS and local use
- Training and data usage guidelines

---

## File Structure

```text
.
├── README.md
├── requirements.txt
├── demo.ipynb
├── run_training.py
├── submit_job_multi_node_scitas.sh
├── small_vgg.csv (example of how our csv for preparing the data were structered)
├── img/
│   └── epfl.png
├── nanofm/
│   ├── data/
│   │   ├── multimodal/
│   │   │   ├── adapted_multimodal_dataset.py (we added)
│   │   │   ├── masking.py                      
│   │   │   ├── simple_multimodal_dataset.py
│   │   │   └── utils.py
│   │   ├── tokenizers/ (we added)
│   │   │   ├── label_map.py                  
│   │   │   ├── video_tokenizer.py
│   │   │   ├── audio_tokenizer.py
│   │   │   ├── image_tokenizer.py
│   │   │   ├── dataset.py
│   │   │   ├── main.py (first script to tokenize our data)
│   │   │   ├── main2.py (second script to tokenize our data)
│   │   │   ├── check_token_shapes.py       
│   │   │   ├── test_image_tokenizer.py
│   │   │   ├── test_dataset.py
│   │   │   ├── video_image_tokenizer.py
│   │   │   └── WavTokenizer/... (all files regarding the implementation of an audio tokenizer)
│   │   │       
│   │   ├── utils.py
│   │   └── test_utils.py
│   ├── models/...
│   ├── modeling/...
│   └── utils/...
├── cfgs/
│   └── nano4M/
│       ├── SAGAnano4M.yaml
│       ├── SAGAnano4M_audio.yaml
│       ├── SAGAnano4M_depth.yaml
│       └── SAGAnano4M_reduce_max.yaml
└── dataset_module/ (we added)
    └── src/
        └── downloader/
            ├── split_batch_csv.py (our csv maker file)
            └── download_RR.sh (our downloading script, see more below)
```

---

## Environment Setup (SCITAS)

In the root of the project, execute the following commands to set up your environment with all required libraries :

```bash
# Create and activate a conda environment
conda create -n saga304 python=3.10 -y
conda activate saga304

# Upgrade pip
pip install --upgrade pip

# Install project in editable mode (if needed)
pip install -e . --upgrade

# Install Cosmos Tokenizer (if needed)
pip install git+https://github.com/NVIDIA/Cosmos-Tokenizer.git --no-dependencies

# Install the requirements
conda install -c conda-forge --file requirements.txt --yes

# For any packages not available via conda, use pip:
pip install -r requirements.txt

# Install ffmpeg (for audio/video processing)
conda install -c conda-forge ffmpeg --yes

# (Optional) Register the Jupyter kernel for this environment
python -m ipykernel install --user --name saga304 --display-name "nano4M kernel (saga304)"
```

---

### Data Location on SCITAS

The dataset used for training and evaluation is available on the shared SCITAS storage at:

```
/work/com-304/SAGA/tokens_16_05
```

---
## Configuration Files (`cfgs/`)

The `cfgs/` directory contains YAML configuration files used for training and running experiments. These files specify model parameters, dataset paths, training hyperparameters, logging options (e.g., wandb), and other settings.

You can create or modify YAML files in `cfgs/` to customize experiments according to your needs.

---

### Training (on SCITAS)

Make sure to change the wandb attributes in the `.yaml` files:
```yaml
wandb_project: ...
wandb_entity: ...
wandb_run_name: ...
```
And in the root of the project run this command to run any of the configuration files in the `./cfgs/` folder:
```bash
sbatch submit_job_multi_node_scitas.sh <config_file> <your_wandb_key>
```

---

### Run notebooks on SCITAS (interactive session)
```bash
srun -t 120 -A com-304 --qos=com-304 --gres=gpu:1 --mem=16G --pty bash
jupyter lab --no-browser --port=8888 --ip=$(hostname -i)
```
On your local machine, run:
```bash
ssh -L 8888:<IP+Port> -l <SCITAS_username> izar.epfl.ch -f -N
```

---

## Dataset Module Scripts

The `dataset_module/src/downloader` directory contains the main source code for dataset downloading, manipulating cookies with a Round Robin algorithm to allow the most efficient downloading from youtube without going over the threshold allowed by Youtube. It also contains a script that we used for csv manipulations.


## Useful resources
[Foundation Models repository](https://github.com/EPFL-VILAB/com-304-FM-project)

[Overleaf Project](https://www.overleaf.com/read/brbpqrkfsnmn#35fa19)

