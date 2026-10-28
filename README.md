# CoReMatch

PyTorch implementation of **CoReMatch (Comparison-Ready Representation Matching)**
for graph similarity learning.

## Requirements

Python 3.10 and CUDA 12.1.

```text
torch==2.1.2+cu121
torch-geometric==2.5.3
torch-scatter==2.1.2+pt21cu121
numpy==1.26.4
scipy==1.11.4
tqdm==4.66.5
networkx==3.2.1
```

Install the dependencies:

```bash
conda create -n corematch python=3.10 -y
conda activate corematch
python -m pip install -r requirements.txt
```

Run the commands below from the project root.

## Run Pretrained Model

The graph data and precomputed GED annotations for AIDS700nef, LINUX, and
IMDBMulti are included in `datasets/`. Pretrained checkpoints are provided in
`checkpoints/<dataset>/best_mse.pt`.

Evaluate the pretrained model on each dataset:

```bash
# AIDS700nef
GPU=0 bash scripts/test/test_aids.sh

# LINUX
GPU=0 bash scripts/test/test_linux.sh

# IMDBMulti
GPU=0 bash scripts/test/test_imdb.sh
```

Each run saves `test_result.json` and `test_result.csv` in
`results/<dataset>_<YYYYMMDD_HHMMSS>/`. The timestamp uses Asia/Shanghai time.
The JSON file includes the full-precision metrics.

Evaluate all three datasets:

```bash
GPU=0 bash scripts/test/test_all.sh
```

The batch test uses one timestamp for the dataset folders and saves the combined
results in `results/summary_<YYYYMMDD_HHMMSS>.csv`.

## Running Example

Train CoReMatch from scratch:

```bash
# AIDS700nef
GPU=0 bash scripts/train/train_aids.sh

# LINUX
GPU=0 bash scripts/train/train_linux.sh

# IMDBMulti
GPU=0 bash scripts/train/train_imdb.sh
```

The dataset scripts contain the corresponding hyperparameter settings. Training
outputs are saved in `runs/<dataset>/`, including the checkpoint with the lowest
validation MSE (`best_mse.pt`), its selection metadata (`selection.json`), and
test results (`result.json` and `result.csv`). Use an empty output directory for
each training run.


To evaluate a checkpoint from a training run:

```bash
GPU=0 bash scripts/test/test_imdb.sh --checkpoint runs/IMDBMulti/best_mse.pt
```

Pass the matching architecture settings when evaluating a different model
configuration. Add `--device cpu` to run on CPU, or set `PYTHON_BIN` to select a
Python interpreter. View all command-line options with:

```bash
python src/train.py --help
```
