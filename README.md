# MoE Streamflow Prediction

Code for **A Mixture-of-Experts Deep Learning Architecture for Daily Streamflow Prediction across the Sierra Nevada in California**.

Run all commands from the project root. Prepared data are included in `processed_data/`.

## Setup

Create a Conda environment and install the provided requirements:

```bash
conda create -n moe_tau python=3.10 -y
conda activate moe_tau
python3 -m pip install -r requirements.txt
```

## Training and testing

```bash
python3 runs/run_moe_tau.py
```

This trains and tests MoE-tau using `runs/moe_tau.yml`. Results and checkpoints are saved in `runs/moe_tau_<timestamp>/`. Add `--gpu -1` to run on CPU.

## Analyses

Replace `RUN_DIR` below with your trained run folder. Keep its saved `config.yml` and `train_data/` alongside the checkpoints.

Each script runs independently and saves figures in `output/`. To select an exact checkpoint, replace `--run-dir RUN_DIR` with `--checkpoint /path/to/model_epoch199.pt`.

### Figure 5: Gating-weight heatmaps

Plot training and testing expert gating weights under low-, mid-, and high-flow regimes.

```bash
python3 analysis/plot_heatmap.py --run-dir RUN_DIR
```

### Figure 6: Dataset distributions

Plot monthly discharge, precipitation, and temperature for ISB, NML, and YRS. This uses only the prepared data; no checkpoint is needed.

```bash
python3 analysis/plot_dataset_distribution.py
```

### Figures 7–8: Watershed expert usage

Plot the mean expert gating weights for one selected watershed.

```bash
python3 analysis/plot_selected_watershed_expert_usage.py --run-dir RUN_DIR --watershed YRS
```

For Figure 7, run the expert-usage script separately for **ISB, NML, and YRS**. For Figure 8, use **YRS, FOL, and SCC**. Combine the individual panels in your manuscript.

### Spearman analysis

Plot the relationship between watershed data-distribution differences and expert-usage differences, using YRS as the reference watershed.

```bash
python3 analysis/plot_spearman.py --run-dir RUN_DIR --watershed YRS
```

Built on [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology).
