# Memory-efficient malaria cell classification (Haar wavelet + spatial attention + CNN-Transformer)

Code of the paper: a 4-level Haar wavelet decomposition with spatial attention, a multi-scale CNN and a transformer block, trained with stage-wise gradient checkpointing, micro-batch gradient accumulation and light separable convolutions, and compressed with post-training quantization (TFLite fp16 / int8).

## Notebook

`01_memory_efficient_pipeline.ipynb` runs the whole study on the **NIH malaria cell images** (binary: `Uninfected` / `Parasitized`):

1. **Automatic hyperparameter search**: Bayesian optimisation with Optuna (TPE sampler, median pruner) on a held-out tuning pool that is never used for the final evaluation. Nothing is hard-coded: the best configuration is saved to `outputs/best_hyperparameters.json`.
2. Sanity checks (checkpointed gradients equal the normal gradients, the training step learns).
3. Memory benchmark (peak GPU memory, activation memory, step time) with and without each technique, and model size (parameters, weights file, TFLite files).
4. 5-fold stratified cross-validation (mean ± SD, paired test against the original architecture).
5. Quantitative explainability metrics (faithfulness, localization, robustness) of the spatial-attention maps against gradient saliency and a random baseline.

## How to run

1. Set `DATASET_PATH` in the first code cell to the folder that contains the two class sub-folders (`Uninfected/`, `Parasitized/`).
2. Run the notebook on a machine with a GPU. Results, weights and figures are written to `outputs/`.
3. `QUICK_TEST = True` (first code cell) runs a tiny version in a few minutes, to check that everything works.

The cross-validation cell saves its results after every fold and stops cleanly at `MAX_HOURS`; rerun it to resume (set `INPUT_RESULTS` to the saved `accuracy_by_fold.json` when resuming in a new session).

## Hyperparameters

They are found by the search and stored in a JSON file, so a new search can give slightly different values. `best_hyperparameters_trial11.json` contains the best configuration of the search used for the reported results. To reproduce those numbers, copy it to `outputs/best_hyperparameters.json` (the notebook then loads it instead of searching) or set `INPUT_HP_FILE`. To search again, delete `outputs/best_hyperparameters.json` or set `FORCE_NEW_SEARCH = True`.


Python 3, TensorFlow 2.x with a GPU, scikit-learn, pandas, scipy, matplotlib, Pillow, optuna (installed automatically if missing).
