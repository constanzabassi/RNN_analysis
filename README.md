# RNN_analysis

Core workflow for loading, aligning, and preparing neural data for RNN analyses.

## Structure
- `helper_functions/data_loader.py`: load imaging data and trial metadata
- `helper_functions/data_aligner.py`: align trials and compute per-trial summaries
- `helper_functions/data_pipeline.py`: build dataset loaders and cell-type info
- `utils/dataset_processor.py`: batch alignment helper
- `notebooks/run_RNN_analysis.ipynb`: end-to-end example

## Quick start
1. Update the dataset list in `notebooks/run_RNN_analysis.ipynb`.
2. Run the notebook to load and align data.
3. Use `rnn_inputs` (shape: trials x time x neurons) as model input.

Notes:
- Pupil/latent-state handling has been removed from the core workflow.
