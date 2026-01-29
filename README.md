# G2LNet

G2LNet is a graph-based deep learning framework for materials property prediction with a focus on reliable data sampling, stratification, and reproducible training.

## Highlights
- Graph neural network architecture tailored for crystalline materials.
- Stratified data sampling and out-of-distribution (OOD/SES) set construction.
- End-to-end pipeline from dataset preparation to training and evaluation.

## Requirements
- Python 3.9+
- PyTorch and PyTorch Geometric
- Additional dependencies listed in requirements.txt

Install dependencies:
- pip install -r requirements.txt

## Data preparation
Dataset loading and preprocessing are handled by the utilities in utils. Prepare your dataset under data with the expected structure before training. If you use JARVIS datasets, ensure you comply with their licenses and citation requirements.

## Quick start
Training example:
- python main.py --config_file ./config.yml --task_type train --points 200 --epochs 3

## Configuration
Edit config.yml to set dataset paths, model parameters, sampling strategy, and training settings.

## Reproducibility
For reproducible results, document and fix:
- Random seed(s) used for data sampling and training.
- Exact dataset versions and preprocessing steps.
- Hardware (GPU model, CUDA version) and software versions.
- Config file used for each reported experiment.

## Outputs
Training outputs, checkpoints, and metrics are stored under output_multitask by default.

## Project structure
- main.py: entry point for training and evaluation
- model.py: model definitions
- get_data.py: data sampling and dataset generation
- utils/: dataset utilities, metrics, and training helpers

## Citation
If you use this codebase, please cite it using the metadata in CITATION.cff.

## Model card
See MODEL_CARD.md for intended use, limitations, and evaluation guidance.

## License
See LICENSE for the licensing terms.

## Contributing and security
Please read CONTRIBUTING.md and SECURITY.md before reporting issues or submitting changes.
