# Model Card: G2LNet

## Model details
- Model type: Graph neural network for crystalline materials.
- Primary use: Predict materials properties from crystal structures.

## Intended use
- Research and benchmarking of materials property prediction.
- Data-driven screening of candidate materials.

## Out-of-scope use
- Safety-critical decision making without expert validation.
- Use on datasets with unknown licensing or provenance.

## Training data
- Dataset: JARVIS-derived datasets prepared by get_data.py.
- Data preprocessing: Stratified sampling and OOD/SES construction.
- Note: Users must ensure dataset licensing and proper citations.

## Evaluation
- Metrics: Reported metrics should include MAE/RMSE and calibration where applicable.
- Splits: Report train/validation/test splits and OOD evaluation.

## Reproducibility
- Fix random seeds in config.yml and document them.
- Report hardware and software versions.

## Limitations
- Performance depends on dataset quality and coverage.
- OOD generalization is not guaranteed.

## Ethical considerations
- Ensure data sources are properly licensed and cited.
- Avoid over-claiming model capabilities beyond validated benchmarks.
