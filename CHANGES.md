# Changelog

## [1.0.0] - 2024-03-19

### Added
- Initial implementation of the text processing and classification pipeline
- RDD-based text processing for term frequency analysis
- DataFrame-based text processing pipeline with feature selection
- SVM-based text classification with hyperparameter tuning
- Jupyter notebook conversion for interactive analysis
- Requirements.txt for dependency management

### Features
- Text preprocessing with custom tokenization and stopword removal
- Chi-square feature selection for identifying significant terms
- SVM classifier with grid search for optimal hyperparameters
- Cross-validation for model evaluation
- Performance metrics calculation (F1 score)
- Interactive notebook environment for analysis

### Dependencies
- pyspark>=3.5.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- jupytext>=1.16.0 