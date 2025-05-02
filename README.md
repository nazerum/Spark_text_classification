# DIC 2025 Assignment 2: Text Processing and Classification using Apache Spark

## Project Structure
- `assignment_2_notebook.ipynb`: Main Jupyter Notebook with implementation
- `stopwords.txt`: List of stopwords for text preprocessing
- `output_rdd.txt`: RDD-based chi-square term selection results
- `output_ds.txt`: DataFrame-based feature selection results

## Requirements
- Apache Spark
- PySpark
- Python 3.8+

## Notebook Contents
1. **Part 1**: RDD-based Chi-Square Term Selection
   - Preprocesses review texts
   - Calculates chi-square values for terms
   - Outputs top terms per category

2. **Part 2**: DataFrame/ML Pipeline
   - Tokenization
   - Stopword removal
   - TF-IDF feature extraction
   - Chi-square feature selection

3. **Part 3**: Text Classification
   - SVM classifier
   - Grid search for hyperparameter tuning
   - Performance evaluation using F1 score

## Usage
1. Open the notebook in Jupyter
2. Adjust the `dev_data_path` to point to your development dataset
3. Run cells sequentially

## Notes
- Uses development dataset to keep cluster usage low
- Implements best practices for ML experiment design
- Follows Assignment 2 instructions precisely
