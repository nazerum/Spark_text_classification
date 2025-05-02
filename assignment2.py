# Import required libraries
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import json
import re
import os

# (Keeping ML, numpy, plotting imports for later use)
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import *
from pyspark.ml import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Configuration parameters
# -------------------------------
config = {
    # Execution mode: 'local' or 'cluster'
    'mode': 'local',
    # Data paths
    'local_data_path': './data/reviews_devset.json',
    'hdfs_dev_data_path': 'hdfs:///user/dic25_shared/amazon-reviews/full/reviews_devset.json',
    'hdfs_full_data_path': 'hdfs:///user/dic25_shared/amazon-reviews/full/reviewscombined.json',
    # Output paths
    'local_task1_output_path': './output/output_rdd.txt',
    'local_task2_output_path': './output/output_ds.txt',
    'hdfs_output_dir': 'hdfs:///user/dic25_shared/output/assignment2',
    # Spark config
    'spark_master': 'local[*]',  # ignored in cluster mode
    'spark_app_name': 'Assignment2_ChiSquare',
    # Chi-square and output params
    'top_terms_per_category': 75,
    'random_seed': 42,
    'num_features': 2000,
}

# -------------------------------
# Initialize Spark
# -------------------------------
builder = SparkSession.builder.appName(config['spark_app_name'])
if config['mode'] == 'local':
    builder = builder.master(config['spark_master'])
spark = builder.getOrCreate()
sc = spark.sparkContext

# -------------------------------
# Load Stopwords
# -------------------------------
with open('stopwords.txt', 'r') as f:
    stopwords = set(f.read().splitlines())

# -------------------------------
# Helper Functions
# -------------------------------
def preprocess_text(text):
    """Tokenize, lowercase, remove stopwords and single chars"""
    tokens = re.split(r"[\s\d()\[\]{}.!?;:+=\-_'`~#@&*%€$§\\/]+", text.lower())
    return [t for t in tokens if t and len(t) > 1 and t not in stopwords]


def parse_review(line):
    """Parse JSON line to (category, reviewText) or None on failure"""
    try:
        review = json.loads(line)
        return review.get('category'), review.get('reviewText')
    except Exception:
        return None

# -------------------------------
# RDD-based Processing
# -------------------------------
# 1) Read data
data_path = config['local_data_path'] if config['mode'] == 'local' else config['hdfs_dev_data_path']
reviews_rdd = sc.textFile(data_path)

# 2) Map to ((category, term), count)
#    - Map: each review -> [( (cat, term), 1 ) ...]
category_term = (
    reviews_rdd
      .map(parse_review)
      .filter(lambda x: x is not None)
      .flatMap(lambda ct: [((ct[0], term), 1) for term in preprocess_text(ct[1])])
)

# 3) term_freq: ((cat, term), freq_ct)
term_freq = category_term.reduceByKey(lambda a, b: a + b)

# 4) term_total: (term, freq_t)
#    Sum across all categories: column totals for contingency
term_total = (
    term_freq
      .map(lambda kv: (kv[0][1], kv[1]))
      .reduceByKey(lambda a, b: a + b)
)

# 5) category_total: (category, total_tokens_in_cat)
category_total = (
    term_freq
      .map(lambda kv: (kv[0][0], kv[1]))
      .reduceByKey(lambda a, b: a + b)
)

# 6) Grand total N of all tokens
N = term_freq.map(lambda kv: kv[1]).sum()

# Broadcast small side-data
term_total_bc = sc.broadcast(dict(term_total.collect()))
category_total_bc = sc.broadcast(dict(category_total.collect()))
N_bc = sc.broadcast(N)

# -------------------------------
# 7) Compute full chi-square per (category, term)
#    Using: A = freq_ct; B = freq_t - A; C = cat_total - A; D = N - A - B - C
#    chi2 = N * (A*D - B*C)^2 / ((A+B)*(C+D)*(A+C)*(B+D))
# -------------------------------
def compute_chi(kv):
    (cat, term), A = kv
    T = term_total_bc.value.get(term, 0)
    C_tot = category_total_bc.value.get(cat, 0)
    B = T - A
    C = C_tot - A
    D = N_bc.value - A - B - C
    # Avoid zero divisions
    denom = (A + B) * (C + D) * (A + C) * (B + D)
    if denom == 0:
        chi2 = 0.0
    else:
        chi2 = N_bc.value * (A * D - B * C) ** 2 / denom
    return (cat, (term, chi2))

chi_sq_rdd = term_freq.map(compute_chi)

# -------------------------------
# 8) Top-K terms per category
#    Use mapPartitions + heap to pre-aggregate per partition then per key
# -------------------------------
from heapq import nlargest

def topk_per_category(iterator):
    # local dict of heaps: cat -> list of (chi2, term)
    local = {}
    k = config['top_terms_per_category']
    for cat, (term, score) in iterator:
        if cat not in local:
            local[cat] = []
        local[cat].append((score, term))
    # yield top-K for each cat in this partition
    for cat, pairs in local.items():
        for score, term in nlargest(k, pairs):
            yield (cat, (term, score))

# Pre-aggregate in partitions
partial_topk = chi_sq_rdd.mapPartitions(topk_per_category)
# Final top-K across all partitions
top_terms_rdd = (
    partial_topk
      .groupByKey()
      .mapValues(lambda iter_pairs: sorted(iter_pairs, key=lambda x: x[1], reverse=True)[:config['top_terms_per_category']])
)

# -------------------------------
# 9) Merged dictionary: sorted unique terms from chi_sq_rdd
# -------------------------------

def clean_term(term):
    """Strip leading/trailing non-alphanumerics (including quotes)"""
    return re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', term)

merged_dict = (
    chi_sq_rdd
      .map(lambda kv: kv[1][0])
      .map(clean_term)
      .filter(lambda t: t)
      .distinct()
      .sortBy(lambda x: x)
      .collect()
)
merged_dict_line = " ".join(merged_dict)

# -------------------------------
# 10) Write results
#    - Local: driver writes output.txt
#    - Cluster/HDFS: use saveAsTextFile
# -------------------------------
output_lines_rdd = (
    top_terms_rdd
      .sortByKey()
      .map(lambda kv: f"{kv[0]}\t" + " ".join([f"{t}:{chi2:.6f}" for t, chi2 in kv[1]]))
)

if config['mode'] == 'local':
    os.makedirs(os.path.dirname(config['local_task1_output_path']), exist_ok=True)
    with open(config['local_task1_output_path'], 'w') as out:
        for line in output_lines_rdd.collect():
            out.write(line + "\n")
        out.write(merged_dict_line + "\n")
else:
    # Write top terms file
    output_lines_rdd.union(sc.parallelize([merged_dict_line])) \
        .saveAsTextFile(config['hdfs_output_dir'])

#######################################################################################################################################################################################################################

# Part 2: DataFrame-based Text Processing Pipeline
print("\nPart 2: DataFrame-based Text Processing Pipeline")
reviews_df = spark.read.json(data_path)

# Create the text processing pipeline
tokenizer = Tokenizer(inputCol="reviewText", outputCol="tokens")
stopwords_remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="filtered_tokens",
    stopWords=list(stopwords)
)
hashing_tf = HashingTF(
    inputCol="filtered_tokens",
    outputCol="raw_features",
    numFeatures=config['num_features']
)
idf = IDF(inputCol="raw_features", outputCol="features")
# Convert category from string to numeric
category_indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
chi_sq_selector = ChiSqSelector(
    numTopFeatures=config['num_features'],
    featuresCol="features",
    outputCol="selected_features",
    labelCol="categoryIndex"
)

# Create the pipeline
pipeline = Pipeline(stages=[
    tokenizer,
    stopwords_remover,
    hashing_tf,
    idf,
    category_indexer,
    chi_sq_selector
])

# Fit the pipeline
model = pipeline.fit(reviews_df)

# Transform the data
transformed_df = model.transform(reviews_df)

# Get the selected features
selected_features = model.stages[-1].selectedFeatures

# Get the tokens from the transformed data
tokens_df = transformed_df.select("filtered_tokens").collect()
all_tokens = set()
for row in tokens_df:
    all_tokens.update(row.filtered_tokens)

# Sort the tokens and select the ones corresponding to the selected features
sorted_tokens = sorted(list(all_tokens))
selected_terms = [sorted_tokens[i] for i in selected_features]

# Write the selected terms to output file
with open(config['local_task2_output_path'], 'w') as f:
    f.write(" ".join(sorted(selected_terms)) + "\n")

###################################################################################################################################################################################################

"""# Part 3: Text Classification using SVM
print("\nPart 3: Text Classification using SVM")

# Split the data into training, validation, and test sets
train_df, val_df, test_df = reviews_df.randomSplit(
    [config['train_ratio'], config['val_ratio'], config['test_ratio']], 
    seed=config['random_seed']
)

# Create the text processing and classification pipeline
tokenizer = Tokenizer(inputCol="reviewText", outputCol="tokens")
stopwords_remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="filtered_tokens",
    stopWords=list(stopwords)
)
hashing_tf = HashingTF(
    inputCol="filtered_tokens",
    outputCol="raw_features",
    numFeatures=config['num_features']
)
idf = IDF(inputCol="raw_features", outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="normalized_features", p=2.0)
# Convert category from string to numeric
category_indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
# Use LogisticRegression instead of LinearSVC for multi-class classification
lr = LogisticRegression(
    featuresCol="normalized_features",
    labelCol="categoryIndex",
    predictionCol="prediction",
    family="multinomial"
)

# Create the pipeline
pipeline = Pipeline(stages=[
    tokenizer,
    stopwords_remover,
    hashing_tf,
    idf,
    normalizer,
    category_indexer,
    lr
])

# Define the parameter grid for grid search
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .addGrid(lr.maxIter, [10, 100]) \
    .build()

# Create the evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="categoryIndex",
    predictionCol="prediction",
    metricName="f1"
)

# Create the cross-validator
cross_validator = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=config['num_folds'],
    seed=config['random_seed']
)

# Fit the cross-validator
cv_model = cross_validator.fit(train_df)

# Make predictions on the validation set
val_predictions = cv_model.transform(val_df)

# Evaluate the model on the validation set
val_f1 = evaluator.evaluate(val_predictions)
print(f"Validation F1 score: {val_f1}")

# Make predictions on the test set
test_predictions = cv_model.transform(test_df)

# Evaluate the model on the test set
test_f1 = evaluator.evaluate(test_predictions)
print(f"Test F1 score: {test_f1}")

# Get the best parameters
best_params = cv_model.bestModel.stages[-1].extractParamMap()
print("Best parameters:")
for param, value in best_params.items():
    print(f"{param.name}: {value}")

# Save model and results
model_path = os.path.join(output_dir, 'logistic_regression_model')
cv_model.bestModel.write().overwrite().save(model_path)"""

# Stop the Spark session
spark.stop() 