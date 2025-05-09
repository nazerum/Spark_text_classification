# Import required libraries
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import json
import re
import os


from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import *
from pyspark.ml import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F

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

config.update({
    'train_ratio': 0.8,
    'val_ratio':   0.1,
    'test_ratio':  0.1,
    'num_folds':   5,
    # where to save your model locally
    'local_model_output_path': './output/model'
})



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

'''
# -------------------------------
# Part 1RDD-based Processing
# -------------------------------
print("\nPart 1: RDD-based Processing")

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
'''
#######################################################################################################################################################################################################################

# -------------------------------
# Part 2: DataFrame-based Text Processing and Chi-Square Scoring
# -------------------------------
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml import Pipeline

print("\nPart 2: DataFrame-based Text Processing and Chi-Square Scoring")


# temp
data_path = config['local_data_path'] if config['mode'] == 'local' else config['hdfs_dev_data_path']
reviews_df = spark.read.json(data_path)


# Build pipeline
tokenizer = RegexTokenizer(
    inputCol="reviewText",
    outputCol="tokens",
    pattern=r"[\s\t\d()\[\]{}.!?,;:+=\-_'`~#@&*%€$§\\/]+",
    toLowercase=True,
    minTokenLength=2
)

#remove stopwords   
remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="filtered_tokens",
    stopWords=list(stopwords)
)

#count vectorizer
cv = CountVectorizer(
    inputCol="filtered_tokens",
    outputCol="raw_features",
    vocabSize=config['num_features'],
    minDF=2
)

#idf
idf = IDF(inputCol="raw_features", outputCol="features")

#indexer
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")

#pipeline
pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, indexer])
model = pipeline.fit(reviews_df)
transformed = model.transform(reviews_df)

# Chi-Square test over all features
chi2_result = ChiSquareTest.test(
    transformed,
    featuresCol="features",
    labelCol="categoryIndex"
).head()
stats = chi2_result.statistics
stats_array = stats.toArray() if hasattr(stats, "toArray") else stats

# Map indices to (term, chi2) and pick top K
vocab = model.stages[2].vocabulary
indexed = list(enumerate(stats_array))
topk = sorted(indexed, key=lambda x: x[1], reverse=True)[:config['num_features']]
selected = [(vocab[i], stats_array[i]) for i, _ in topk]

# Write out tokens and chi2 scores in descending chi2 order
if config['mode'] == 'local':
    os.makedirs(os.path.dirname(config['local_task2_output_path']), exist_ok=True)
    with open(config['local_task2_output_path'], 'w') as f:
        for term, score in selected:
            f.write(f"{term}\t{score:.6f}\n")
else:
    sc.parallelize([f"{term}\t{score:.6f}" for term, score in selected]) \
      .saveAsTextFile(config['hdfs_ds_output_dir'])

#spark.stop()'''

###################################################################################################################################################################################################

# -------------------------------
# Part 3 (revised): cache & classify
# -------------------------------

from pyspark.ml.feature import ChiSqSelector, StandardScaler, Normalizer
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# 1) Precompute TF–IDF features once and cache
prepared_df = model.transform(reviews_df) \
                   .select("categoryIndex", "features") \
                   .cache()
# force materialization so it stays in memory
prepared_df.count()

# 2) Split into train/val/test on the *feature* DataFrame
train_df, val_df, test_df = prepared_df.randomSplit(
    [config['train_ratio'], config['val_ratio'], config['test_ratio']],
    seed=config['random_seed']
)
train_val_df = train_df.union(val_df).cache()
test_df.cache()

# 3) Build *only* the classification pipeline
selector = ChiSqSelector(
    featuresCol="features",
    outputCol="selectedFeatures",
    labelCol="categoryIndex"
)
scaler = StandardScaler(
    inputCol="selectedFeatures",
    outputCol="scaledFeatures",
    withMean=False  # keep sparse
)
normalizer = Normalizer(
    inputCol="scaledFeatures",
    outputCol="normFeatures",
    p=2
)
lsvc = LinearSVC(
    featuresCol="normFeatures",
    labelCol="categoryIndex"
)
ovr = OneVsRest(
    classifier=lsvc,
    labelCol="categoryIndex",
    featuresCol="normFeatures"
)

cls_pipeline = Pipeline(stages=[selector, scaler, normalizer, ovr])

# 4) Hyper-parameter grid (chi2 top K, standardization, regParam, maxIter)
#    See Assignment 2 Part 3 :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
paramGrid = (ParamGridBuilder()
    .addGrid(selector.numTopFeatures, [2000, 500])
    .addGrid(scaler.withStd, [True, False])
    .addGrid(lsvc.regParam, [0.01, 0.1, 1.0])
    .addGrid(lsvc.maxIter, [10, 100])
    .build()
)

evaluator = MulticlassClassificationEvaluator(
    labelCol="categoryIndex",
    predictionCol="prediction",
    metricName="f1"
)

# 5) TrainValidationSplit *on the cached feature DataFrame*

# this is to figure out best parallelisation parameter
if config["mode"] == "local":
    parallelism = 1
else:
    executor_infos = sc._jsc.sc().statusTracker().getExecutorInfos()
    active_executors = [
        e.host()
        for e in executor_infos
        if "driver" not in str(e)  # Skip the driver
    ]
    num_executors = len(active_executors)
    executor_cores = int(sc.getConf().get("spark.executor.cores", "1"))
    parallelism = min(len(paramGrid), num_executors * executor_cores)


tvs = TrainValidationSplit(
    estimator=cls_pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    # only classify, so split ratio is train/(train+val)
    trainRatio=config['train_ratio'] / (config['train_ratio'] + config['val_ratio']),
    parallelism=parallelism,                     # <— lower to avoid local OOM/worker crashes
    seed=config['random_seed']
)

# Fit on train+val
tvsModel = tvs.fit(train_val_df)

# 6) Evaluate on the held‐out test set
predictions = tvsModel.transform(test_df)
f1 = evaluator.evaluate(predictions)
print(f"Test F1 score: {f1:.4f}")

# 7) (Optional) persist best model
if config['mode']=='local':
    tvsModel.bestModel.write().overwrite().save(config['local_model_output_path'])
else:
    tvsModel.bestModel.write().overwrite().save(os.path.join(config['hdfs_output_dir'], "best_model"))

# 8) Inspect best params
sel_m    = tvsModel.bestModel.stages[0]
scaler_m = tvsModel.bestModel.stages[1]
best_svm = tvsModel.bestModel.stages[-1].getClassifier()

print("Best hyper‐parameters:")
print(f"  numTopFeatures = {sel_m.getNumTopFeatures()}")
print(f"  withStd        = {scaler_m.getOrDefault('withStd')}")
print(f"  regParam       = {best_svm.getRegParam()}")
print(f"  maxIter        = {best_svm.getMaxIter()}")

spark.stop()