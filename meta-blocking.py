import pandas as pd
import numpy as np

import json
from nltk import ngrams
import time

import clustering
import blocking
import weighting
import similarity
from itertools import product


def read_dataset_to_df(datasetDir, datasetName):
    df = pd.read_json(datasetDir)
    df['dataset_name'] = datasetName
    return df

def get_entities_by_df(df, dfName):
    indexes = []
    for idx, row in df.iterrows():
        indexes.append({'id': idx, 'df': row['dataset_name']})

    return indexes

df1 = read_dataset_to_df('./data/dataset1.json', 'df1')
df2 = read_dataset_to_df('./data/dataset2.json', 'df2')

datasetMapping = {
    'df1': df1,
    'df2': df2
}
df1Entities = get_entities_by_df(df1, 'df1')
df2Entities = get_entities_by_df(df2, 'df2')

def use_jaccard_weighting(comparisonsForDataset, start, useWeightEdgePruning):
    results = []
    jaccardComparisonResults = None
    if useWeightEdgePruning:
        # Calculate jaccard weights globally and prune all under median threshold
        jaccardWeightingResults = weighting.jaccard_scheme(comparisonsForDataset, df1Entities, df2Entities)
        jaccardComparisonResults = weighting.jaccard_weight_edge_pruning(jaccardWeightingResults)
    else:
        # Prune blocking graph according to local threshold (drop all under average local jaccard weighted items)
        cardinalityJaccardResults = weighting.jaccard_cardinality_node_pruning(comparisonsForDataset, df1Entities, df2Entities)
        jaccardComparisonResults = cardinalityJaccardResults.groupby(by=['mainIndex', 'mainDf', 'secondaryIndex', 'secondaryDf']).size()
    (comparisons, duplicates) = similarity.compare_jaccard_results(df1, df2, jaccardComparisonResults, datasetMapping)
    end = time.time()
    print("In the process at the end total of " + str(comparisons) + " comparisons were made")
    print("Process took " + str(end - start) + " seconds")
    for item in duplicates:
        print(item)
        print("*********")

def use_common_blocks(comparisonsForDataset, start, useWeightEdgePruning):
    pruneResults = None
    if useWeightEdgePruning:
        pruneResults = weighting.common_blocks_scheme(comparisonsForDataset)
    else:
        cardinalityPrunedComparisons = weighting.cardinality_node_pruning(comparisonsForDataset, df1Entities, df2Entities)
        pruneResults = cardinalityPrunedComparisons.groupby(by=['mainIndex', 'mainDf', 'secondaryIndex', 'secondaryDf']).size()
    (comparisons, duplicates) = similarity.get_duplicates_by_comparisons(df1, df2, pruneResults, datasetMapping)
    end = time.time()
    print("In the process at the end total of " + str(comparisons) + " comparisons were made")
    print("Process took " + str(end - start) + " seconds")
    for item in duplicates:
        print(item)
        print("*********")
# List of entities and its indeses in both datasets

# Create links between properties of df1 and df2
def test_jaccard():
    print("Jaccard weight edge pruning")
    df1 = read_dataset_to_df('./data/dataset1.json', 'df1')
    df2 = read_dataset_to_df('./data/dataset2.json', 'df2')
    start = time.time()
    clusters = clustering.get_clusters(df1, df2)
    setOfClusters = blocking.create_blocks_from_clusters(clusters, datasetMapping)
    # Get all comparisons for these two datasets
    # blocks with 1 entity and all entities in same dataset are filtered
    # due to clean-clean nature of data
    comparisonsForDataset = blocking.get_comparisons_according_to_entity_clustering(setOfClusters)
    # here we have four options:
    # 1. Jaccard weighting with weight edge pruning
    use_jaccard_weighting(comparisonsForDataset, start, useWeightEdgePruning=True)
    print()
    print()
    print("---------------")
    print()
    print()
# 2. Jaccard weighting with cardinality node pruning
#use_jaccard_weighting(comparisonsForDataset, useWeightEdgePruning=False)
# 3. common blocks weighting with weight edge pruning
def test_common_blocking():
    print("Common blocking WeightEdgePruning")
    df1 = read_dataset_to_df('./data/dataset1.json', 'df1')
    df2 = read_dataset_to_df('./data/dataset2.json', 'df2')
    start = time.time()
    clusters = clustering.get_clusters(df1, df2)
    setOfClusters = blocking.create_blocks_from_clusters(clusters, datasetMapping)
    comparisonsForDataset = blocking.get_comparisons_according_to_entity_clustering(setOfClusters)
    use_common_blocks(comparisonsForDataset, start, useWeightEdgePruning=True)
    print()
    print()
    print("---------------")
    print()
    print()
    # 4. commons blocks with cardinality node pruning
    print("Common blocking with cardinality pruning")
    df1 = read_dataset_to_df('./data/dataset1.json', 'df1')
    df2 = read_dataset_to_df('./data/dataset2.json', 'df2')
    start = time.time()
    clusters = clustering.get_clusters(df1, df2)
    setOfClusters = blocking.create_blocks_from_clusters(clusters, datasetMapping)
    comparisonsForDataset = blocking.get_comparisons_according_to_entity_clustering(setOfClusters)
    use_common_blocks(comparisonsForDataset, start, useWeightEdgePruning=False)

test_common_blocking()
print("********** JACCARD **************")
test_jaccard()