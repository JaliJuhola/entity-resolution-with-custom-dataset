import pandas as pd
import numpy as np

import json
from nltk import ngrams

import clustering
import blocking
import weighting
import similarity
from itertools import product


def read_dataset_to_df(datasetDir, datasetName):
    df = pd.read_json(datasetDir)
    df['dataset_name'] = datasetName
    return df

def execute_attribute_clustering_and_token_blocking(df1, df2):
    datasetMapping = {
        'df1': df1,
        'df2': df2
    }
    clusters = clustering.get_clusters(df1, df2)
    blocks = blocking.create_blocks_from_clusters(clusters, datasetMapping)
    duplicates = blocking.get_duplicates(blocks)
    for duplicate in duplicates:
        print("************ Duplicate *************")
        print(duplicate['item1'])   
        print("------------------------------------")
        print(duplicate['item2'])
        print("*************************")

def get_entities_by_df(df, dfName):
    indexes = []
    for idx, row in df.iterrows():
        indexes.append({'id': idx, 'df': row['dataset_name']})

    return indexes


def execute_meta_blocking(df1, df2):
    datasetMapping = {
        'df1': df1,
        'df2': df2
    }
    df1Entities = get_entities_by_df(df1, 'df1')
    df2Entities = get_entities_by_df(df2, 'df2')
    clusters = clustering.get_clusters(df1, df2)
    setOfClusters = blocking.create_blocks_from_clusters(clusters, datasetMapping)
    comparisonsForDataset = blocking.get_comparisons_according_to_blocking(setOfClusters)
    commonBlocksResults = weighting.common_blocks_scheme(comparisonsForDataset)
    #cardinalityPrunedComparisons = weighting.cardinality_node_pruning(comparisonsForDataset, df1Entities, df2Entities)
    results = similarity.get_duplicates_by_comparisons(df1, df2, commonBlocksResults, datasetMapping)
    for item in results:
        print(item)


df1 = read_dataset_to_df('./data/dataset1.json', 'df1')
df2 = read_dataset_to_df('./data/dataset2.json', 'df2')
execute_meta_blocking(df1, df2)
