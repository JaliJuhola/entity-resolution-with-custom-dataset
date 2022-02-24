import pandas as pd
import numpy as np
import time
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
    start = time.time()
    datasetMapping = {
        'df1': df1,
        'df2': df2
    }
    # Get clusters of attributes to create blocks with.
    clusters = clustering.get_clusters(df1, df2)
    # Create blocks by tokenized values inside of cluster.
    blocks = blocking.create_blocks_from_clusters(clusters, datasetMapping)
    # Compare all possible items
    
    (duplicates, total_comparisons) = blocking.get_duplicates(blocks)
    end = time.time()
    print("There were " + str(total_comparisons) + " comparisons at total")
    print("Process took " + str(end - start) + " seconds")
    print("************ Duplicates *************")
    for duplicate in duplicates:
        print(duplicate['item1'])   
        print("")
        print(duplicate['item2'])
        print("*********************************************************")

df1 = read_dataset_to_df('./data/dataset1.json', 'df1')
df2 = read_dataset_to_df('./data/dataset2.json', 'df2')

execute_attribute_clustering_and_token_blocking(df1, df2)