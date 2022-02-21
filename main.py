import pandas as pd
import numpy as np

import json
from nltk import ngrams

import clustering
import blocking

JACCARD_THRESHOLD = 0.5

def read_dataset_to_df(datasetDir, datasetName):
    df = pd.read_json(datasetDir)
    df['dataset_name'] = datasetName
    return df

def get_duplicates(blocks):
    already_compared = set()
    duplicates = []
    for blocks in blocks:
        for blockName in blocks.groups:
            if blockName not in ['df1', 'df2']:
                block = blocks.get_group(blockName)
                if len(block.index) > 1:
                    for mainIndex, mainRow in block.iterrows():
                        for secondaryIndex, secondaryRow in block.iterrows():
                            compared_id = f"{mainRow['dataset_name']}-{mainIndex}*{secondaryRow['dataset_name']}-{secondaryIndex}"
                            if mainRow['dataset_name'] != secondaryRow['dataset_name'] and compared_id not in already_compared:
                                already_compared.add(compared_id)
                                already_compared.add(f"{secondaryRow['dataset_name']}-{secondaryIndex}*{mainRow['dataset_name']}-{mainIndex}")
                                mainRowDuplicateCleaned = mainRow.drop(labels=['dataset_name', 'tokens']).dropna()
                                secondaryRowDuplicateCleaned = secondaryRow.drop(labels=['dataset_name', 'tokens']).dropna()
                                if(is_duplicate(mainRow, secondaryRow)):
                                    duplicates.append({'item1': mainRowDuplicateCleaned, 'item2': secondaryRowDuplicateCleaned})
    return duplicates      

def jaccard_similarity(list1, list2):
    intersectionOfLists = list1.intersection(list2)
    unionOfSeries = list1.union(list2)
    return float(len(intersectionOfLists) / len(unionOfSeries))

def is_duplicate(row1, row2):
    row1 = row1.drop(labels=['dataset_name', 'tokens']).dropna()
    row2 = row2.drop(labels=['dataset_name', 'tokens']).dropna()
    row1Values = row1.values.tolist()
    row2Values = row2.values.tolist()
    jaccardScore = jaccard_similarity(set(row1Values), set(row2Values))
    return jaccardScore >= JACCARD_THRESHOLD


df1 = read_dataset_to_df('./data/dataset1.json', 'df1')
df2 = read_dataset_to_df('./data/dataset2.json', 'df2')
datasetMapping = {
    'df1': df1,
    'df2': df2
}
clusters = clustering.get_clusters(df1, df2)
blocks = blocking.create_blocks_from_clusters(clusters, datasetMapping)
duplicates = get_duplicates(blocks)
for duplicate in duplicates:
    print("************ Duplicate *************")
    print(duplicate['item1'])   
    print("------------------------------------")
    print(duplicate['item2'])
    print("*************************")
