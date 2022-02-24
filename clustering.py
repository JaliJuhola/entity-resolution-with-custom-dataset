import pandas as pd
import numpy as np

import json
from nltk import ngrams

def jaccard_similarity(series1, series2):
    intersectionOfSeries = np.intersect1d(series1, series2, assume_unique=True)
    unionOfSeries = np.union1d(series1, series2)
    return float(intersectionOfSeries.size / unionOfSeries.size)

def getMostSimilarColumn(column, targetColumnNames, similarities, primary):
    largestSimilarityValue = -1
    largestSimilarityName = None
    for targetColumnName in targetColumnNames:
        similarityName = f"{column}*{targetColumnName}" if primary else f"{targetColumnName}*{column}"
        if similarities[similarityName] > largestSimilarityValue:
            largestSimilarityName = targetColumnName
            largestSimilarityValue = similarities[similarityName]
    return largestSimilarityName

def similarity_between_properties(df1, df1Columns, df2, df2Columns):
    df2ColumnValues = {}
    similarities = {}
    for df1ColumnName in df1Columns:
        df1ColumnValuesList = df1[df1ColumnName].unique()
        for df2ColumnName in df2Columns:
            # Check poor mans cache
            df2ColumnValuesList = None 
            if df2ColumnName in df2ColumnValues:
                df2ColumnValuesList = df2ColumnValues[df2ColumnName]
            else:
                df2ColumnValuesList = df2[df2ColumnName].unique()
                df2ColumnValues[df2ColumnName] = df2ColumnValuesList
            similarity = jaccard_similarity(df1ColumnValuesList, df2ColumnValuesList)
            similarities[f"{df1ColumnName}*{df2ColumnName}"] = similarity
    return similarities

def getMostSimilarPairs(dfColumns, comparisonColumns, similarities, isPrimary):
    mostSimilarPairs = {}
    for dfColumn in dfColumns:
        mostSimilar = getMostSimilarColumn(dfColumn, comparisonColumns, similarities, isPrimary)
        mostSimilarPairs[dfColumn] = mostSimilar
    return mostSimilarPairs

def getClustersWithMostSimilarPairs(mostSimilarPairsDf1, mostSimilarPairsDf2, suffix):
    clusters = []
    already_at_cluster = set()
    secondarySuffix = "df1" if suffix == "df2" else "df2"
    initialIteration = mostSimilarPairsDf1.copy() if suffix == 'df1' else mostSimilarPairsDf2.copy()
    for (startsName, endsName) in initialIteration.items():
        currentSuffix = suffix
        currentCluster = set()
        currentCluster.add(f"{currentSuffix}*{startsName}")
        currentCluster.add(f"{secondarySuffix}*{endsName}")
        currentDf = mostSimilarPairsDf1.copy() if suffix == 'df2' else mostSimilarPairsDf2.copy()
        clusters.append(currentCluster)
    return clusters

def get_clusters(df1, df2):
    df1Columns = df1.columns.tolist()
    df2Columns = df2.columns.tolist()
    similaritiesBetweenProperties = similarity_between_properties(df1, df1Columns, df2, df2Columns)
    mostSimilarPairsDf1 = getMostSimilarPairs(df1Columns, df2Columns, similaritiesBetweenProperties, True)
    mostSimilarPairsDf2 = getMostSimilarPairs(df2Columns, df1Columns, similaritiesBetweenProperties, False)
    df1Df2Cluster = getClustersWithMostSimilarPairs(mostSimilarPairsDf1, mostSimilarPairsDf2, 'df1')
    df2Df1Cluster = getClustersWithMostSimilarPairs(mostSimilarPairsDf1, mostSimilarPairsDf2, 'df2')
    return [*df1Df2Cluster, *df2Df1Cluster]
