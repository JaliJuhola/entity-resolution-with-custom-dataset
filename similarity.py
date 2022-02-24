import itertools
import numpy as np
import pandas as pd

JACCARD_THRESHOLD = 0.5

def jaccard_similarity(list1, list2):
    intersectionOfLists = list1.intersection(list2)
    unionOfSeries = list1.union(list2)
    return float(len(intersectionOfLists) / len(unionOfSeries))

def flatten(inputList):
    return list(itertools.chain(*inputList))

def cleanAndExplodeSeries(series):
    cleanedSeries = series.drop(labels=['dataset_name', 'tokens']).dropna().apply(lambda row : row.split())
    return flatten(cleanedSeries.values.tolist())

def is_duplicate(row1, row2):
    row1 = cleanAndExplodeSeries(row1)
    row2 = cleanAndExplodeSeries(row2)
    jaccardScore = jaccard_similarity(set(row1), set(row2))
    return jaccardScore >= JACCARD_THRESHOLD

def get_duplicates_by_comparisons(df1, df2, comparisons, datasetMapping):
    duplicates = []
    already_compared = set()
    totalComparisons = 0
    for index, value in comparisons.iteritems():
        firstItemDf = datasetMapping[index[1]]
        secondItemDf = datasetMapping[index[3]]
        firstItem = firstItemDf[firstItemDf.index==index[0]].drop(columns=['dataset_name', 'tokens'])
        secondItem = secondItemDf[secondItemDf.index==index[2]].drop(columns=['dataset_name', 'tokens'])
        mainItemCompared = f"{index[0]}-{index[1]}*{index[2]}*{index[3]}"
        if mainItemCompared not in already_compared:
            totalComparisons = totalComparisons + 1
            secondItemCompared = f"{index[2]}-{index[3]}*{index[0]}*{index[1]}"
            already_compared.add(mainItemCompared)
            already_compared.add(secondItemCompared)
            firstItemTokens = flatten([item.split() for item in firstItem.values[0]])
            secondItemTokens = flatten([item.split() for item in secondItem.values[0]])
            similarity = jaccard_similarity(set(firstItemTokens), set(secondItemTokens))
            if similarity > JACCARD_THRESHOLD:
                duplicates.append(pd.concat([firstItem, secondItem]))
    return (totalComparisons, duplicates)

def compare_jaccard_results(df1, df2, comparisons, datasetMapping):
    duplicates = []
    totalComparisons = 0
    already_compared = set()
    for index, value in comparisons.iterrows():
        firstItemDf = datasetMapping[value['mainDf']]
        secondItemDf = datasetMapping[value['secondaryDf']]
        firstItem = firstItemDf[firstItemDf.index==value['mainIndex']].drop(columns=['dataset_name', 'tokens'])
        secondItem = secondItemDf[secondItemDf.index==value['secondaryIndex']].drop(columns=['dataset_name', 'tokens'])
        mainItemCompared = f"{value['mainDf']}-{value['mainIndex']}*{value['secondaryDf']}*{value['secondaryIndex']}"
        if mainItemCompared not in already_compared:
            totalComparisons = totalComparisons + 1
            secondItemCompared = f"{value['secondaryDf']}-{value['secondaryIndex']}*{value['mainDf']}*{value['mainIndex']}"
            already_compared.add(mainItemCompared)
            already_compared.add(secondItemCompared)
            firstItemTokens = flatten([item.split() for item in firstItem.values[0]])
            secondItemTokens = flatten([item.split() for item in secondItem.values[0]])
            similarity = jaccard_similarity(set(firstItemTokens), set(secondItemTokens))
            if similarity > JACCARD_THRESHOLD:
                duplicates.append(pd.concat([firstItem, secondItem]))
    return (totalComparisons, duplicates)
