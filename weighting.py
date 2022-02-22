import pandas as pd
PURGING_THRESHOLD = 5

def common_blocks_scheme(comparisons):
    groupedComparison = comparisons.groupby(by=['mainIndex', 'mainDf', 'secondaryIndex', 'secondaryDf']).size()
    purgedValues = weight_edge_pruning(groupedComparison)
    return purgedValues


def weight_edge_pruning(groupedComparison):
    return groupedComparison.loc[lambda x : x > PURGING_THRESHOLD]

def cardinality_node_pruning(comparisons, df1Items, df2Items):
    comparisons = comparisons[comparisons['mainIndex'] != comparisons['secondaryIndex']]
    for itemIdx in df1Items:
        indexesToFilter = []
        filteredFirstComparisons = comparisons[((comparisons['mainDf'].str.match(itemIdx['df']) & (comparisons['mainIndex'] == itemIdx['id'])))]
        first_comparisons = filteredFirstComparisons.groupby(by=['secondaryIndex']).size()
        filteredSecondComparisons = comparisons[(comparisons['secondaryDf'].str.match(itemIdx['df']) & (comparisons['secondaryIndex'] == itemIdx['id']))]
        second_comparisons = filteredSecondComparisons.groupby(by=['mainIndex']).size()
        twoWaysValues = second_comparisons.add(first_comparisons)
        localThreshold = twoWaysValues.mean()
        for idx, weight in twoWaysValues.iteritems():
            if weight <= localThreshold:
                indexesToFilter.append(idx)
        comparisons = comparisons[((comparisons['mainDf'] != itemIdx['df']) | (comparisons['mainIndex'] != itemIdx['id']) | (~comparisons['secondaryIndex'].isin(indexesToFilter)))]
        comparisons = comparisons[((comparisons['secondaryDf'] != itemIdx['df']) | (comparisons['secondaryIndex'] != itemIdx['id']) | (~comparisons['mainIndex'].isin(indexesToFilter)))]
    return comparisons
