import pandas as pd
PURGING_THRESHOLD = 5
import warnings
warnings.filterwarnings('ignore')
def common_blocks_scheme(comparisons):
    groupedComparison = comparisons.groupby(by=['mainIndex', 'mainDf', 'secondaryIndex', 'secondaryDf']).size()
    purgedValues = weight_edge_pruning(groupedComparison, groupedComparison.mean())
    return purgedValues

def weight_edge_pruning(groupedComparison, globalMean):
    return groupedComparison.loc[lambda x : x > globalMean]

def get_comparisons_by_item(comparisons, idx, dfName):
    filteredFirstComparisons = comparisons[((comparisons['mainDf'].str.match(dfName) & (comparisons['mainIndex'] == idx)))]
    filteredSecondComparisons = comparisons[(comparisons['secondaryDf'].str.match(dfName) & (comparisons['secondaryIndex'] == idx))]
    return pd.concat([filteredFirstComparisons, filteredSecondComparisons])

def get_comparisons_by_two_items(comparisons, idx1, df1, idx2):
    return comparisons[(((comparisons['mainDf'] == df1) & (comparisons['mainIndex'] == idx1) & (comparisons['secondaryIndex'] == idx2)) | ((comparisons['secondaryDf'] == df1) & (comparisons['secondaryIndex'] == idx1) & (comparisons['mainIndex'] == idx2)))]

def update_column_value_by_comparison(comparisons, weight):
    comparisons.loc[:, 'weight'] = weight

def jaccard_weight_edge_pruning(jaccardComparisons):
    globalMean = jaccardComparisons['weight'].mean()
    return jaccardComparisons.loc[lambda x : x.weight > globalMean]

def jaccard_scheme(comparisons, df1Items, df2Items):
    comparisonsByItem = {}
    comparisons['weight'] = 0
    resultComparisons = pd.DataFrame()
    for itemIdx in df1Items:
        itemsComparisons = get_comparisons_by_item(comparisons, itemIdx['id'], itemIdx['df'])
        primaryBelongsToNumberOfBlocks = itemsComparisons["blockId"].unique().shape[0]
        duplicateClearedRows = itemsComparisons.drop(columns=['blockId']).drop_duplicates(subset=itemsComparisons.columns.difference(['blockId', 'weight']))
        processed = []
        for index, value in duplicateClearedRows.iterrows():
            (secondaryItemIndex, secondaryItemDf) = (value['secondaryIndex'], value['secondaryDf']) if value['mainDf'] == itemIdx['df'] else (value['mainIndex'], value['mainDf'])
            if secondaryItemIndex not in processed:
                processed.append(secondaryItemIndex)
                comparisonsSecondary = None
                if secondaryItemIndex not in comparisonsByItem:
                    comparisonsSecondary = get_comparisons_by_item(comparisons, secondaryItemIndex, secondaryItemDf)
                    comparisonsByItem[secondaryItemIndex] = comparisonsSecondary
                else:
                    comparisonsSecondary = comparisonsByItem[secondaryItemIndex]
                comparisonsByTwoItems = get_comparisons_by_two_items(itemsComparisons, itemIdx['id'], itemIdx['df'], secondaryItemIndex)
                jaccardWeightByPair = (comparisonsByTwoItems.shape[0] / (primaryBelongsToNumberOfBlocks + comparisonsSecondary.shape[0]))
                update_column_value_by_comparison(comparisonsByTwoItems, jaccardWeightByPair)
                if resultComparisons.empty:
                    resultComparisons = comparisonsByTwoItems
                else:
                    resultComparisons = pd.concat([resultComparisons, comparisonsByTwoItems])
    return resultComparisons.drop_duplicates(subset=itemsComparisons.columns.difference(['blockId', 'weight']))

def cardinality_node_pruning(comparisons, df1Items, df2Items):
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
