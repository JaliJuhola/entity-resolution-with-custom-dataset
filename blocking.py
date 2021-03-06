import pandas as pd
import similarity

def tokenize_column(df, column):
    df['tokens'] = df.apply(lambda row: ["".join(k) for k in row[column].split()], axis=1)
    return df

def create_blocks_from_clusters(clusters, datasetMapping):
    clusterBlocks = []
    for cluster in clusters:
        entitiesInCluster = pd.DataFrame()
        for propertyNameWithDfInfo in cluster:
            splittedProperty = propertyNameWithDfInfo.split("*")
            rowTokenized = tokenize_column(datasetMapping[splittedProperty[0]], splittedProperty[1])
            explodedRow = rowTokenized.explode('tokens')
            if entitiesInCluster.empty:
                entitiesInCluster = explodedRow
            else:
                entitiesInCluster = entitiesInCluster.reset_index().merge(explodedRow.reset_index(), how='outer').set_index('index')
        groupedBlock = group_by_blocks(entitiesInCluster)
        clusterBlocks.append(groupedBlock)
    return clusterBlocks

def group_by_blocks(initialBlock):
    grouped = initialBlock.groupby(by=["tokens"], dropna=False)
    return grouped

def process_block_by_block_name(block, already_compared, duplicates):
    total_comparisons = 0
    for mainIndex, mainRow in block.iterrows():
        for secondaryIndex, secondaryRow in block.iterrows():
            compared_id = f"{mainRow['dataset_name']}-{mainIndex}*{secondaryRow['dataset_name']}-{secondaryIndex}"
            reverse_compared_id = f"{secondaryRow['dataset_name']}-{secondaryIndex}*{mainRow['dataset_name']}-{mainIndex}"
            if mainRow['dataset_name'] != secondaryRow['dataset_name'] and compared_id not in already_compared:
                already_compared.add(compared_id)
                already_compared.add(reverse_compared_id)
                already_compared.add(f"{secondaryRow['dataset_name']}-{secondaryIndex}*{mainRow['dataset_name']}-{mainIndex}")
                mainRowDuplicateCleaned = mainRow.drop(labels=['dataset_name', 'tokens']).dropna()
                secondaryRowDuplicateCleaned = secondaryRow.drop(labels=['dataset_name', 'tokens']).dropna()
                total_comparisons = total_comparisons + 1
                if(similarity.is_duplicate(mainRow, secondaryRow)):
                    duplicates.append({'item1': mainRowDuplicateCleaned, 'item2': secondaryRowDuplicateCleaned})
    return total_comparisons

def get_duplicates(setOfClustersOfEntities):
    already_compared = set()
    duplicates = []
    total_comparisons = 0
    for clusterOrEntities in setOfClustersOfEntities:
        for blockName in clusterOrEntities.groups:
            if blockName not in ['df1', 'df2']:
                block = clusterOrEntities.get_group(blockName)
                if len(block.index) > 1:
                   total_comparisons += process_block_by_block_name(block, already_compared, duplicates)
    return (duplicates, total_comparisons)

def get_comparisons_according_to_entity_clustering(setOfClusters):
    comparisons = pd.DataFrame()
    blockId = 0
    for cluster in setOfClusters:
        for blockName in cluster.groups:
            if blockName not in ['df1', 'df2']:
                block = cluster.get_group(blockName)
                if len(block.index) > 1:
                    blockId = blockId + 1
                    iteratedComparisons = iterate_block_to_comparisons(block, blockId)
                    if comparisons.empty:
                        comparisons = iteratedComparisons
                    else:
                        comparisons = pd.concat([comparisons, iteratedComparisons])
    return comparisons

def iterate_block_to_comparisons(block, blockId):
    comparisons = []
    for mainIndex, mainRow in block.iterrows():
        for secondaryIndex, secondaryRow in block.iterrows():
            if mainRow['dataset_name'] != secondaryRow['dataset_name']:
                comparisons.append([mainIndex, mainRow['dataset_name'], secondaryIndex, secondaryRow['dataset_name'], blockId])
    return pd.DataFrame(comparisons, columns=['mainIndex', 'mainDf', 'secondaryIndex', 'secondaryDf', 'blockId'])