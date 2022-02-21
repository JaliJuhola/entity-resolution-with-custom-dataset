import pandas as pd


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