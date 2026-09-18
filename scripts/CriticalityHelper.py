import statistics
import numpy as np
import pandas as pd
import math
from scipy import signal
import scipy.cluster.hierarchy as spc
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

def filterDFByTime(df, timeName, timeValues):
    return df.loc[:, timeValues == timeName]


def filterDFByGenes(df, genes, negative=False):
    if type(genes) is list:
        if negative:
            return df.loc[~ df.index.isin(genes), :]
        return df.loc[df.index.isin(genes), :]
    return df.loc[genes, :]


def filterDFByGeneExpression(df, gene, threshold=1):
    return df.loc[:, df.loc[gene, :] >= threshold]


def filterDFByMinimumGeneExpression(df, includeCriteria=None, threshold=0):
    df = df if includeCriteria is None else df.loc[:, includeCriteria]
    geneValues = df.mean(axis=1)
    return df.loc[[val > threshold for val in geneValues], :]


def filterDFByGeneVariance(df, threshold=0.01, proportionKept=None):
    variance = df.var(axis=1)
    if proportionKept is not None:
        lowerQuantile = np.quantile(variance, 1 - proportionKept)
        return df.loc[variance > lowerQuantile, :]
    return df.loc[variance > threshold, :]


def filterDFByTimeAndCluster(df, timeName, timeValues, clusterName, clusterValues, negative=False):
    if negative:
        return df.loc[:, np.logical_and(clusterValues != clusterName, timeValues == timeName)]
    return df.loc[:, np.logical_and(clusterValues == clusterName, timeValues == timeName)]


def filterDFByTimeAndGeneExpression(df, gene, timeName, timeValues, threshold=1, negative=False):
    if negative:
        return df.loc[:, np.logical_and(df.loc[gene, :] <= threshold, timeValues == timeName)]
    return df.loc[:, np.logical_and(df.loc[gene, :] > threshold, timeValues == timeName)]


def getClusterCoefficientOfVariation(cluster):
    CVList = []
    for gene in cluster.index:
        geneArray = cluster.loc[gene, :]
        mean = statistics.mean(geneArray)
        sd = np.std(geneArray)
        CVList.append(sd / mean)
    return statistics.mean(CVList)


def getClusterStandardDeviation(cluster, timePoint=None):
    return statistics.mean(cluster.std(axis=1)) # Standard deviation of each row in df, i.e. of each gene


def getInternalCorrelationOneToMany(cluster, gene, timePoint=None):
    correlationList = []
    mainGeneSeries = filterDFByGene(cluster, gene)
    for otherGene in cluster.index:
        if otherGene != gene:
            corr = mainGeneSeries.corr(filterDFByGene(cluster, otherGene))
            if math.isnan(corr):
                continue
            else:
                correlationList.append(abs(corr))

    return correlationList, statistics.mean(correlationList)


def getInternalCorrelationManyToMany(cluster):
    correlationList = []
    geneCount = len(cluster.index)

    for i in range(geneCount):
        gene1Series = filterDFByGenes(cluster, cluster.index[i])
        for j in range(geneCount):
            if i == j:
                # continue
                pass
            gene2Series = filterDFByGenes(cluster, cluster.index[j])

            corr = gene1Series.corr(gene2Series)
            if math.isnan(corr):
                continue
            else:
                correlationList.append(abs(corr))

    # return sum(correlationList) / geneCount
    return sum(correlationList) / geneCount**2


def getExternalCorrelationOneToMany(df, timeName, timeValues, geneOfInterest,
                                     clusterName=None, clusterValues=None,
                                     clusterGene=None, expressionThreshold=1,
                                     varianceThreshold=None):

    if varianceThreshold is not None:
        print("Filtering by gene variance...")
        df = filterDFByGeneVariance(df, varianceThreshold)

    if clusterGene is not None:
        print("Filtering by time and gene expression...")
        inDF = filterDFByTimeAndGeneExpression(df, clusterGene, timeName, timeValues, threshold=expressionThreshold, negative=False)
        outDF = filterDFByTimeAndGeneExpression(df, clusterGene, timeName, timeValues, threshold=expressionThreshold, negative=True)
    elif clusterName is not None and clusterValues is not None:
        print("Filtering by time and cluster...")
        inDF = filterDFByTimeAndCluster(df, timeName, timeValues, clusterName, clusterValues, negative=False)
        outDF = filterDFByTimeAndCluster(df, timeName, timeValues, clusterName, clusterValues, negative=True)
    else:
        print("No cluster selected as input!")
        return None, None

    print("Filtering by gene of interest...")
    mainGeneSeries = filterDFByGene(inDF, geneOfInterest)
    print(mainGeneSeries)
    
    correlationList = []
    i = 0
    print("Calculating correlations...")
    for geneOut in outDF.index:
        # print(geneOut)
        i += 1
        if i % 100 == 0:
            print(i)
        if i > 3:
            break
        geneOutSeries = filterDFByGenes(outDF, geneOut)
        # print(geneOutSeries)
        corr = signal.correlate(mainGeneSeries, geneOutSeries)

        # corr = mainGeneSeries.corr(geneOutSeries)
        print(corr)
        # return [mainGeneSeries, geneOutSeries]
        if not math.isnan(corr):
            correlationList.append(abs(corr))

    return correlationList


def getExternalCorrelationManyToMany(df, timeName, timeValues,
                                     DNB=None,
                                     clusterName=None, clusterValues=None,
                                     gene=None, expressionThreshold=1,
                                     varianceThreshold=None):

    if varianceThreshold is not None:
        print("Filtering by gene variance...")
        df = filterDFByGeneVariance(df, varianceThreshold)

    if DNB is not None:
        print("Filtering by DNB...")
        inDF = filterDFByGenes(df, DNB, negative=False)
        outDF = filterDFByGenes(df, DNB, negative=True)
    elif gene is not None:
        print("Filtering by time and gene expression...")
        inDF = filterDFByTimeAndGeneExpression(df, gene, timeName, timeValues, threshold=expressionThreshold, negative=False)
        outDF = filterDFByTimeAndGeneExpression(df, gene, timeName, timeValues, threshold=expressionThreshold, negative=True)
    elif clusterName is not None and clusterValues is not None:
        print("Filtering by time and cluster...")
        inDF = filterDFByTimeAndCluster(df, timeName, timeValues, clusterName, clusterValues, negative=False)
        outDF = filterDFByTimeAndCluster(df, timeName, timeValues, clusterName, clusterValues, negative=True)
    else:
        print("No cluster selected as input!")
        return None, None

    correlationList = []
    i = 0
    print("Calculating correlations...")
    for geneOut in outDF.index:
        i += 1
        if i % 1000 == 0:
            print(i)

        geneOutSeries = filterDFByGenes(outDF, geneOut)

        for geneIn in inDF.index:
            geneInSeries = filterDFByGenes(inDF, geneIn)
            corr = geneOutSeries.corr(geneInSeries)
            if not math.isnan(corr):
                correlationList.append(abs(corr))

    # return sum(correlationList) / len(df.index)
    return sum(correlationList) / (len(inDF.index) * len(outDF.index))


def getIScore(cluster, clusterList, gene, timePoint=None):
    cov = getClusterCovariance(cluster, timePoint=timePoint)
    internalCorr = getInternalCorrelation(cluster, gene, timePoint=timePoint)
    externalCorr = getExternalCorrelation(clusterList, gene, timePoint=timePoint)
    return cov * internalCorr / externalCorr


def getSummaryValue(df, timeName, timeValues, DNB, varianceThreshold=0.1, summaryType="CI"):
    df = filterDFByTime(df, timeName, timeValues)
    df = filterDFByGeneVariance(df, varianceThreshold)
    cluster = filterDFByGenes(df, DNB)
    if summaryType == "CI":
        toReturn = getClusterStandardDeviation(cluster)
    elif summaryType == "CV":
        toReturn = getClusterCoefficientOfVariation(cluster)
    internalCorr = getInternalCorrelationManyToMany(cluster)
    externalCorr = getExternalCorrelationManyToMany(df, timeName, timeValues, DNB=DNB)
    return [toReturn * internalCorr / externalCorr, toReturn, internalCorr, externalCorr]


def getCellEntropy():
    pass


def getGeneEntropy(df, gene, timeName=None, timeValues=None):
    if timeName is not None and timeValues is not None:
        geneSeries = df.loc[gene, timeValues==timeName]
    else:
        geneSeries = filterDFByGenes(df, gene)

    # totalGeneCount = np.sum(geneSeries)
    # print(totalGeneCount)
    # if totalGeneCount == 0:
    #     return 0
    geneSeriesLength = len(geneSeries)

    hist, bins = np.histogram(geneSeries, bins=geneSeriesLength // 2)
    totalEntropy = 0
    for sample in hist:
        if sample != 0:
            # p = sample / totalGeneCount
            # p = sample / (binCount)
            p = sample / geneSeriesLength
            totalEntropy += math.log2(p) * p
    return -1 * totalEntropy


# Get differential expression table of genes of a cell type against other cell types
def getDifferentiallyExpressedGenes(annObject, differentialColumn, case, individualCompare=False, useRaw=False, includeCriteria=None, basis=None):
    
    if basis is not None:
        diffTableMap = {}
        for comparisonGroup in basis.columns:
            print("Currently comparing against " + comparisonGroup)
            if comparisonGroup != case:
                diff = basis[case] - basis[comparisonGroup]
                diffTableMap[comparisonGroup] = pd.DataFrame({comparisonGroup: diff}, index=basis.index)
        return diffTableMap

    # Copy AnnData and filter
    annObjectCopy = annObject.copy() if basis is None else annObject
    annObjectCopy = annObjectCopy if includeCriteria is None else annObjectCopy[includeCriteria]
    annotations = annObjectCopy.obs[differentialColumn]
    # if annotations.dtype == 'category':
    #     annotations = annotations.cat.remove_unused_categories()
    annotations = annotations.cat.remove_unused_categories() if annotations.dtype == 'category' else annotations

    print("Normalizing...")
    sc.pp.normalize_total(annObjectCopy, inplace=True)
    # annObjectCopy = sc.pp.normalize_total(annObjectCopy, inplace=False)
    sc.pp.log1p(annObjectCopy, copy=False)
    # annObjectCopy = sc.pp.log1p(annObjectCopy, copy=True)
    
    if individualCompare:
        diffTableMap = {}
        for comparisonGroup in set(annotations):
            print("Currently comparing against " + comparisonGroup)
            if comparisonGroup != case:
                filteredAnnObject = annObjectCopy[annotations.isin([case, comparisonGroup])].copy()
                sc.tl.rank_genes_groups(filteredAnnObject, differentialColumn, method='wilcoxon', use_raw=useRaw, copy=False)
                diffTableMap[comparisonGroup] = sc.get.rank_genes_groups_df(filteredAnnObject, group=case).set_index("names")
        del annObjectCopy
        del filteredAnnObject
        print("Done!")
        return diffTableMap

    sc.tl.rank_genes_groups(annObjectCopy, differentialColumn, method='wilcoxon', use_raw=useRaw, copy=False)
    diffTable = sc.get.rank_genes_groups_df(annObjectCopy, group=case).set_index("names")
    # reducedDiff = diffTable.loc[np.logical_and(diffTable['pvals_adj'] < 0.05, abs(diffTable['logfoldchanges']) > 2), :]
    print("Complete")
    return diffTable, annObjectCopy


# Given a DEG table, filter table to genes meeting certain conditions
def getTopGenes(diffTable, minimumChange=0, minimumQuantileChange=None, diffType="scores", maximumPVal=0.05, checkSurface=False, requireOverexpression=False, useBasis=False,
                outFile=None, surfaceGeneFile="/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Transdifferentiation/subcellular_location.tsv", sep="\t"):
    
    # Set conditions for differential expression
    changes = diffTable.iloc[:, 0] if useBasis else diffTable[diffType]
    changes = changes if requireOverexpression else changes.abs()
    if minimumQuantileChange is not None and useBasis:
        minimumChange = np.quantile(changes[changes > 0] if requireOverexpression else changes, minimumQuantileChange)
    conditions = [changes > minimumChange] if useBasis else [diffTable['pvals_adj'] < maximumPVal, changes > minimumChange]

    # Set condition for surface genes
    if checkSurface:
        cellSurfaceGenes = pd.read_csv(surfaceGeneFile, sep=sep)
        conditions.append(diffTable.index.isin(cellSurfaceGenes.loc[cellSurfaceGenes["Main location"] == "Plasma membrane", :]["Gene name"]))

    # Apply all conditions
    combinedCondition = conditions[0]
    for i in range(1, len(conditions)):
        combinedCondition = np.logical_and(combinedCondition, conditions[i])
    filteredGenes = diffTable.loc[combinedCondition, :]

    if outFile is not None:
        filteredGenes.to_csv(outFile)
    return filteredGenes


# Given DEG between a group and each other, get table of genes with high expression and universal high differential expression
def getCombinedTopGenes(diffTableMap, df, includeCriteria=None, expressionThreshold=None, missesAllowed=0, outFile=None, checkSurface=False, requireOverexpression=False, minimumChange=0, minimumQuantileChange=None, diffType="scores", maximumPVal=0.05, useBasis=False, secondDF=None):
    if includeCriteria is not None: # Usually filter to the case, as in annotations == target
        df = df.loc[:, includeCriteria]

    allGenes = pd.DataFrame(index=df.index)
    comparisonGroups = list(diffTableMap.keys())
    for group in comparisonGroups:
        validGenes = getTopGenes(diffTableMap[group], minimumChange=minimumChange, minimumQuantileChange=minimumQuantileChange, diffType=diffType, maximumPVal=maximumPVal, 
                                 checkSurface=checkSurface, requireOverexpression=requireOverexpression, useBasis=useBasis).index
        valid = [int(val in validGenes) for val in df.index]
        allGenes[group] = valid

    # allGenes.set_index("Gene", inplace=True)
    totalValid = allGenes.sum(axis=1)
    allGenes["Successes"] = totalValid

    filteredGenes = df.loc[allGenes["Successes"] > len(comparisonGroups) - missesAllowed - 1, :]
    filteredGenes = filteredGenes if expressionThreshold is None else filterDFByMinimumGeneExpression(filteredGenes, threshold=expressionThreshold)
    filteredGenes = filteredGenes if secondDF is None else filteredGenes[filterDFByMinimumGeneExpression(secondDF[filteredGenes.index], threshold=expressionThreshold).index]
    targetDF = pd.DataFrame({"Average Normalized Expression": filteredGenes.mean(axis=1)}, index=filteredGenes.index)
    for group in comparisonGroups:
        DEG = diffTableMap[group][group] if useBasis else diffTableMap[group].loc[:, diffType].copy().rename(diffType + " " + group)
        targetDF = targetDF.join(DEG, how="left", rsuffix=group)

    if missesAllowed > 0:
        targetDF = targetDF.join(allGenes["Successes"], how="left")
    if outFile is not None:
        targetDF.to_csv(outFile)

    return targetDF


# Get a pd df containing the pathways and genes of a specified set of comparisons
def getPathwaySummary(pathways, comparisons=None, states=None, names=None, simplified=False, significance=0.1, tolerance=0.7, outFile=None):

    # Initialize key objects
    comparisons = comparisons or pathways.keys()
    states = states or set([state for state in pathways[comparison].keys() for comparison in comparisons])
    names = names or set([name for comparison in comparisons for state in states if state in pathways[comparison].keys() for name in pathways[comparison][state].keys()])
    pathwaysMap, pathwayHits, identifierSet = ({}, {}, set())
    
    # Create a map of the genes and significances of each comparison of each pathway
    for comparison in comparisons:  # Case vs control
        for state in states:  # Cell identities being assessed
            if state not in pathways[comparison].keys():
                continue
            for name in names:  # Datasets included
                if name not in pathways[comparison][state].keys():
                    continue
                pathwayFrame, identifier = (pathways[comparison][state][name], name + " " + comparison + " " + state)
                identifierSet.add(identifier)

                for pathway in pathwayFrame.index:  # Pathway being evaluated for significance
                    
                    # Initialize pathway hits (successes) in global map
                    FDRVal, FWERVal = (pathwayFrame.loc[pathway, "FDR q-val"], pathwayFrame.loc[pathway, "FWER p-val"])
                    if pathway not in pathwaysMap.keys():
                        pathwaysMap[pathway], pathwayHits[pathway] = ({}, {})
                        pathwayHits[pathway]["Hits"] = 0
                        if not simplified:
                            pathwayHits[pathway]["FWER Hits"] = 0
                            
                    # Register hit if metric below significance
                    if FDRVal < significance:
                        pathwayHits[pathway]["Hits"] += 1
                    pathwaysMap[pathway][identifier + " FDR q-val"] = FDRVal
                    if not simplified:
                        pathwaysMap[pathway][identifier + " FWER p-val"] = FWERVal
                        if FWERVal < significance:
                            pathwayHits[pathway]["FWER Hits"] += 1
    
                    # Record genes in pathway and pathway significance
                    genes = list(pathwayFrame["Lead_genes"][pathwayFrame.index == pathway])[0]
                    pathwaysMap[pathway][identifier] = genes.split(";")
    
    # Find the genes that appeared most often in successful pathways  
    topFDRGenes, topFWERGenes = ([], [])
    for pathway in pathwayHits.keys():  # Pathway being assessed for best genes
        geneHits = {}
        geneFWERHits = {}
        for identifier in identifierSet:  # The combination of name, comparison, and state
    
            # Comparisons that didn't find the pathway don't display anything
            if identifier not in pathwaysMap[pathway].keys():
                pathwaysMap[pathway][identifier] = pathwaysMap[pathway][identifier + " FDR q-val"] = ""
                if not simplified:
                    pathwaysMap[pathway][identifier + " FWER p-val"] = ""

            # Increment the number of successes for each gene in the pathway if the comparison was significant
            else:
                validFDR, validFWER = (pathwaysMap[pathway][identifier + " FDR q-val"] < significance, (not simplified) and (pathwaysMap[pathway][identifier + " FWER p-val"] < significance))
                genes = pathwaysMap[pathway][identifier]
                for gene in genes:
                    if validFDR:
                        geneHits[gene] = 1 if gene not in geneHits.keys() else geneHits[gene] + 1
                    if validFWER:
                        geneFWERHits[gene] = 1 if gene not in geneFWERHits.keys() else geneFWERHits[gene] + 1
    
        # Record genes that appeared close to the max number of times in the pathway when significant 
        maxFDRVal, maxFWERVal = (max(list(geneHits.values()) + [0.9]), None if simplified else max(list(geneFWERHits.values()) + [0.9]))  # Value between 0-1 needed so max runs when nothing is significant
        FDRGenes, FWERGenes = ([gene for gene in geneHits.keys() if geneHits[gene] >= maxFDRVal * tolerance], [] if simplified else [gene for gene in geneFWERHits.keys() if geneFWERHits[gene] >= maxFWERVal * tolerance])
        topFDRGenes.append(FDRGenes)
        topFWERGenes.append(FWERGenes)
    
    # Convert map of pathway information to dataframe
    pathwaysSummary = pd.DataFrame.from_dict(pathwaysMap, orient="index")
    sortedColumns = sorted(pathwaysSummary.columns) # Track these columns so they are ordered together and in the middle
    hitsFrame = pd.DataFrame.from_dict(pathwayHits, orient="index")
    pathwaysSummary[hitsFrame.columns] = hitsFrame
    pathwaysSummary = pathwaysSummary[list(hitsFrame.columns) + sortedColumns]
    pathwaysSummary["Top Genes"] = topFDRGenes
    if not simplified:
        pathwaysSummary["Top FWER Genes"] = topFWERGenes
    
    pathwaysSummary = pathwaysSummary.sort_values("Hits" if simplified else "FWER Hits", ascending=False)
    pathwaysSummary.index.name = "Pathway"
    if outFile is not None:
        pathwaysSummary.to_csv(outFile)
    return pathwaysSummary


def clusterGenesByCorrelation(df):
    # Get correlation "distances"
    corr = df.T.corr().values
    pdist_uncondensed = 1.0 - abs(corr)
    pdist_condensed = np.concatenate([row[i+1:] for i, row in enumerate(pdist_uncondensed)])

    # Cluster based on these distances
    linkage = spc.linkage(pdist_condensed, method='complete')
    idx = spc.fcluster(linkage, 0.5 * pdist_condensed.max(), 'distance')

    # Create map of clusters to their genes
    clusterDict = {}
    for i in range(len(idx)):
        cluster = int(idx[i])
        gene = df.index.iloc[i]
        if cluster not in clusterDict.keys():
            clusterDict[cluster] = [gene]
        else:
            clusterDict[cluster].append(gene)
    return clusterDict


def getDominantGroups(df, clustersList, timeValues, timesSorted, differentialColumn, case, control):

    print("Normalizing...")
    normalizedDF = df.copy()
    for gene in df.index:
        geneControl = df.loc[df.index == gene, differentialColumn == control]
        geneCase = df.loc[df.index == gene, differentialColumn == case]
        meanControl = statistics.mean(geneControl)
        sdControl = np.std(geneControl)
        normalizedDF.loc[normalizedDF.index == gene, :] = (geneCase - meanControl) / sdControl

    print("Finding DNBs...")
    clusterValues = {}
    for time in timesSorted:
        timeDF = filterDFByTime(normalizedDF, time, timeValues)
        clusterValues[time] = {}
        for clusterList in clustersList:
            # Ideally screen for requirements first or also
            if len(clusterList) > 2:
                for cluster in clusterList:
                    genes = clusterList[cluster]
                    clusterValues[time][genes] = getSummaryValue(timeDF, time, timeValues, genes, summaryType="CI")

    return clusterValues


def findDNB(annObject, timeColumnName, timesSorted, differentialColumn, case, control):
    annObjectCopy = annObject.copy()
    metadata = annObjectCopy.obs
    annObjectCopy = annObjectCopy[np.logical_or(metadata[differentialColumn] == case, metadata[differentialColumn] == control)]
    metadata = annObjectCopy.obs
    df = annObjectCopy.to_df().T

    print("Clustering...")
    clustersList = []
    for time in timesSorted:
        print(time)
        timeDF = df.loc[:, metadata[timeColumnName] == time]
        timeAnnObject = annObjectCopy.copy()
        timeAnnObject = timeAnnObject[timeDF.columns, :]
        timeDF = timeAnnObject.to_df().T
        highVarianceGenes = timeDF.loc[timeDF.var(axis=1) > 0.01, :].index
        timeAnnObject = timeAnnObject[:, list(highVarianceGenes)] # Filter by variance
        timeDF = timeAnnObject.to_df().T
        genesOfInterest = getDifferentiallyExpressedGenes(timeAnnObject, differentialColumn, case)
        clustersList.append(clusterGenesByCorrelation(timeDF.loc[timeDF.index.isin(genesOfInterest), :]))

    print("Finding DNB...")
    rankedGroups = getDominantGroups(df, clustersList, metadata[timeColumnName], timesSorted, differentialColumn, case, control)
    print("Done!")
    return rankedGroups


# Get distance of each point to mean of cluster 
def getClusterDistances(df):
    mean = df.mean(axis=1)
    distances = np.linalg.norm(df.values - np.tile(mean, (len(df.columns), 1)).T, axis=0)
    return distances, mean


# Filter dataframe of n-dimensional points down to x % of data closest to the mean 
def getQuantileData(df, includeCriteria=None, quantile=0.8, quantileMin=None, minSamples=50, metric="cosine"):
    if includeCriteria is not None:
        df = df.loc[:, includeCriteria]
    if len(df.columns) < minSamples:
        return None
    if metric == "mean":
        distances, mean = getClusterDistances(df)
    else:
        distances = np.mean(cdist(df.values.T, df.values.T, metric=metric), axis=1)

    upperQuantile = np.quantile(distances, quantile)
    lowerQuantile = 0 if quantileMin is None else np.quantile(distances, quantileMin)
    return df.loc[:, np.logical_and(distances < upperQuantile, distances > lowerQuantile)]


def getDistance(df1, df2, method="min", metric="cosine"):
    distances = cdist(df1.values.T, df2.values.T, metric=metric)
    if method == "min":
        return np.min(distances), np.unravel_index(np.argmin(distances), distances.shape)
    elif method == "mean":
        return np.mean(distances), (0, 0)


# Get DataFrame of minimal distances between clusters given dict of cluster names to points
def getDistanceMap(clusterMap, method="min", metric="cosine"):

    # Initialize dict of distances between states
    stateDistancesMap = {}
    stateList = sorted(clusterMap.keys())
    stateCount = len(stateList)
    for state in stateList:
        stateDistancesMap[state] = {}
        stateDistancesMap[state][state] = 0

    # Set distance between each pair of clusters
    for i in range(0, stateCount - 1):
        state1 = stateList[i]
        print("Getting distances for state " + state1)
        for j in range(i + 1, stateCount):
            state2 = stateList[j]
            stateDistancesMap[state1][state2] = stateDistancesMap[state2][state1] = getDistance(clusterMap[state1], clusterMap[state2], method=method, metric=metric)[0]
    distances = pd.DataFrame.from_dict(stateDistancesMap, orient="index")
    
    return distances


# Using state clusters filtered by quantiles, plot matrix of distances between clusters
def stateDistancePlot(topObject, projectionName, alternateDF=None, includeCriteria=None, quantile=0.8, quantileMin=None, minSamples=50, 
                      figX=12, figY=12, title="", outFile=None, method="min", metric="cosine", getNewDistances=False, axisFontSize=16):

    # If not recalculating distances, simply use existing ones
    if hasattr(topObject, "distances") and not getNewDistances:
        distances = topObject.distances
    else:
        # Get quantiles
        stateCloseValuesMap = {}
        df = topObject.projections[projectionName] if alternateDF is None else alternateDF
        df = df if includeCriteria is None else df.loc[:, includeCriteria]
        for state in topObject.sortedCellTypes:
            reducedState = getQuantileData(df, quantile=quantile, quantileMin=quantileMin, metric=metric, minSamples=minSamples, includeCriteria=topObject.annotations == state)
            if reducedState is not None:
                stateCloseValuesMap[state] = reducedState
    
        # Get distances
        distances = getDistanceMap(stateCloseValuesMap, method=method, metric=metric)
        topObject.distances = distances
    
    # Plot results
    plt.subplots(1, 1, figsize=(figX, figY))
    labels = sorted(distances.keys())
    ax = sns.heatmap(distances, annot=True, fmt=".2f", cmap='pink', xticklabels=labels, yticklabels=labels,
        annot_kws={"size": axisFontSize // 1.2}, cbar=True)
    plt.xticks(rotation=90, fontsize=axisFontSize // 1.1)
    plt.yticks(rotation=0, fontsize=axisFontSize // 1.1)
    plt.title(title, fontsize=axisFontSize)
    ax.figure.axes[-1].tick_params(labelsize=axisFontSize // 1.2)
    plt.tight_layout()
    if outFile is not None:
        plt.savefig(outFile, bbox_inches='tight', dpi=300)
    plt.show()


# Plot average distance to centroid of each cluster
def selfMeanDistancePlot(clusterMap):
    stateList = sorted(clusterMap.keys())
    meanDistances = []
    for state in stateList:
        distances = getClusterDistances(clusterMap[state])[0]
        meanDistances.append(sum(distances) / len(distances))
    plt.bar(stateList, meanDistances)
    plt.xticks(rotation=90)
    plt.show()


def getPCA(df):
    pca = PCA(100)
    return pca.fit_transform(df)


