"""
This is a python version of this function:
https://github.com/yeatmanlab/AFQ/blob/master/functions/AFQ_MultiCompCorrection.m
"""

import random
import numpy as np
import scipy.stats


def get_significant_areas(pvals, clusterFWE, alpha=0.05):
    """
    Mark clusters of size clusterFWE of consecutive values smaller than alpha with 1. All other will be 0.
    Used for plotting significant areas.
    """
    result = []
    ctr = 0
    for i in range(len(pvals)):
        p = pvals[i]
        if p > alpha:
            if ctr > 0:
                # cluster was not big enough -> append as many 0 as cluster had elements
                result += [0] * ctr
            ctr = 0
            result.append(0)
        else:
            ctr += 1
            if ctr >= clusterFWE:
                # cluster is big enough and the next element would end the cluster to is end of array -> add cluster
                # to results
                if i == len(pvals) - 1 or pvals[i + 1] > alpha:
                    result += [1] * ctr
                    ctr = 0

        # Array ends, but still elements in ctr (cluster started, but was not big enough before array ended)
        if i == len(pvals) - 1 and ctr > 0:
            result += [0] * ctr

    return np.array(result)


def _corr(a, b):
    """
    Correlate a with each row of b

    Args:
        a: 1d array
        b: 2d array

    Returns:
        c: 1d array with correlations
        p: 1d array with p-values
    """
    b = b.T
    c = []
    p = []
    for i in range(len(b)):
        c_i, p_i = scipy.stats.pearsonr(a, b[i])
        c.append(c_i)
        p.append(p_i)
    return c, p


def AFQ_MultiCompCorrection(data=None, y=None, alpha=0.05, cThresh=None, nperm=1000):
    """
    Compute a multiple comparison correction for Tract Profile data

    This is an implementation of the permutation method described by Nichols
    and Holmes (2001). Nonparametric permutation tests for functional
    neuroimaging: A primer with examples. Human Brain Mapping.  This will
    return the faily wise error (FWE) corrected alpha value for pointwise
    comparisons.  It will also compute the FWE corrected cluster size at the
    user defined alpha.  This means that significant clusters of this size or
    greater are pass the multiple comparison threshold and do not need
    further p-value adjustment.

    Written by Jason D. Yeatman, August 2012
    Ported to python by Jakob Wasserthal, September 2019

    Args:
        data:  Either a matrix of data for a single tract, or a matrix of data
               for all the tracts combined.
        y:     A vector of either behavioral measurements or a binary
               grouping variable for which pointwise statistics will be
               computed on the Tract Profile and the p-value adjusted for
               mulltiple comparisons will be determined.  If y is a
               continuous variable then correlations will be computed. If y
               is a binary vector then T-tests will be computed.
        alpha: The desired alpha (pvalue) to adjust
        cThresh: For clusterwise corrections the threshold for computing a
                 cluster can be different than the desired alpha. For example
                 you can set a cluster threshold of 0.01 and then find clusters
                 that a large enough to pass FWE at a threshold of 0.05.
        nperm: number of permutations

    Returns:
        alphaFWE: This is the alpha (p value) that corresponds after adjustment
                  for multiple comparisons
        statFWE:  This is the value of the statistic corresponding to alphaFWE.
                  statFWE will either be a correlation coeficient or T-statistic
        clusterFWE: Clusters of points on a Tract Profile that are larger than
                    clusterFWE are significant at pvalue = alpha.
        stats:    A structure containing the results of each permutation

    There are two ways how to use these results:
    - p-values below alphaFWE are considered significant with multiple comparisons correction.
    - A cluster (of at least size clusterFWE) with  p-values below alpha are considered significant with multiple
    comparisons correction.
    """

    if cThresh is None:
        cThresh = alpha

    # If y is continues perform a correlation if binary perform a ttest
    if y is None or len(y) == 0:
        y = np.random.randn(data.shape[0], 1)
        print('No behavioral data provided so randn will be used')
        stattest = 'corr'
    else:
        if len(y) == np.sum((y == np.logical_or(0, y)) == 1) or len(y) == np.sum((y == np.logical_or(1, y)) == 2):
            stattest = 'ttest'
        else:
            stattest = 'corr'

    # print("using stattest: {}".format(stattest))

    p = np.zeros([nperm, data.shape[1]])
    stat = np.zeros([nperm, data.shape[1]])
    clusMax = np.zeros([nperm])
    stats = {}

    if ('corr') == (stattest):
        for ii in range(nperm):
            # Shuffle the rows of the data
            rows = np.array(random.sample(range(len(y)), len(y)))  # random shuffling of row indices
            stat[ii, :], p[ii, :] = _corr(y, data[rows, :])
    else:
        if ('ttest') == (stattest):
            for ii in range(nperm):
                rows = np.array(random.sample(list(y), len(y)))
                rows = rows > 0  # to bool
                ttest_res = scipy.stats.ttest_ind(data[rows, :], data[~rows, :])   #independent t-test
                p[ii, :] = ttest_res.pvalue
                stat[ii, :] = ttest_res.statistic

    # Sort the pvals and associated statistics such that the first
    # entry is the most significant
    stats["pMin"] = np.sort(p.min(axis=1))
    stats["statMax"] = np.sort(stat.max(axis=1))[::-1]
    alphaFWE = stats["pMin"][int(round(alpha*nperm))]
    statFWE = stats["statMax"][int(round(alpha*nperm))]

    # If a cluster size is defined, also determine the significant
    # cluster size at the specified alpha value
    # Threshold the pvalue
    pThresh = p < cThresh
    pThresh = np.array(pThresh)

    for ii in range(nperm):
        # Find indices where significant clusters end.
        # The method used requires significant p-values to be included
        # between non-significant p-values. 0 are therefore added at
        # both ends of the thresholded p-value vector
        # (for cases when significant p-values are located at its ends)
        pThresh_ii = [0] + list(pThresh[ii, :].astype(np.uint8)) + [0]
        pThresh_ii = np.array(pThresh_ii)
        clusEnd = np.where(pThresh_ii == 0)[0]
        clusSiz = np.diff(clusEnd)
        clusMax[ii] = clusSiz.max()
    # Sort the clusters in descending order of significance
    stats["clusMax"] = np.sort(clusMax)[::-1]
    clusterFWE = stats["clusMax"][int(round(alpha*nperm))]

    return alphaFWE, statFWE, clusterFWE, stats


# if __name__ == '__main__':
#     data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 4, 2, 3, 5],
#                      [1, 4, 2, 9, 5], [5, 4, 2, 9, 5], [5, 4, 2, 9, 1]])
#     y = np.array([0.3, 1.2, 1.5, 0.1, 0.2, 1.9]).T
#
#     alphaFWE, statFWE, clusterFWE, stats = AFQ_MultiCompCorrection(data, y)
#
#     print(alphaFWE)
#     print(statFWE)
#     print(clusterFWE)
