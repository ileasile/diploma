# -*- coding: utf-8 -*-
"""
f final matches is ', len(matches)


@author: sn
"""


from scipy.special import comb
import numpy as np
import MDMST.SGMFunctions as sgm
import time
import pickle as pkl
# import imp
# imp.reload(sgm)

np.random.seed(5)

#nTrials = 10
#QSizeList = [5, 10, 15]
#ASizeList = [100, 500, 1000]
#pConnList = [.1, .3, .5]
#nVarList = [5,10,15]

# nTrials = 20
nTrials = 1
QSizeList = [5]
# ASizeList = [100]
ASizeList = [10000]
pConnList = [.02]
nVarList = [100]
nTargetsList = [10]
insertCliques = True
insertClutter = True
clutterNumEdgesRemoved = 1

resultsDict = {}

def PrintMatches(matches):
    # Print out matches
    print('Number of final matches is ', len(matches))
    for match in matches:
        print(match)

def PrintGroundTruth(targetSolutions, nTargets):
    # Print out ground truth
    print('# Target Subgraphs: ', nTargets)
    i = 0
    for target in targetSolutions:
        if not target['isClutter']:
            i += 1
            print('Target Subgraph', i)
            print(target['nodes'])
            print(target['edges'])
    if insertClutter:
        print('# Clutter Subgraphs: ', nTargets)
        i = 0
        for target in targetSolutions:
            if target['isClutter']:
                i += 1
                print('Clutter Subgraph ', i)
                print(target['nodes'])
                print(target['edges'])

def ComputeMetrics(targetSolutions, nTargets, matches):

    trueMatchThreshold = 0.5   # within-match minimum precision to be considered hard match for recall
    trueMatches = 0.0; falseMatches = 0.0;
    unpairedMatches = 0.0;
    trueNodeMatches = 0.0; falseNodeMatches = 0.0
    trueEdgeMatches = 0.0; falseEdgeMatches = 0.0
    numTargetNodes = 0.0; numTargetEdges = 0.0
    numMatchNodes = 0.0; numMatchEdges = 0.0
    for match in matches:
        # Greedily pair each match to the best available target
        bestScore = 0
        bestTarget = None
        bestTN = 0
        bestTE = 0
        bestFN = 0
        bestFE = 0
        mNodes = list()
        mEdges = list()
        for val in list(match.values()):
            if isinstance(val, int) or np.issubdtype(type(val), np.integer):
                mNodes.append(val)
            else:
                mEdges.append(val)
        for target in targetSolutions:
            if not (('match' in target) | target['isClutter']):
                tNodes = target['nodes']
                tEdges = target['edges']
                # Compare nodes
                tn = 0.0; te = 0.0; fn = 0.0; fe = 0.0
                for tNode in tNodes:
                    if tNode in mNodes:
                        tn += 1.0
                    else:
                        fn += 1.0
                # Compare edges - todo check if order matters
                found = False
                for tEdge in tEdges:
                    for mEdge in mEdges:
                        if (tEdge[0] == mEdge[0]) & (tEdge[1] == mEdge[1]):
                            te += 1.0
                            found = True
                if not found:
                    fe += 1.0
                score = (tn / len(tNodes)) + (te / len(tEdges)) * 0.5
                if score > bestScore:
                    bestScore = score
                    bestTarget = target
                    bestTN = tn; bestTE = te; bestFN = fn; bestFE = fe
        if bestTarget != None:
            bestTarget['match'] = match
            bestTarget['matchScore'] = bestScore
            trueNodeMatches += bestTN
            trueEdgeMatches += bestTE
            falseNodeMatches += bestFN
            falseEdgeMatches += bestFE
            for val in list(match.values()):
                if isinstance( val, int ):
                    numMatchNodes += 1
                else:
                    numMatchEdges += 1
            numTargetNodes += len(tNodes)
            numTargetEdges += len(tEdges)
        else:
            falseNodeMatches += len(mNodes)
            falseEdgeMatches += len(mEdges)
            unpairedMatches += 1

    # Compute hard recall, precision, using a threshold
    paired = 0.0
    for target in targetSolutions:
        if not target['isClutter']:
            if 'match' in target:
                paired += 1.0
                if target['matchScore'] >= trueMatchThreshold:
                    trueMatches += 1.0
                else:
                    falseMatches += 1.0
            else:
                # Don't forget to include unpaired targets in these counts
                numTargetNodes += len(target['nodes'])
                numTargetEdges += len(target['edges'])
    hardRecall = trueMatches / nTargets
    hardPrecision = trueMatches / len(matches)
    softNodeRecall = trueNodeMatches / numTargetNodes
    softNodePrecision = trueNodeMatches / numMatchNodes
    softEdgeRecall = trueEdgeMatches / numTargetEdges
    softEdgePrecision = trueEdgeMatches / numMatchEdges
    softRecall = (softNodeRecall + softEdgeRecall) * 0.5
    softPrecision = (softNodePrecision + softEdgePrecision) * 0.5

    # Print the results
    print()
    print('Number of Targets: ', nTargets)
    print('True matches: ', trueMatches, '   False matches: ', falseMatches, '   Unpaired matches; ', unpairedMatches)
    print()
    print('Recall: ' , hardRecall, '    Precision: ', hardPrecision)
    print('SoftRecall: ' , softRecall, '    SoftPrecision: ', softPrecision)
    print()
    print('SoftNodeRecall: ' , softNodeRecall, '    SoftNodePrecision: ', softNodePrecision)
    print('SoftEdgeRecall: ' , softEdgeRecall, '    SoftEdgePrecision: ', softEdgePrecision)
    print()

def main():
    for ASize in ASizeList:
        for pConn in pConnList:
            for nVars in nVarList:
                for nTargets in nTargetsList:
                    with open('ResultsAdditional.data', 'wb') as filn:
                        pkl.dump(resultsDict, filn)

                    for trialNum in range(nTrials):
                        if trialNum==1:
                            stophere = 1
                        #Generate A.
                        print('Trial '+str(trialNum) + ' A size:' + str(ASize) + ' pConn: ' + str(pConn) + ' nVars: ' + str(nVars))
                        print('Generating A')
                        start = time.time()
                        nodeDist = np.random.rand(nVars)
                        nodeDist = nodeDist/np.sum(nodeDist)
                        edgeDist = nodeDist
                        A = sgm.GenRandomGraph(ASize, pConn) #Generate a graph
                        A = sgm.AddNodeAttributes(A, nodeDist, 'nValue') #Add Node Attributes
                        A = sgm.AddEdgeAttributes(A, edgeDist, 'eValue') #Add Edge Attributes
                        print('Done at '+str(time.time()-start))
                        print('Printing A')
    #                    sgm.PrintGraph(A)

                        for QSize in QSizeList:
                            #Figure out if we have this trio in the results dictionary.
                            if (ASize, pConn, nVars, QSize) not in list(resultsDict.keys()):
                                resultsDict[(ASize, pConn, nVars, QSize)] = list()
                            print('Grabbing Q')

                            targetSolutions = list()

                            if insertCliques:
                                Q = sgm.insertCliqueSubgraph(A, QSize, edgeDist, 'eValue', targetSolutions)
                            else:
                                Q = sgm.getRandomSubgraph(A, QSize)
                            # njp begin
                            print('Printing Q')
                            sgm.PrintGraph(Q)
                            sgm.InsertTargets(A, Q, nTargets-1, targetSolutions, False)
                            print('Done inserting targets')
                            # Insert clutter using a version of Q with n edges removed
                            clutterStructs = list()
                            if insertCliques & insertClutter:
                                ClutterQ = Q.copy()
                                numRemove = min(clutterNumEdgesRemoved, QSize)
                                for i in range(numRemove):
                                    ClutterQ.remove_edge(i, QSize-1)
                                sgm.InsertTargets(A, ClutterQ, nTargets, targetSolutions, True)
                                print('Done inserting clutter')

                            #sgm.PrintGraph(A)

                            # njp end

                            start = time.time()
                            stuffLeftMDST = None
                            if not (Q is None):
                                minScore = 0
                                if insertCliques:
                                    minScore = QSize + int(comb(QSize, 2))
                                stuffLeftMDST = sgm.ReduceSpace(Q, A, minScore, method='MDST') #Nodes/Edges tuple
                                print('Done computing MDST coarse graph')
        #                        stuffLeftNorm  = sgm.ReduceSpace(Q, A, method='Normal')
        #                        print 'Done computing Normal coarse graph'
        #                        resultsDict[(ASize, pConn, nVars, QSize)].append((stuffLeftMDST, stuffLeftNorm))
                                resultsDict[(ASize, pConn, nVars, QSize)].append(stuffLeftMDST)
                            else:
                                print('Failed to grab Q in reasonable time, this archive sucks.')
                            print('Done in ' + str(time.time() - start))

                            matches = stuffLeftMDST[2]

                            PrintMatches(matches)

                            PrintGroundTruth(targetSolutions, nTargets)

                            ComputeMetrics(targetSolutions, nTargets, matches)


