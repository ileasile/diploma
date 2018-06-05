
import numpy as np
import networkx as nx
import time
import collections as coll
import random

def CalcRandomSpanningTree(Q):
    
    currNode = np.random.choice([n for n in Q.nodes()])
    rootNode = currNode
    nodesVisited = set([currNode])
    edgesUsed = set();
    nNodes = len(Q.nodes())
    while len(nodesVisited)<nNodes:
        #Choose a random neighbour
        nextNode = np.random.choice(Q.neighbors(currNode))
        if nextNode not in nodesVisited:
            edgesUsed.add((currNode, nextNode))
            nodesVisited.add(nextNode)
        currNode = nextNode
    
    T = nx.DiGraph()
    T.add_nodes_from(nodesVisited)
    T.add_edges_from(edgesUsed)
    for n in T.nodes():
        T.node[n]=Q.node[n]
    for e1,e2 in T.edges():
        if (e1,e2) in Q.edges():
            T.adj[e1][e2] = Q.adj[e1][e2]
        else:
            T.adj[e1][e2] = Q.adj[e2][e1]
    return T
    
def CalculateMDSTv2(G, nIdx, eIdx, used_stuff=set()):
    #Step 1: Figure out the weights.
    nAttName = list(nIdx.keys())[0]
    eAttName = list(eIdx.keys())[0]
    #Create an MDSTWeight vector on the nodes and edges.
    for n in G.nodes():
        if G.node[n][nAttName] in list(nIdx[nAttName].keys()):
            G.node[n]['MDSTWeight'] = len(nIdx[nAttName][G.node[n][nAttName]])/nIdx[nAttName]['size']
        else:
            G.node[n]['MDSTWeight']=0
    
    for e in G.edges():
        if e in used_stuff:
            G.adj[e[0]][e[1]]['MDSTWeight'] = 1
        else:
            if G.adj[e[0]][e[1]][eAttName] in list(eIdx[eAttName].keys()):
                G.adj[e[0]][e[1]]['MDSTWeight'] = len(eIdx[eAttName][G.adj[e[0]][e[1]][eAttName]])/eIdx[eAttName]['size']
            else:
                G.adj[e[0]][e[1]]['MDSTWeight'] = 0
                
#    for e1,e2 in G.edges():
#        G.adj[e1][e2]['Nonsense'] = 5
            
    #Step 2: Calculate the MST.
    T = nx.algorithms.minimum_spanning_tree(G,weight='MDSTWeight')
#    T = nx.algorithms.minimum_spanning_tree(G,weight='Nonsense')
    #Step 3: Figure out which root results in us doing the least work.
    bestT = None
    bestScore = np.inf
    for root in T.nodes():
        #Generate a new tree
        newT = nx.bfs_tree(T,root)
        #add attributes
        for n in newT.nodes():
            copy_node_attrs(newT, n, T, n)
        for e1,e2 in newT.edges():
            copy_edge_attrs(newT, (e1, e2), T, (e1, e2))
        newTScore = MDSTScorev2(T,root,weight='MDSTWeight')
        if newTScore<bestScore:
            bestT = newT
            bestScore = newTScore
            
    return tuple([bestT,bestScore])


def copy_node_attrs(G_to, node_to, G_from, node_from):
    for attr_name, attr_val in G_from.node[node_from].items():
        nx.set_node_attributes(G_to, {node_to: attr_val}, attr_name)

def copy_edge_attrs(G_to, edge_to, G_from, edge_from):
    for attr_name, attr_val in G_from[edge_from[0]][edge_from[1]].items():
        nx.set_edge_attributes(G_to, {edge_to: attr_val}, attr_name)

def MDSTScorev2(T, root, weight='MDSTWeight'): #Use Old formula for now
    T.node[root]['PercentLeft'] = T.node[root][weight]
    totalScore = T.node[root]['PercentLeft']
    for e in nx.bfs_edges(T,root):  
        startNode = e[0]
        endNode = e[1]
        T.node[endNode]['PercentLeft'] = T.node[endNode][weight]*T.adj[startNode][endNode][weight]*T.node[startNode]['PercentLeft']
        totalScore+= T.node[endNode]['PercentLeft']
        
    return totalScore
        
def CalculateMDST(G, numTrees, nIdx, eIdx, used_stuff=set()):
    treeList = PureGenRST(G, numTrees)
    treeScores = [tuple([t,MDSTScore(t,nIdx,eIdx,used_stuff)]) for t in treeList]
    sortedTreeList = sorted(treeScores, key=lambda score: score[1])
    return sortedTreeList[0]
    
def MDSTScore(T, nIdx, eIdx, used_stuff):
    nAttName = list(nIdx.keys())[0]
    eAttName = list(eIdx.keys())[0]
    
    root = [n for n,d in list(T.in_degree().items()) if d==0]
    root = root[0]
    if root in used_stuff:
        T.node[root]['PercentLeft'] = 1
    else:
        T.node[root]['PercentLeft']=len(nIdx[nAttName][T.node[root][nAttName]])/nIdx[nAttName]['size']
    totalScore = T.node[root]['PercentLeft']    
    for e in nx.bfs_edges(T, root):
        startNode = e[0]
        endNode = e[1]
        if e in used_stuff:
            edgeScore = 1
            endNodeScore = 1
        else:
            
            edgeScore = len(eIdx[eAttName][T.adj[startNode][endNode][eAttName]])/eIdx[eAttName]['size']
            if endNode in used_stuff:
                endNodeScore = 1
            else:
                endNodeScore = len(nIdx[nAttName][T.node[endNode][nAttName]])/nIdx[nAttName]['size']
        T.node[endNode]['PercentLeft'] = endNodeScore*T.node[startNode]['PercentLeft']*edgeScore
        totalScore += T.node[endNode]['PercentLeft']
    
    return totalScore

def insertCliqueSubgraph(A, nNodes, attDist, attName, targetSolutions):
    qNodes = np.random.choice(A.nodes(), nNodes, replace=False)
    AtoQ = dict()
    QtoA = dict()
    newTarget = dict()
    newTarget['nodes'] = list()
    newTarget['edges'] = list()
    newTarget['isClutter'] = False
    for i, n in enumerate(qNodes):
        AtoQ[n] = i  # Maps from archive node to T.
        QtoA[i] = n  # Maps from T to archive node
        newTarget['nodes'].append(n)
    for i in range(nNodes):
        for j in range(i+1, nNodes):
            n1 = QtoA[i]
            n2 = QtoA[j]
            if A.has_edge(n1,n2):
                newTarget['edges'].append((n1, n2))
                continue
            A.add_edge(n1, n2)
            newTarget['edges'].append((n1,n2))
            AddSingleEdgeAttribute(A, n1, n2, attDist, attName)
    targetSolutions.append(newTarget)
    return createQGraph(A, QtoA, AtoQ, qNodes)

#def getRandomSubgraphOld(A, nNodes):
#    for i in range(1000):
#        qNodes = np.random.choice(A.nodes(), nNodes, replace=False)
#        AtoQ = dict()
#        QtoA = dict()
#        for i, n in enumerate(qNodes):
#            AtoQ[n] = i  # Maps from archive node to T.
#            QtoA[i] = n  # Maps from T to archive node
#
#        Q = nx.Graph()
#        Q.add_nodes_from(QtoA.keys());
#
#        for eStart, eEnd in A.edges():
#            if eStart in qNodes and eEnd in qNodes:
#                Q.add_edge(AtoQ[eStart], AtoQ[eEnd])
#
#        if len([g for g in nx.connected_components(Q.to_undirected())]) > 1:
#            continue
#
#        for n in Q.nodes():
#            Q.node[n] = A.node[QtoA[n]]
#            Q.node[n]['realNode'] = QtoA[n]
#        for n1, n2 in Q.edges():
#            Q.adj[n1][n2] = A.adj[QtoA[n1]][QtoA[n2]]
#            Q.adj[n1][n2]['realEdge'] = (QtoA[n1], QtoA[n2])
#        return Q
#    return None

def getRandomSubgraph(A, nNodes):
    for i in range(1000):
        qNodes = np.random.choice(A.nodes(), nNodes,replace=False)
        AtoQ = dict()
        QtoA = dict()
        for i, n in enumerate(qNodes):
            AtoQ[n] = i #Maps from archive node to T.
            QtoA[i] = n #Maps from T to archive node
        Q = createQGraph(A, QtoA, AtoQ, qNodes)
        if not acceptQGraph(Q):
            continue
        return Q
    return None

def acceptQGraph(Q):
    return len([g for g in nx.connected_components(Q.to_undirected())]) <= 1

def createQGraph(A, QtoA, AtoQ, qNodes):
    Q = nx.Graph()
    Q.add_nodes_from(list(QtoA.keys()))
    for eStart, eEnd in A.edges():
        if eStart in qNodes and eEnd in qNodes:
            Q.add_edge(AtoQ[eStart], AtoQ[eEnd])
        continue
    for n in Q.nodes():
        for node_attr, attr_val in A.node[QtoA[n]].items():
            nx.set_node_attributes(Q, {n: attr_val}, node_attr)
        # Note: This line somehow affects both Q and A, even though the dict objects are different! - njp
        # Instead of keeping all the mappings as annotations on Q or A, they are stored elsewhere in targetSolutions
        nx.set_node_attributes(Q, {n: QtoA[n]}, 'fromANode')  # use 'fromANode' to point to the original prototype node
#        A.node[QtoA[n]][nodeLabel] = QtoA[n]
    for n1, n2 in Q.edges():
        for edge_attr, attr_val in A[QtoA[n1]][QtoA[n2]].items():
            nx.set_edge_attributes(Q, {(n1, n2): attr_val}, edge_attr)

        nx.set_edge_attributes(Q, {(n1, n2): (QtoA[n1], QtoA[n2])}, 'fromAEdge')# use this to point to prototype edge
#        A.adj[QtoA[n1]][QtoA[n2]][edgeLabel] = (QtoA[n1],QtoA[n2])
    return Q


def InsertTargets(A: nx.Graph, Q: nx.Graph, nTargets, targetSolutions, isClutter):
    for i in range(nTargets):
        # Map Q to a corresponding set of random nodes in A
        mapToA = dict()
        newTarget = dict()
        newTarget['nodes'] = list()
        newTarget['edges'] = list()
        newTarget['isClutter'] = isClutter
        for node in Q.nodes():
            ind = -1
            while True:
                ind = np.random.choice(list(range(len(A.nodes()))))
                if not IsNodeInExistingTarget(ind, targetSolutions):
                    break
            mapToA[node] = ind
            newTarget['nodes'].append(ind)
            # copy node attributes exactly
            for key, val in list(Q.node[node].items()):
                if (key == 'nValue'):
                    A.node[ind][key] = val
        # Add the same edges in Q to A
        for edge in Q.edges():
            srcA = mapToA[edge[0]]
            destA = mapToA[edge[1]]
            edgeData = Q.get_edge_data(edge[0], edge[1]).copy()

            A.add_edge(srcA, destA)
            for edge_attr, attr_val in edgeData.items():
                nx.set_edge_attributes(Q, {(srcA, destA): attr_val}, edge_attr)

            newTarget['edges'].append((srcA,destA))
        targetSolutions.append(newTarget)
    print(A)

def IsNodeInExistingTarget(ind, targetSolutions):
    for target in targetSolutions:
        if ind in target['nodes']:
            return True
    return False

def GenRandomGraph(nNodes, pConnected):
    G = nx.erdos_renyi_graph(nNodes, pConnected)
#    G = nx.Graph()
#    G.add_nodes_from(range(nNodes))
#    for n1 in G.nodes():
#        for n2 in range(n1+1,len(G.nodes())):
#            if np.random.rand(1)<pConnected:
#                G.add_edge(n1,n2)
    return G
    
def AddNodeAttributes(G,attDist,attName):
    for n in G.nodes():
        G.node[n][attName] = np.random.choice(list(range(len(attDist))),p=attDist)
    return G
    
def AddEdgeAttributes(G,attDist,attName):
    for n1,n2 in G.edges():
        AddSingleEdgeAttribute(G,n1,n2,attDist,attName)
    return G

def AddSingleEdgeAttribute(G,src,dest,attDist,attName):
    val = np.random.choice(list(range(len(attDist))), p=attDist)
    G.adj[src][dest][attName] = val

def CreateDict(G, nAttName, eAttName):
    nodeDict = dict()
    nodeDict[nAttName] = dict()
    nodeDict[nAttName]['size'] = len(G.nodes())
    #find all the total values
    uniqueKeys = np.unique([G.node[n][nAttName] for n in G.nodes()])
    for k in uniqueKeys:
        nodeDict[nAttName][k] = set([n for n in G.nodes() if (G.node[n][nAttName]==k)])
        
    edgeDict = dict()
    edgeDict[eAttName] = dict()
    edgeDict[eAttName]['size'] = len(G.edges())
    #find all the total values
    uniqueKeys = np.unique([G.adj[e1][e2].get(eAttName, -1) for e1,e2 in G.edges()])
    for k in uniqueKeys:
        if k == -1:
            continue
        edgeDict[eAttName][k] = set([e for e in G.edges() if G.adj[e[0]][e[1]].get(eAttName, -1) == k])
        edgeDict[eAttName][k].update(set([(e[1],e[0]) for e in G.edges() if (G.adj[e[0]][e[1]].get(eAttName, -1) == k)]))

    return tuple([nodeDict, edgeDict])

def SGMMatch(T, G, delta, tau, nIdx, eIdx):
    #T is a query tree
    #G is a query graph
    #Delta is the score delta that we can accept from perfect match
    #tau is how far off this tree is from the graph, at most.
    #nIdx is an index containing node attributes
    #eIdx is an index containing edge attributes
    rootMatch = [n for n,d in list(T.in_degree().items()) if d==0]
    root = rootMatch[0]
    nK = list(nIdx.keys())[0]
    eK = list(eIdx.keys())[0]

#    print 'Building matching graph'

    print('Printing MDST Graph')
    print(root)
    PrintGraph(T)
    
    #Step 1: Get all the matches for the nodes
    nodeMatches = dict()
    for n in T.nodes():
        if T.node[n][nK] in list(nIdx[nK].keys()):
            nodeMatches[n] = nIdx[nK][T.node[n][nK]]
        else:
            nodeMatches[n] = set()
    
    #Step 2: Get all the edge matches for the node
    edgeMatches = dict()
    for e1,e2 in T.edges():
        if T.adj[e1][e2][eK] in list(eIdx[eK].keys()):
            edgeMatches[tuple([e1,e2])] = eIdx[eK][T.adj[e1][e2][eK]]
        else:
            edgeMatches[tuple([e1,e2])] = set()
        #Make sure you count just the ones that have matching nodes too.
        edgeMatches[tuple([e1,e2])] = set([e for e in edgeMatches[tuple([e1,e2])] if e[0] in nodeMatches[e1] and e[1] in nodeMatches[e2]])
        
    #Scoring, initially, is going to be super-simple:  You get a 1 if you match, and a 0 if you don't.  Everything's created equal.
        
    #Score everything and put it in a graph.
    
    for k in list(edgeMatches.keys()):
        if len(edgeMatches[k])==0:
            stophere = 1
    
    matchGraph = nx.DiGraph()
#    for nT in T.nodes():
#        for nG in nodeMatches[nT]:
#            MatchGraph.add_node(tuple([nT,nG]),score=1,solo_score=1)
    
    for eT1, eT2 in T.edges():
        for eG1, eG2 in edgeMatches[tuple([eT1,eT2])]:
            matchGraph.add_edge(tuple([eT1,eG1]), tuple([eT2, eG2]),score=1,solo_score=1)
            
    for nM in matchGraph.nodes():
        matchGraph.node[nM] = {'solo_score':1, 'score':1, 'path':coll.deque() }
        matchGraph.node[nM]['path'].appendleft(nM)
        
    #Get rid of anybody flying solo
    matchGraph = ClearUnconnected(matchGraph,root) #this is clearly not working.
            
    #Now acquire/organize all hypotheses with scores above Max_Score - tau - delta
            
    #Figure out how much score you could possibly get at every node in the query.
    for n in T.nodes():
        T.node[n]['max_score']=1
    for eT1,eT2 in T.edges():
        T.adj[eT1][eT2]['max_score']=1
    
    bfsedges = list(nx.bfs_edges(T,root))
    reversebfsedges = list(reversed(bfsedges))
    
    for eT1,eT2 in reversebfsedges: #Reverse BFS search - should do leaf nodes first.
        #What's the best score we could get at this node?
        T.node[eT1]['max_score'] += T.node[eT2]['max_score']+T.adj[eT1][eT2]['max_score']
        
        #Find all the edges equivalent to this one in the match graph
        edgeMatches = [tuple([eG1,eG2]) for eG1,eG2 in matchGraph.edges() if eG1[0]==eT1 and eG2[0]==eT2]
                
        parentNodes = set([eM1 for eM1,eM2 in edgeMatches])
        
        for p in parentNodes:
            childNodes = [eM2 for eM1,eM2 in edgeMatches if eM1==p]
            #First, check if the bottom node has a score
            bestScore = 0
            bestNode = None
            for c in childNodes:
                cScore = matchGraph.node[c]['score'] + matchGraph.adj[p][c]['score']
                cPath = matchGraph.node[c]['path']
                if cScore > bestScore:
                    bestScore = cScore
                    bestChildPath = cPath
            matchGraph.node[p]['score'] += bestScore
            for pathNode in cPath:
                matchGraph.node[p]['path'].appendleft(pathNode)

    #CLEAN IT UP.
    for n in matchGraph.nodes():
        if matchGraph.node[n]['score']<T.node[n[0]]['max_score']-delta:   # - tau in original SWGraphFunctions.py
            matchGraph.remove_node(n)

    #Get rid of anybody flying solo
    matchGraph = SaveRootChildren(matchGraph,root)

    # Score the root solutions
    return matchGraph


def ScoreSolution(Q, A, solution):
    # ARchive graph A
    # query graph Q
    # Solution = list of tuples, length |G|, e.g. (1,3),(2,8),(3,12)
    solutionScore = 0;
    solDict = dict()
    # Current impl simply adds 1 for each matching node or edge
    for s in solution:  # s should be a tuple e.g. (1,3)
        qNode = s[0]
        aNode = s[1]
        if qNode in Q.node and aNode in A.node:
            if 'nValue' in Q.node[qNode] and 'nValue' in A.node[aNode]:
                solutionScore += Q.node[qNode]['nValue'] == A.node[aNode]['nValue']
                solDict[qNode] = aNode

    for e in Q.edges():
        if e[0] in solDict and e[1] in solDict:
            if A.has_edge(solDict[e[0]], solDict[e[1]]):
                if Q.adj[e[0]][e[1]]['eValue'] == A.adj[solDict[e[0]]][solDict[e[1]]]['eValue']:
                    solutionScore += 1
                    solDict[(e[0],e[1])] = (solDict[e[0]], solDict[e[1]])

    return solDict, solutionScore


def ClearUnconnected(G,keyNode):
    #Get rid of any component that doesn't contain the key node.
    compList = nx.connected_components(G.to_undirected());
    for c in compList:
        if len([n for n in c if n[0]==keyNode])==0:
            G.remove_nodes_from(c)
    return G
    
def SaveRootChildren(G,rootID):
    ToSave = set()
    roots = [n for n in G.nodes() if n[0]==rootID];
    for n in roots:
        T = nx.algorithms.dfs_tree(G, n)
        ToSave.update(T.nodes())
    ToKill = set(G.nodes()) - ToSave
    G.remove_nodes_from(ToKill)
    return G
    
def path_exists(G,source,target):
    for path in nx.all_simple_paths(G,source,target):
        return True  #if it finds one, it returns True, and gets out of the function.  It doesn't look for the next.
    return False  #if it didn't find one, it gets out of the function.
    
def SubsampleArchiveFromMatching(A, mg, T, eIdx):
    APrime = A.copy()
        
    toSave = set([n[1] for n in mg.nodes()])    
    
    #get rid of all the nodes in A that aren't there
    for n in A.nodes():
        if not n in toSave:
            APrime.remove_node(n)
        
    return APrime
    
def ReduceSpace(Q, A, minScore, method='MDST'):
    NumEdgesA = list()
    NumNodesA = list()
    usedEdges = set();
    delta = 0;  # set to non-zero for corruption experiments - njp

    # Hash
    print('Hashing...')
    start = time.time()
    [nIdx, eIdx] = CreateDict(A, 'nValue', 'eValue')  # Create an attribute index
    #        PrintWeights(nIdx,eIdx)
    print('Done at ' + str(time.time() - start))

    algStart = time.time()

    # not sure why this loop is here, as results are same each time - njp
    for idx in range(1):   # changed to 1 - njp
        NumEdgesA.append(len(A.edges()))
        NumNodesA.append(len(A.nodes()))

        print('Calculating MDST')
        start = time.time()

        if method=='MDST' and len(usedEdges)<2*len(Q.edges()):
            T, TScore = CalculateMDSTv2(Q, nIdx, eIdx, used_stuff = usedEdges)
        if method=='Normal' or len(usedEdges)>=2*len(Q.edges()):
            if method=='MDST':
                stophere = 1
            T = CalcRandomSpanningTree(Q)
        print('Done at '+str(time.time()-start))
    
        #Add used stuff to used.
        edges_used = T.edges()
        usedEdges.update(edges_used)
        usedEdges.update([tuple([e2,e1]) for e1,e2 in edges_used])
        
        #also figure out what is unused.    
        tau = len(Q.edges()) - len(T.edges()) #Dumb way of calculating tau
    
        print('Matching')
        start = time.time()
        mg = SGMMatch(T, A, delta, tau, nIdx, eIdx)

        print('Done at '+str(time.time()-start))
        print('Printing mg:')
        root = [n for n, d in list(mg.in_degree().items()) if d == 0]
        root = root[0]
        print(root)
        PrintGraph(mg)
        APrime = SubsampleArchiveFromMatching(A,mg,T,eIdx)
#        print 'Printing APrime'
#        PrintGraph(APrime)
        A = APrime

        # Score the solutions
        print('Scoring solutions:')
        thresholdMatches = list()
        roots = [n for n in mg.nodes() if mg.in_degree(n) == 0];
        for root in roots:
            sol = mg.node[root]['path']
            match, score = ScoreSolution(Q, APrime, sol)
            print('Got match with score: ', score)
            print(match)
            if score >= minScore:
                thresholdMatches.append(match)


    NumEdgesA.append(len(A.edges()))
    NumNodesA.append(len(A.nodes()))

    print('Algorithm Post-hash Stages Done at ' + str(time.time() - algStart))

    return(NumNodesA, NumEdgesA, thresholdMatches)


## I/O Utilities

def PrintGraph(G):
    nodes = G.nodes()
    for node in nodes:
        print(node, list(G.node[node].items()))
    for n, nbrs in G.adjacency():
        for nbr, eattr in list(nbrs.items()):
            print(('(%s, %s, %s)' % (n, nbr, eattr)))


def PrintWeights(nIdx,eIdx):
    nAttName = list(nIdx.keys())[0]
    eAttName = list(eIdx.keys())[0]
    
    for k in nIdx[nAttName]:
        if not k=='size':
            print('Node attribute ' + str(k)+':'+str(len(nIdx[nAttName][k])/nIdx[nAttName]['size']))

    for k in eIdx[eAttName]:
        if not k=='size':
            print('Edge attribute ' + str(k)+':'+str(len(eIdx[eAttName][k])/eIdx[eAttName]['size']))
            
