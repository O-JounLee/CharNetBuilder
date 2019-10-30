# -*- coding: utf-8 -*-

import numpy 
import json
import networkx as nx
import math
import matplotlib.pyplot as plt
import operator
from xml.etree.ElementTree import parse
import os

"""
def dataset_reader(path):
    
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    
    name = path.strip(".json").split("/")[-1]
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])
    centrality = nx.degree_centrality()

    if "features" in data.keys():
        features = data["features"]
    else:
        features = nx.degree(graph)

    features = {int(k):v for k,v, in features.items()}
    return graph, features, name
"""

class CharNetReader:
    """
    Read CharNets and Convert to Graphs
    """
    def __init__(self,path):
        
        self.Path = path
        
        #self.nodes = self.graph.nodes()
        self.CharNets = []
        self.CharNets = self.readCharNet()
        self.CharNetStr = self.CharNets[0]
        self.LastScene = self.CharNets[1]
        self.NumOfScenes = self.CharNets[2]
        self.NumOfChar = self.CharNets[3]
        self.centrality = {}
        self.role = {}
        
        self.CharNames = []
        self.errorChars = []

        self.CharNetGraphs = []
        self.CharNet2Graph(self.CharNetStr, self.NumOfScenes, self.NumOfChar)
        self.discretizeCharNet()
    
    #moved to corpus manager
    """
    def getCharNetsList(self):
        
        path_dir = './XML'
        file_list = os.listdir(path_dir)
        file_list.sort()
        
        #print(file_list)
        
        return file_list
    """
    
    def readCharNet(self):
        #tree = parse("C:\Users\O-Joun Lee\Story2Vec\CharNetsInXML\Kung-Fu-Panda_characterNet.xml")
        #tree = parse("C:\\Users\\O-Joun Lee\\Story2Vec\\CharNetsInXML\\Kung-Fu-Panda_characterNet.xml")
        tree = parse(self.Path)
        CharNetXML = tree.getroot()
        CharNetStream = [numpy.array([0])]

        CharList = CharNetXML.findtext("characterList")
        ECharList = CharNetXML.findtext("EcharacterList")
        
        self.CharNames = CharList.split('  ')
        self.errorChars = ECharList.split('  ')
        
        self.CharNames.pop()
        self.errorChars.pop()

        #print(self.CharNames)
        #print(self.errorChars)

        for CharNetSet in CharNetXML.findall("characterNet"): 
            Scene = int(CharNetSet.findtext("sceneNum"))
            CharNet = CharNetSet.findtext("accumulativeCharNet")
            CharNet = CharNet.split('  //')
            for CharNum in range(len(CharNet)):
                CharNet[CharNum] = CharNet[CharNum].split('  ')
                
            CharNet.pop()
            #print(Scene)
            CharNet = numpy.array(CharNet)
            CharNetStream.insert(Scene-1, CharNet)
            
        CharNetStream.pop()
        
        #print(CharNetStream[662])
        
        LastScene = CharNetStream[len(CharNetStream)-1]
        NumOfScenes = len(CharNetStream)
        NumOfChar = int(math.sqrt(CharNetStream[len(CharNetStream)-1].size))
    
        return CharNetStream, LastScene, NumOfScenes, NumOfChar
    
    
    def CharNet2Graph(self,CharNetStream, NumOfScenes, NumOfChar):
        
        #CharNetGraphs = []
        #print(NumOfScenes)
        #print(len(CharNetStream))
        #print(CharNetStream[NumOfScenes-2])
        
        for i in range(NumOfScenes):
            CharNetGraph = nx.DiGraph()
            
            for j in range(NumOfChar):
                if float(CharNetStream[i].item(j,j)) != 0:
                    CharNetGraph.add_node('c'+ str(j))
                
            for j in range(NumOfChar):
                for k in range(NumOfChar):
                    if float(CharNetStream[i].item(j,k)) != 0:
                        #print(CharNetStream[i].item(j,k))
                        CharNetGraph.add_edge('c'+ str(j), 'c'+ str(k), weight=float(CharNetStream[i].item(j,k)))
            
            self.CharNetGraphs.insert(i, CharNetGraph)
            
            if i == NumOfScenes-1:
                centrality = centralities_as_dict(CharNetGraph)
                #print(centrality)
                for j in range(NumOfChar):
                    self.centrality['c'+ str(j)] = (centrality['weighted_deg']['c'+ str(j)] + centrality['closeness_cent']['c'+ str(j)] + centrality['betweeness_cent']['c'+ str(j)])

        nx.draw(self.CharNetGraphs[len(self.CharNetGraphs)-1])
        plt.show()

        return #CharNetGraphs

    def discretizeCharNet(self):
        gap = []
        rank = sorted(self.centrality.items(), key=operator.itemgetter(1), reverse=True)
        
        for i in range(self.NumOfChar-1):
            gap.insert(i,(rank[i][0], rank[i+1][0], rank[i][1] - rank[i+1][1]))
            
        sortedGap = sorted(gap, key=operator.itemgetter(2), reverse=True)
        BoundaryMain = sortedGap[0][0]
        BoundaryMinor = sortedGap[1][0]
        
        #print(sortedGap)
        centGap = [((f + s), g) for f,s,g in sortedGap]
        zip(*centGap)
        #plt.plot(*zip(*centGap))
        #plt.show()
        
        
        for i in range(self.NumOfChar):
            if i == 0:
                self.role[rank[i][0]] = 'P'
            elif rank[i][1] >= self.centrality[BoundaryMain]:
                self.role[rank[i][0]] = 'M'
            elif rank[i][1] >= self.centrality[BoundaryMinor]:
                self.role[rank[i][0]] = 'm'
            else:
                self.role[rank[i][0]] = 'e'
                
        
        #print(self.role)
        #print(self.CharNetGraphs)
        #self.CharNetGraphs[self.NumOfScenes-1].edges(data=True)
        
        
        
        for i in range(self.NumOfScenes):
            proximity = {n:0 for n in self.CharNetGraphs[i].edges()}
            BoundaryHigh = 0
            BoundaryMedium = 0
            
            #print(proximity)
            for speaker, listener, freq in self.CharNetGraphs[i].edges(data=True):
                if speaker != listener:
                    proximity[(speaker, listener)] = freq['weight']
                 
            rank = sorted(proximity.items(), key=operator.itemgetter(1), reverse=True)
            gap = []
            
            for j in range(len(rank)-1): 
                gap.insert(j,(rank[j][0], rank[j+1][0], rank[j][1] - rank[j+1][1]))
            
            sortedGap = sorted(gap, key=operator.itemgetter(2), reverse=True)
            if len(gap) >= 1:
                BoundaryHigh = proximity[sortedGap[0][0]]
            if len(gap) >= 2:
                BoundaryHigh = max(proximity[sortedGap[0][0]],proximity[sortedGap[1][0]])
                BoundaryMedium = min(proximity[sortedGap[0][0]],proximity[sortedGap[1][0]])
            
            
            
            #print('Scene:', i)
            #print(gap)
            #print('Scene:', i)
            #print(sortedGap)
            #print(BoundaryHigh, BoundaryMedium)
            
            for speaker, listener, freq in self.CharNetGraphs[i].edges(data=True):
                if freq['weight'] >= BoundaryHigh:
                    self.CharNetGraphs[i][speaker][listener]['weight'] = 'H'
                elif freq['weight'] >= BoundaryMedium:
                    self.CharNetGraphs[i][speaker][listener]['weight'] = 'I'
                else:
                    self.CharNetGraphs[i][speaker][listener]['weight'] = 'L'
            
            #print('Scene:', i)
            #print(self.CharNetGraphs[i].edges(data=True))
            
            #print(sortedGap)
            
            #proxGap = [((f1 + s1), g) for (f1,s1),(f2,s2),g in sortedGap]
            #zip(*proxGap)
            #plt.plot(*zip(*proxGap))
            #plt.show()
            
            
            #print(BoundaryHigh)
            #print(BoundaryMedium)
            #self.CharNetGraphs[i]
            #print(i)
        
        return


def centralities_as_dict(input_g):

    def return_weighted_degree_centrality(input_g, normalized=True):
        w_d_centrality = {n:0.0 for n in input_g.nodes()}
        for u, v, d in input_g.edges(data=True):
            w_d_centrality[u]+= d['weight']
            w_d_centrality[v]+= d['weight']
        if normalized==True:
            weighted_sum = sum(w_d_centrality.values())
            weighted_sum += 1 #to avoid ZeroDivisionError
            return {k:v/weighted_sum for k, v in w_d_centrality.items()}
        else:
            return w_d_centrality
        
    def return_closeness_centrality(input_g):
        new_g_with_distance = input_g.copy()
        for u,v,d in new_g_with_distance.edges(data=True):
            if 'distance' not in d:
                d['distance'] = 1.0/ d['weight']
        return nx.closeness_centrality(new_g_with_distance, distance='distance')
    
    def return_betweenness_centrality(input_g):
        return nx.betweenness_centrality(input_g, weight='weight')
    
    return {
        'weighted_deg':return_weighted_degree_centrality(input_g),
        'closeness_cent':return_closeness_centrality(input_g), 
        'betweeness_cent':return_betweenness_centrality(input_g),
    }





#Testing Codes
CharNetReader("C:/Users/OJ/Documents/XML/GOOD WILL HUNTING_sample.xml")