# -*- coding: utf-8 -*-

import numpy 
import json
import networkx as nx
import math
import matplotlib.pyplot as plt
import operator
import seaborn as sns
import pandas as pd
import numpy as np
import math
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

        self.CharNames = []
        self.errorChars = []
        
        #self.nodes = self.graph.nodes()
        self.CharNets = []
        self.CharNets = self.readCharNet()
        self.CharNetStr = self.CharNets[0]
        self.LastScene = self.CharNets[1]
        self.NumOfScenes = self.CharNets[2]
        self.NumOfChar = self.CharNets[3]
        self.centrality = {}
        self.role = {}
        
        self.AcCharNetGraphs = []
        self.DisCharNetGraphs = []

        self.CharNet2Graph(self.CharNetStr, self.NumOfScenes, self.NumOfChar)
        #self.discretizeCharNet()

        self.calcFeatures()
    
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

        CharList = CharNetXML.findtext("characterList")
        ECharList = CharNetXML.findtext("EcharacterList")
        
        self.CharNames = CharList.split('  ')
        self.errorChars = ECharList.split('  ')
        
        self.CharNames.pop()
        self.errorChars.pop()

        #print(self.CharNames)
        #print(self.errorChars)

        AcCharNetStream = [numpy.array([0])]
        DisCharNetStream = [numpy.array([0])]
        AcDialCharNetStream = [numpy.array([0])]
        DisDialCharNetStream = [numpy.array([0])]

        BiAcCharNetStream = [numpy.array([0])]
        BiDisCharNetStream = [numpy.array([0])]
        BiAcDialCharNetStream = [numpy.array([0])]
        BiDisDialCharNetStream = [numpy.array([0])]

        for CharNetSet in CharNetXML.findall("characterNet"): 
            Scene = int(CharNetSet.findtext("sceneNum"))

            ######################################################
            AcCharNet = CharNetSet.findtext("accumulativeCharNet")
            AcCharNet = AcCharNet.split('  //')
            for CharNum in range(len(AcCharNet)):
                AcCharNet[CharNum] = AcCharNet[CharNum].split('  ')
                
            AcCharNet.pop()
            #print(Scene)
            AcCharNet = numpy.array(AcCharNet)
            AcCharNetStream.insert(Scene-1, AcCharNet)
            
            ######################################################
            DisCharNet = CharNetSet.findtext("discreteCharNet")
            DisCharNet = DisCharNet.split('  //')
            for CharNum in range(len(DisCharNet)):
                DisCharNet[CharNum] = DisCharNet[CharNum].split('  ')
                
            DisCharNet.pop()
            #print(Scene)
            DisCharNet = numpy.array(DisCharNet)
            DisCharNetStream.insert(Scene-1, DisCharNet)
            

            ######################################################
            AcDialCharNet = CharNetSet.findtext("accumulativeDialCharNet")
            AcDialCharNet = AcDialCharNet.split('  //')
            for CharNum in range(len(AcDialCharNet)):
                AcDialCharNet[CharNum] = AcDialCharNet[CharNum].split('  ')
                
            AcDialCharNet.pop()
            #print(Scene)
            AcDialCharNet = numpy.array(AcDialCharNet)
            AcDialCharNetStream.insert(Scene-1, AcDialCharNet)
            
            ######################################################
            DisDialCharNet = CharNetSet.findtext("discreteDialCharNet")
            DisDialCharNet = DisDialCharNet.split('  //')
            for CharNum in range(len(DisDialCharNet)):
                DisDialCharNet[CharNum] = DisDialCharNet[CharNum].split('  ')
                
            DisDialCharNet.pop()
            #print(Scene)
            DisDialCharNet = numpy.array(DisDialCharNet)
            DisDialCharNetStream.insert(Scene-1, DisDialCharNet)



            ######################################################
            BiAcCharNet = CharNetSet.findtext("BiaccumulativeCharNet")
            BiAcCharNet = BiAcCharNet.split('  //')
            for CharNum in range(len(BiAcCharNet)):
                BiAcCharNet[CharNum] = BiAcCharNet[CharNum].split('  ')
                
            BiAcCharNet.pop()
            #print(Scene)
            BiAcCharNet = numpy.array(BiAcCharNet)
            BiAcCharNetStream.insert(Scene-1, BiAcCharNet)
            
            ######################################################
            BiDisCharNet = CharNetSet.findtext("BidiscreteCharNet")
            BiDisCharNet = BiDisCharNet.split('  //')
            for CharNum in range(len(BiDisCharNet)):
                BiDisCharNet[CharNum] = BiDisCharNet[CharNum].split('  ')
                
            BiDisCharNet.pop()
            #print(Scene)
            BiDisCharNet = numpy.array(BiDisCharNet)
            BiDisCharNetStream.insert(Scene-1, BiDisCharNet)
            

            ######################################################
            BiAcDialCharNet = CharNetSet.findtext("BiaccumulativeDialCharNet")
            BiAcDialCharNet = BiAcDialCharNet.split('  //')
            for CharNum in range(len(BiAcDialCharNet)):
                BiAcDialCharNet[CharNum] = BiAcDialCharNet[CharNum].split('  ')
                
            BiAcDialCharNet.pop()
            #print(Scene)
            BiAcDialCharNet = numpy.array(BiAcDialCharNet)
            BiAcDialCharNetStream.insert(Scene-1, BiAcDialCharNet)
            
            ######################################################
            BiDisDialCharNet = CharNetSet.findtext("BidiscreteDialCharNet")
            BiDisDialCharNet = BiDisDialCharNet.split('  //')
            for CharNum in range(len(BiDisDialCharNet)):
                BiDisDialCharNet[CharNum] = BiDisDialCharNet[CharNum].split('  ')
                
            BiDisDialCharNet.pop()
            #print(Scene)
            BiDisDialCharNet = numpy.array(BiDisDialCharNet)
            BiDisDialCharNetStream.insert(Scene-1, BiDisDialCharNet)




        ######################################################
        AcCharNetStream.pop()
        DisCharNetStream.pop()
        AcDialCharNetStream.pop()
        DisDialCharNetStream.pop()

        BiAcCharNetStream.pop()
        BiDisCharNetStream.pop()
        BiAcDialCharNetStream.pop()
        BiDisDialCharNetStream.pop()
    
        #print(CharNetStream[662])
        LastScene = AcCharNetStream[len(AcCharNetStream)-1]
        NumOfScenes = len(AcCharNetStream)
        NumOfChar = int(math.sqrt(AcCharNetStream[len(AcCharNetStream)-1].size))

        CharNetStreams = []
        CharNetStreams.append(AcCharNetStream)
        CharNetStreams.append(DisCharNetStream)
        CharNetStreams.append(AcDialCharNetStream)
        CharNetStreams.append(DisDialCharNetStream)
        CharNetStreams.append(BiAcCharNetStream)
        CharNetStreams.append(BiDisCharNetStream)
        CharNetStreams.append(BiAcDialCharNetStream)
        CharNetStreams.append(BiDisDialCharNetStream)

        return CharNetStreams, LastScene, NumOfScenes, NumOfChar
    
    
    def CharNet2Graph(self,CharNetStreams, NumOfScenes, NumOfChar):
        
        #CharNetGraphs = []
        #print(NumOfScenes)
        #print(len(CharNetStream))
        #print(CharNetStream[NumOfScenes-2])

        #print(NumOfChar)
        #print(len(self.CharNames))
        #print(self.CharNames)
        
        AcCharNetStream = CharNetStreams[0]

        for i in range(NumOfScenes):
            AcCharNetGraph = nx.DiGraph()
            
            for j in range(NumOfChar):
                if not j in self.errorChars:
                    if float(CharNetStreams[4][i].item(j,j)) != 0:
                        #AcCharNetGraph.add_node('c'+ str(j)) self.CharNames
                        AcCharNetGraph.add_node(self.CharNames[j])
                
            for j in range(NumOfChar):
                for k in range(NumOfChar):
                    if (not j in self.errorChars) and (not k in self.errorChars):
                        if float(AcCharNetStream[i].item(j,k)) != 0:
                            #print(CharNetStream[i].item(j,k))
                            #AcCharNetGraph.add_edge('c'+ str(j), 'c'+ str(k), weight=float(AcCharNetStream[i].item(j,k)))
                            AcCharNetGraph.add_edge(self.CharNames[j], self.CharNames[k], weight=float(AcCharNetStream[i].item(j,k)))
            
            self.AcCharNetGraphs.insert(i, AcCharNetGraph)
            
            if i == NumOfScenes-1:
                centrality = centralities_as_dict(AcCharNetGraph)
                #print(centrality)
                for j in range(NumOfChar):
                    if not j in self.errorChars:
                        if float(CharNetStreams[4][i].item(j,j)) != 0:
                            #self.centrality['c'+ str(j)] = (centrality['weighted_deg']['c'+ str(j)] + centrality['closeness_cent']['c'+ str(j)] + centrality['betweeness_cent']['c'+ str(j)])
                            self.centrality[self.CharNames[j]] = (centrality['weighted_deg'][self.CharNames[j]] + centrality['closeness_cent'][self.CharNames[j]] + centrality['betweeness_cent'][self.CharNames[j]])

        #nx.draw(self.AcCharNetGraphs[len(self.AcCharNetGraphs)-1], with_labels = True)
        #plt.show()
        nx.write_graphml(self.AcCharNetGraphs[len(self.AcCharNetGraphs)-1], "C:/Users/OJ/Documents/XML/GOOD WILL HUNTING_sample_AcCharNet_Last.graphml")

        return #CharNetGraphs

    def calcFeatures(self):

        DisCharNetStr = self.CharNetStr[1]
        DisDialCharNetStr = self.CharNetStr[3]
        BiDisCharNetStr = self.CharNetStr[5]

        #importance
        Importance = []

        for i in range(self.NumOfScenes):
            ImportanceInScene = []
            MaxFreq = 0
            for j in range(self.NumOfChar):
                ImportanceInScene.append(int(DisCharNetStr[i].item(j,j)))
                if MaxFreq < int(DisCharNetStr[i].item(j,j)): 
                    MaxFreq = int(DisCharNetStr[i].item(j,j))
            for j in range(self.NumOfChar):
                if MaxFreq != 0:
                    ImportanceInScene[j] = ImportanceInScene[j]/MaxFreq
                else:
                    ImportanceInScene[j] = 0
            Importance.append(ImportanceInScene)

        #Length
        Length = []

        for i in range(self.NumOfScenes):
            LengthInScene = []
            MaxFreq = 0
            for j in range(self.NumOfChar):
                if float(DisCharNetStr[i].item(j,j)) != 0:
                    LengthInScene.append(float(DisDialCharNetStr[i].item(j,j))/float(DisCharNetStr[i].item(j,j)))
                else:
                    LengthInScene.append(0)
            Length.append(LengthInScene)

        
        LengthForScene = []
        for i in range(self.NumOfScenes):
            #LengthForScene.append(np.average(np.asarray(Length[i])))
            sumLength = 0
            numChar = 0
            for j in range(self.NumOfChar):
                if float(BiDisCharNetStr[i].item(j,j)) != 0:
                    sumLength += Length[i][j]
                    numChar += 1
            if numChar != 0:
                LengthForScene.append(sumLength/numChar)
            else:
                LengthForScene.append(0)

        MaxLength = max(np.asarray(LengthForScene))
        for i in range(self.NumOfScenes):
            LengthForScene[i] = LengthForScene[i]/MaxLength
        

        #Ratios
        Ratios = []

        for i in range(self.NumOfScenes):
            RatioInScene = []
            MaxFreq = 0
            for j in range(self.NumOfChar):
                if float(BiDisCharNetStr[i].item(j,j)) != 0:
                    RatioInScene.append(float(DisCharNetStr[i].item(j,j))/float(BiDisCharNetStr[i].item(j,j)))
                else:
                    RatioInScene.append(0)
            Ratios.append(RatioInScene)

        
        RatioForScene = []
        for i in range(self.NumOfScenes):
            logRsum = 0
            numChar = 0
            for j in range(self.NumOfChar):
                if float(BiDisCharNetStr[i].item(j,j)) != 0:
                    logRsum -= math.log(Ratios[i][j])
                    numChar += 1
            if numChar != 0:
                logAver = logRsum/numChar
                RatioForScene.append(1/(logAver+1))
            else:
                #logAver = 0
                RatioForScene.append(0)
            #RatioForScene.append(np.average(np.asarray(Ratios[i])))
        #MaxLength = max(np.asarray(LengthForScene))
        #for i in range(self.NumOfScenes):
            #LengthForScene[i] = LengthForScene[i]/MaxLength

        #########################################
        SceneID = []
        ProImportance = []
        MentorImportance = []
        MentorLength = []
        MentorRatio = []
        ProLength = []
        ProRatio = []
        for i in range(self.NumOfScenes):
            SceneID.append(i + 1)
            MentorImportance.append(Importance[i][6])
            ProImportance.append(Importance[i][2])
            MentorLength.append(Length[i][6])
            ProLength.append(Length[i][2])
            MentorRatio.append(Ratios[i][6])
            ProRatio.append(Ratios[i][2])

        #print(RatioForScene)

        ProMaxLength = max(np.asarray(ProLength))
        for i in range(self.NumOfScenes):
            ProLength[i] = ProLength[i]/ProMaxLength

        MentorMaxLength = max(np.asarray(MentorLength))
        for i in range(self.NumOfScenes):
            MentorLength[i] = MentorLength[i]/MentorMaxLength




        f = open("C:/Users/OJ/Documents/2Imp.dat", mode='wt', encoding='utf-8')
        for i in range(self.NumOfScenes):
            f.write(str(i+1) + '    ' + str(ProImportance[i]) + '\n')
        f.close

        f = open("C:/Users/OJ/Documents/6Imp.dat", mode='wt', encoding='utf-8')
        for i in range(self.NumOfScenes):
            f.write(str(i+1) + '    ' + str(MentorImportance[i]) + '\n')
        f.close

        f = open("C:/Users/OJ/Documents/SLen.dat", mode='wt', encoding='utf-8')
        for i in range(self.NumOfScenes):
            f.write(str(i+1) + '    ' + str(LengthForScene[i]) + '\n')
        f.close

        f = open("C:/Users/OJ/Documents/2Len.dat", mode='wt', encoding='utf-8')
        for i in range(self.NumOfScenes):
            f.write(str(i+1) + '    ' + str(ProLength[i]) + '\n')
        f.close

        f = open("C:/Users/OJ/Documents/6Len.dat", mode='wt', encoding='utf-8')
        for i in range(self.NumOfScenes):
            f.write(str(i+1) + '    ' + str(MentorLength[i]) + '\n')
        f.close

        f = open("C:/Users/OJ/Documents/SRot.dat", mode='wt', encoding='utf-8')
        for i in range(self.NumOfScenes):
            f.write(str(i+1) + '    ' + str(RatioForScene[i]) + '\n')
        f.close

        f = open("C:/Users/OJ/Documents/2Rot.dat", mode='wt', encoding='utf-8')
        for i in range(self.NumOfScenes):
            f.write(str(i+1) + '    ' + str(ProRatio[i]) + '\n')
        f.close

        f = open("C:/Users/OJ/Documents/6Rot.dat", mode='wt', encoding='utf-8')
        for i in range(self.NumOfScenes):
            f.write(str(i+1) + '    ' + str(MentorRatio[i]) + '\n')
        f.close

        #plt.plot(SceneID,ProImportance,color='green',marker='o',linestyle='solid')
        #plt.plot(SceneID,MentorImportance,color='blue',marker='o',linestyle='solid')

        #plt.plot(SceneID,ProLength,color='red',marker='o',linestyle='solid')
        #plt.plot(SceneID,LengthForScene,color='blue',marker='o',linestyle='solid')
        #plt.plot(SceneID,ProRatio,color='gray',marker='o',linestyle='solid')
        #plt.plot(SceneID,RatioForScene,color='black',marker='o',linestyle='solid')

        #features = {'Scene Number': SceneID, 'Importance': ProImportance, 'Length': ProLength}
        #df = pd.DataFrame(features)
        #sns.barplot(x='Scene Number', y='Importance', data = df)
        #sns.barplot(x='Scene Number', y='Length', data = df)
        #plt.show()

        #print(ProImportance)

        return

    """
    def discretizeCharNet(self):
        gap = []
        rank = sorted(self.centrality.items(), key=operator.itemgetter(1), reverse=True)
        
        print(len(rank)) 
        print(self.NumOfChar)
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
            proximity = {n:0 for n in self.AcCharNetGraphs[i].edges()}
            BoundaryHigh = 0
            BoundaryMedium = 0
            
            #print(proximity)
            for speaker, listener, freq in self.AcCharNetGraphs[i].edges(data=True):
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
            
            for speaker, listener, freq in self.AcCharNetGraphs[i].edges(data=True):
                if freq['weight'] >= BoundaryHigh:
                    self.AcCharNetGraphs[i][speaker][listener]['weight'] = 'H'
                elif freq['weight'] >= BoundaryMedium:
                    self.AcCharNetGraphs[i][speaker][listener]['weight'] = 'I'
                else:
                    self.AcCharNetGraphs[i][speaker][listener]['weight'] = 'L'
            
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
"""



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
