# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:56:04 2018

@author: O-Joun Lee
"""

#import urllib2
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import re
import string
from xml.etree.ElementTree import Element, SubElement, dump
from lxml import etree
#import urllib2


class ScriptParser:
    """
    Read CharNets and Convert to Graphs
    """
    def __init__(self):
        
        #self.nodes = self.graph.nodes()
        self.ScriptPaths = []
        #path = "https://www.imsdb.com/scripts/Kung-Fu-Panda.html"
        #self.ScriptPaths.insert(1, "https://www.imsdb.com/scripts/Good-Will-Hunting.html")
        
        self.ScriptPaths.append(['GOOD WILL HUNTING', "https://www.imsdb.com/scripts/Good-Will-Hunting.html"]) 
        
        #self.selectedScripts = []
        
        #self.parseListOfScripts()
        self.readScript()
        
    """    
    def parseListOfScripts(self):
        
        Path = "https://www.imsdb.com/all%20scripts/"
        html = requests.get(Path).text
        
        soup = BeautifulSoup(html, 'html.parser')
        
        Titles = soup.select('p > a')
        
        for title in Titles:
            path = title.get('href')
            path = path.replace("/Movie Scripts", "/scripts")
            path = path.replace(" Script.html", ".html")
            path = "https://www.imsdb.com" + path
            
            self.ScriptPaths.append([title.text, path])
        
        return
    """

    def readScript(self):
        #tree = parse("C:\Users\O-Joun Lee\Story2Vec\CharNetsInXML\Kung-Fu-Panda_characterNet.xml")
        ScriptNum = 0
        
        for path in self.ScriptPaths:
            
            try:
                html = requests.get(path[1])
            except requests.HTTPError as e:
                self.ScriptPaths.remove(path)
                continue
            
            #html.raise_for_status()
            html.headers['content-type']
            #html.encoding
            
            html = html.text
            #html = html.encode('CP949')
            
            #request = urllib2.Request(path[1])
            #request.add_header('Accept-Encoding', 'utf-8')
            #response = urllib2.urlopen(request)
            
            soup = BeautifulSoup(html, 'html.parser')
            Entities = soup.select('pre > b')
            Script = []
            Descriptions = []
            pattern = re.compile(r'\s+')
            
            #print(Entities[7].next_sibling)
            #print(Entities[7].next_sibling.split('\n'))

            for entity in Entities: 
                if len(entity.text) > 2: 
                    #if len(entity.text.translate({ ord(c):None for c in string.whitespace })) > 0: 
                    if len(entity.text.strip()) > 0: 
                        Script.append(entity)
                        Descriptions.append(str(entity.next_sibling).split('\n'))
            
            if len(Script) < 15:
                self.ScriptPaths.remove(path)
                continue
            
            ###########################
            Dgaps = []
            
            for des in Descriptions: 
                for d in des:
                    gap = len(d) - len(d.lstrip())
                    Dgaps.append(gap)
            
            #print(Dgaps)

            Dgaps = np.asarray(Dgaps, dtype=np.float32)
            Dgaps = Dgaps.reshape(-1, 1)
            averageGap = np.average(Dgaps)
            
            NormDescription = [[]]
            i = 0
            
            for des in Descriptions: 
                NormDes = []
                for d in des:
                    if Dgaps[i] > averageGap:
                        NormDes.append([d, float(Dgaps[i]), 1, len(d.strip().split(' '))]) #dialogues
                    else:
                        NormDes.append([d, float(Dgaps[i]), 0, len(d.strip().split(' '))]) #descriptions
                    i = i + 1
                NormDescription.append(NormDes)

            DialLength = []
            for des in NormDescription: 
                NumWords = 0
                for d in des:
                    if d[2] == 1:
                        NumWords = NumWords + d[3]
                DialLength.append(NumWords)

            #print(DialLength)
            ###########################

            gaps = []
            
            for entity in Script: 
                gap = len(entity.text) - len(entity.text.lstrip())
                gaps.append(gap)
                
            gaps = np.asarray(gaps, dtype=np.float32)
            gaps = gaps.reshape(-1, 1)
            CentInit = np.array([[min(gaps)], [np.median(gaps)], [max(gaps)]], dtype=np.float32) 
            
            kmeans = KMeans(n_clusters=3, init = CentInit).fit(gaps)
            kmeans.fit(gaps)
            predict = pd.DataFrame(kmeans.predict(gaps))
            predict.columns=['predict']
            
            gaps = pd.DataFrame(gaps)
            gaps.columns=['gaps']
            gapsClusters = pd.concat([gaps,predict],axis=1)
            
            NormScript = []
            i = 0
            
            for entity in Script:
                NormScript.append([entity.text, gapsClusters.iloc[i]['gaps'], gapsClusters.iloc[i]['predict'], DialLength[i]])
                i = i + 1
            
            

            #print(NormScript)


            ########################################
            
            SceneNum = 0
            CharNum = 0
            CharDic = {}
            condition = re.compile(r'\(.*?\)|[^\s]+')
            
            for i in range(len(NormScript)):
                matches = re.findall(condition, NormScript[i][0])
                NormScript[i][0] = " ".join([x for x in matches if "(" not in x])
                if NormScript[i][2] == 1:
                    NormScript[i][0] = re.sub('[^A-Za-z0-9/]+', '', NormScript[i][0])
                
                
            for entity in NormScript:
                
                #matches = re.findall(condition, entity[0])
                #entity[0] = " ".join([x for x in matches if "(" not in x])
                #for match in matches:
                #    entity[0] = entity[0].replace('(' + match + ')', '')
                    
                if re.sub(pattern, '', entity[0]) == 'THEEND':
                    entity[2] = 4
                    continue
                if re.sub(pattern, '', entity[0]) == re.sub(pattern, '', path[0]):
                    NormScript.remove(entity)
                    continue
                if len(re.sub('[^A-Za-z0-9]+', '', entity[0])) < 1:
                    NormScript.remove(entity)
                    continue   
                if entity[0].isdigit():
                    NormScript.remove(entity)
                    continue
                if entity[0].strip() == '':
                    NormScript.remove(entity)
                    continue
                if (entity[0].strip() == path[0].strip()):
                    entity[2] = 5
                    continue
                if (entity[0].strip() == 'FADE IN:'):
                    entity[2] = 4
                    continue
                if (entity[0].strip() == 'CUT TO:'):
                    entity[2] = 4
                    continue
                if (entity[0].strip() == 'BLACKOUT:'):
                    entity[2] = 4
                    continue
                if (entity[0].strip() == 'DISSOLVE TO:'):
                    entity[2] = 4
                    continue
                if (entity[0].strip() == 'TIME DISSOLVE TO:'):
                    entity[2] = 4
                    continue
                
            for entity in NormScript:
                if entity[2] == 1:
                    #print('name \n', entity[0])
                    if not entity[0].strip() in CharDic:
                        CharDic[entity[0].strip()] = CharNum
                        CharNum = CharNum + 1
                        
                elif entity[2] == 0:
                    #print('scene \n', entity[0])
                    #SceneNum = SceneNum + 1
                    if 'INT.' in entity[0]: 
                        SceneNum = SceneNum + 1
                    elif 'EXT.' in entity[0]: 
                        SceneNum = SceneNum + 1
                    else:
                        entity[2] = 4

                #else:
                    #print('instruction \n', entity[0]) 
                    
            #print(SceneNum)
            #print(CharNum)
            print(CharDic)
            #print(path[0])
            #print(NormScript)
            
            
            
            ALLCOUNT = 0
            SceneID = 0
            OccurrenceChars = []
            NumWordsChars = []
            for i in range(len(NormScript)):
                if NormScript[i][2] == 0:
                    if SceneID == 0:
                        #OccurrenceChars.append(OccurrenceInScene)
                        OccurrenceInScene = {}
                        NumWordsInScene = {}
                        ALLCOUNT = 0
                        ALLNUmWords = 0
                    else: 
                        for Char in OccurrenceInScene:
                            if OccurrenceInScene[Char] > 0:
                                OccurrenceInScene[Char] = OccurrenceInScene[Char] + ALLCOUNT
                                NumWordsInScene[Char] = NumWordsInScene[Char] + ALLNUmWords
                        OccurrenceChars.append(OccurrenceInScene)
                        NumWordsChars.append(NumWordsInScene)
                        OccurrenceInScene = {}
                        NumWordsInScene = {}
                        ALLCOUNT = 0
                        ALLNUmWords = 0
                    SceneID = SceneID + 1

                elif NormScript[i][2] == 1: 
                    if NormScript[i][0].strip() == 'ALL':
                        ALLCOUNT = ALLCOUNT + 1
                        ALLNUmWords = ALLNUmWords + NormScript[i][3]
                    elif '/' in NormScript[i][0].strip():
                        CharList = NormScript[i][0].strip().split('/')
                        for char in CharList:
                            OccurrenceInScene[CharDic[char]] += 1
                            NumWordsInScene[CharDic[char]] += NormScript[i][3]
                    elif CharDic[NormScript[i][0].strip()] in OccurrenceInScene: 
                        OccurrenceInScene[CharDic[NormScript[i][0].strip()]] += 1
                        NumWordsInScene[CharDic[NormScript[i][0].strip()]] += NormScript[i][3]
                    else:
                        OccurrenceInScene[CharDic[NormScript[i][0].strip()]] = 1
                        NumWordsInScene[CharDic[NormScript[i][0].strip()]] = NormScript[i][3]
                if (i == len(NormScript) - 1):
                    OccurrenceChars.append(OccurrenceInScene)
                    NumWordsChars.append(NumWordsInScene)
                #elif NormScript[i][2] == 0:
                    #OccurrenceChars.append(OccurrenceInScene)
            
            print(len(OccurrenceChars))
            print(OccurrenceChars)

            print(len(NumWordsChars))
            print(NumWordsChars)
            
            """
            if len(OccurrenceChars) != SceneNum:
                if len(OccurrenceChars) < SceneNum:
                    SceneNum -=1
                else:
                    OccurrenceChars.pop()
                print(len(OccurrenceChars), SceneNum)
                #break
            """

            CharNet = np.zeros((CharNum,CharNum))
            DialCharNet = np.zeros((CharNum,CharNum))
            #NormSceneNum = SceneNum
            #emptyScenes = []
            
            """
            for l in range(SceneNum): 
                if not OccurrenceChars[l]:
                    NormSceneNum -= 1
                    emptyScenes.append(l)

            for i in reversed(range(len(emptyScenes))): 
                OccurrenceChars.remove(OccurrenceChars[emptyScenes[i]])
            
            SceneNum = NormSceneNum
            """

            if (SceneNum < 3) or (CharNum < 3):
                self.ScriptPaths.remove(path)
                continue
            
            AcCharNets = []
            CharNets = []
            AcDialCharNets = []
            DialCharNets = []

            for l in range(SceneNum): 
                CurCharNet = np.zeros((CharNum,CharNum))
                CurDialCharNet = np.zeros((CharNum,CharNum))
                for i in range(CharNum): 
                    if i in OccurrenceChars[l]:
                        for j in range(CharNum): 
                            if j in OccurrenceChars[l]:
                                CurCharNet[i,j] += OccurrenceChars[l][i]
                                CurDialCharNet[i,j] += NumWordsChars[l][i]
                CharNet = CharNet + CurCharNet
                DialCharNet = DialCharNet + CurDialCharNet
                AcCharNets.append(CharNet)
                CharNets.append(CurCharNet)
                AcDialCharNets.append(DialCharNet)
                DialCharNets.append(CurDialCharNet)
                
                
            #print(AcCharNets)
            #print(AcDialCharNets)
            #print(SceneNum)
            
            root = etree.Element("characterNetwork")
            characterList = etree.SubElement(root, "characterList")
            ErrorCharacterList = etree.SubElement(root, "EcharacterList")

            charList = ''
            for char in CharDic:
                #charList = charList + str(char) + ': ' + str(CharDic[char]) + '  '
                charList = charList + str(char) + '  '
            characterList.text = charList

            EcharList = ''
            for char in CharDic:
                if char == 'ALL':
                    EcharList = EcharList + str(CharDic[char]) + '  '
                elif '/' in char:
                    EcharList = EcharList + str(CharDic[char]) + '  '
            ErrorCharacterList.text = EcharList

            
            for l in range(SceneNum): 
                characterNet = etree.SubElement(root, "characterNet")
                sceneNumber = etree.SubElement(characterNet, "sceneNum")
                sceneNumber.text = str(l + 1)
                AccuCharNet = etree.SubElement(characterNet, "accumulativeCharNet")
                DisCharNet = etree.SubElement(characterNet, "discreteCharNet")
                AccuDialCharNet = etree.SubElement(characterNet, "accumulativeDialCharNet")
                DisDialCharNet = etree.SubElement(characterNet, "discreteDialCharNet")
                
                AcCN = ''
                for i in range(CharNum): 
                    for j in range(CharNum): 
                        AcCN = AcCN + str(int(AcCharNets[l][i,j])) + '  '
                    AcCN = AcCN + '//'
                AccuCharNet.text = AcCN

                DsCN = ''
                for i in range(CharNum): 
                    for j in range(CharNum): 
                        DsCN = DsCN + str(int(CharNets[l][i,j])) + '  '
                    DsCN = DsCN + '//'
                DisCharNet.text = DsCN

                AcDCN = ''
                for i in range(CharNum): 
                    for j in range(CharNum): 
                        AcDCN = AcDCN + str(int(AcDialCharNets[l][i,j])) + '  '
                    AcDCN = AcDCN + '//'
                AccuDialCharNet.text = AcDCN

                DsDCN = ''
                for i in range(CharNum): 
                    for j in range(CharNum): 
                        DsDCN = DsDCN + str(int(DialCharNets[l][i,j])) + '  '
                    DsDCN = DsDCN + '//'
                DisDialCharNet.text = DsDCN
                
                root.append(characterNet)
                characterNet.append(sceneNumber)
                characterNet.append(AccuCharNet)
                characterNet.append(DisCharNet)
                characterNet.append(AccuDialCharNet)
                characterNet.append(DisDialCharNet)
                #print('\n\n\n',AcCN)
            
            #etree.write('./XML/' + path[0] + '_sample.xml')

            x_output = etree.tostring(root, pretty_print=True, encoding='UTF-8')
            x_header = '<?xml version="1.0" encoding="UTF-8"?>\n'
            ff=open('C:/Users/OJ/Documents/XML/' + path[0] + '_sample.xml', 'w', encoding="utf-8")
            ff.write(x_header + x_output.decode('utf-8') )
            
            #break
                
                
            
            ScriptNum = ScriptNum + 1
            #break
                
        return 
    
    
def get_html(url):
   _html = ""
   resp = requests.get(url)
   if resp.status_code == 200:
      _html = resp.text
   return _html

parser = ScriptParser()
#parser.parseListOfScripts()