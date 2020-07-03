# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:09:12 2020

@author: Ivan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:39:34 2020
@author: AIC-WS1
"""

import os
import pandas as pd
import re 

def sortPics():
    print('transfering pics...')
    files = os.listdir('./')
    for file in files:
        if file.endswith('.txt'):    
            print(file)
            viewType = './'+file
            foldername = re.search('./(.+?).txt',viewType).group(1)
            imageFolder = './forboxing/1/'
            labelsFolder = './newlabels/1/'
            
            orgnzd_folders = './organized/'+foldername
            
            if not os.path.exists('./organized'):
                os.mkdir('./organized')
                
            if not os.path.exists(orgnzd_folders):
                os.mkdir(orgnzd_folders)
            
            if not os.path.exists(orgnzd_folders+'/yoloV3Labels'):
                os.mkdir(orgnzd_folders+'/yoloV3Labels')
            
            try:
                df = pd.read_csv(viewType,sep = ',',header=None)
                filenames = df[1].values
                
                for filename in filenames:
                    try:
                        os.rename(imageFolder+filename+'.jpg',
                                        orgnzd_folders+'/'+filename+'.jpg')
                        os.rename(labelsFolder+filename+'.txt',
                                        orgnzd_folders+'/yoloV3Labels/'+filename+'.txt')
                    except FileNotFoundError:
                        print(filename, 'already transfered')
                        
            except pd.errors.EmptyDataError:
                print('Empty file Proceding...')
