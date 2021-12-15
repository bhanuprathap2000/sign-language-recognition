import pandas as pd
import shutil
import os
import sys

def dataset_divide():
    include_50=pd.read_csv(r'D:/final year project/Train_Test_Split/test_include50.csv')
    #dataset path
    dataset_path =r'D:/final year project/dataset/'
    #new dataset path 
    dataset_path_new = r"D:/final year project/dataset_include50_test/"


    for Category, Word,Video,FilePath in include_50.values:
        path=FilePath.split('/')
        c=path[0]
        w=path[1]
        if not os.path.exists(dataset_path_new+c+"/"+w):
            os.makedirs(dataset_path_new+c+"/"+w)
        src_path = os.path.join(dataset_path,FilePath)
        dst_path = os.path.join(dataset_path_new,FilePath)
        # print(src_path)
        # print(dst_path)
        
        try:
            shutil.copy(src_path, dst_path)
            print("sucessful")
            
        except IOError as e:
            print(e)
            print('Unable to copy file {} to {}'.format(src_path, dst_path))
            
                
        except:
            print('When try copy file {} to {}, unexpected error: {}'.format(src_path, dst_path, sys.exc_info()))
    
dataset_divide()