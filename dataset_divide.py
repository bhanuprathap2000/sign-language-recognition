import pandas as pd
import shutil
import os
import sys

def dataset_divide(dataset_path,dataset_path_new):

    include_50=pd.read_csv(r'D:/final year project/Train_Test_Split/train_include50.csv')

    for Category, Word,Video,FilePath in include_50.values:
        
        word_folder=os.path.join(dataset_path_new,FilePath)[:-13]#remove the filename and extension for folder creation
        #before D:/final year project/dataset/Animals/4. Bird/MVI_2987.MOV
        #after D:/final year project/dataset/Animals/4. Bird

        #if folder doesn't exisit, create it
        if not os.path.exists(word_folder):
            print("creating folder:",word_folder)
            os.makedirs(word_folder)
        
        #src and dst paths
        src_path = os.path.join(dataset_path,FilePath)
        dst_path = os.path.join(dataset_path_new,FilePath)
        
        #try to copy the file from src to dst path 
        try:
            shutil.copy(src_path, dst_path)
            print("sucessful")
        #if there is an expection raised, print it 
        except IOError as e:
            print(e)
            print('Unable to copy file {} to {}'.format(src_path, dst_path))   
        except:
            print('When try copy file {} to {}, unexpected error: {}'.format(src_path, dst_path, sys.exc_info()))


# to this we need to pass the paths
dataset_divide(r'D:/final year project/dataset/',r"D:/final year project/dataset_include50_test/")