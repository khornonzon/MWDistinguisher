import pickle
from logic.FeatureExtractor import FeaturesExtractor
import numpy as np
from tqdm import tqdm
from logic.Teacher import Teacher
from sklearn.neighbors import KNeighborsClassifier
import os
class Identifier:
    def __init__(self, pg):
        self.model: KNeighborsClassifier = pickle.load(open(pg, 'rb'))
        self.fe = FeaturesExtractor()


    def recognize_gender(self, path):
        feature = np.array(self.fe.extract_features(path)).reshape(1,-1)
        prediction = self.model.predict(feature)
        if prediction[0]==0: return "This is man"
        else: return "This is woman"
    def train_dataset_identification(self, path_names, path_files):
        file = open(path_names, 'r')
        names = [line.split('\t')[0] for line in file]
        file.close()
        file = open(path_names, 'r')
        targets = [int(line.split('\t')[1][0]) for line in file]
        file.close()
        score = 0
        amount = len(names)

        for i in tqdm(range(len(names))): 
             p = path_files + names[i] + ".wav"
             result = self.recognize_gender(p)
             if result == targets[i]:
                 score+=1
        accuracy = score/amount
        print(f'Accuracy is {accuracy}')
        return accuracy
    def test_dataset_prediction(self, path):
        listOfFiles = os.listdir(path)  
        pattern = "*.wav"  
        out = open('test_result.txt', 'w')
        for entry in listOfFiles:  
            feature = np.array(self.fe.extract_features(path+'\\'+entry)).reshape(1,-1)

            out.write(entry+'\t'+ str(self.knn_model.predict(feature))+ '\n')







        

        
        