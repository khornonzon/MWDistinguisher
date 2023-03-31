from hmmlearn import hmm
from FeatureExtractor import FeaturesExtractor
import numpy as np
import librosa
from tqdm import tqdm
import pickle
from AudioDataset import AudioDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from hmmlearn import hmm


 
class Teacher:
    def __init__(self):
        self.dataset = AudioDataset()
        self.knn = KNeighborsClassifier()
        self.svc = SVC(kernel='rbf')
    def training(self, path = "D:\python_projects\\3.2\MWDistinguisher\datasets\\train\\"):
        self.dataset.load_data(path)
        X_train, X_test, y_train, y_test = train_test_split(self.dataset.train_data, self.dataset.train_target,test_size=0.1, random_state=22)
        self.knn.fit(X_train, y_train)
        self.svc.fit(X_train, y_train)
        print(self.knn.score(X_test, y_test))
        print(self.svc.score(X_test, y_test))
        self.save_model(self.knn, 'knn_model.pickle')
        self.save_model(self.svc, 'svc_model.pickle')


    def save_model(self, model, model_name):
        pickle.dump(model, open(model_name, 'wb'))
