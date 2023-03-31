import numpy as np
import numpy as np
from tqdm import tqdm
from FeatureExtractor import FeaturesExtractor
class AudioDataset:
    def __init__(self):
        self.train_data = []
        self.train_target = []
        self.fe = FeaturesExtractor()
        self.men_data = []
        self.men_target = []
        self.women_data = []
        self.women_target = []
    
    def load_data(self, path = "D:\python_projects\\3.2\MWDistinguisher\datasets\\train\\", path_ids = 'targets.txt'):
        file = open(path_ids)
        names = [line.split('\t')[0] for line in file]
        file.close()
        file = open(path_ids)
        targets = [int(line.split('\t')[1][0]) for line in file]
        file.close()
        for i in tqdm(range(0, len(names))):
            p = path + names[i] + ".wav"
            feature = self.fe.extract_features(p)
            self.train_data.append(feature)
            self.train_target.append(targets[i])

        self.train_data = np.array(self.train_data)
        self.train_target = np.array(self.train_target)
        print(self.train_data.shape, self.train_target.shape)

    def load_men(self, path = "D:\python_projects\\3.2\MWDistinguisher\datasets\\train\\", path_ids = 'men.txt'):
        file = open(path_ids)
        names = [line.split('\t')[0] for line in file]
        file.close()
        file = open(path_ids)
        targets = [int(line.split('\t')[1][0]) for line in file]
        file.close()
        features = np.asarray(())
        for i in tqdm(range(0, len(names))):
            p = path + names[i] + ".wav"
            vector    = self.fe.extract_features(p)
                # stack the features
            if features.size == 0:  features = vector
            else:                   features = np.vstack((features, vector))          
            self.men_target.append(targets[i])

        self.men_data = features
        self.men_target = np.array(self.men_target)
        print(self.men_data.shape, self.men_target.shape)

    def load_women(self, path = "D:\python_projects\\3.2\MWDistinguisher\datasets\\train\\", path_ids = 'women.txt'):
        file = open(path_ids)
        names = [line.split('\t')[0] for line in file]
        file.close()
        file = open(path_ids)
        targets = [int(line.split('\t')[1][0]) for line in file]
        file.close()
        features = np.asarray(())
        for i in tqdm(range(0, len(names))):
            p = path + names[i] + ".wav"
            vector    = self.fe.extract_features(p)
                # stack the features
            if features.size == 0:  features = vector
            else:                   features = np.vstack((features, vector))          
            self.women_target.append(targets[i])

        self.women_data = features
        self.women_target = np.array(self.women_target)
        print(self.women_data.shape, self.women_target.shape)

        

        
            