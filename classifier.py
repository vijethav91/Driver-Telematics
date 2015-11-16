import csv
import numpy as np

class Classifier:
    baseFeatureFolder = "features/"

    def __init__(self):
        self.featuresHash = {}

    # Function to load the features from feature file
    def loadFeatures(self, _driverId):
        featureFileName = self.baseFeatureFolder + _driverId + '.csv'
        infile = open(featureFileName, 'r')
        infileReader = csv.reader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        infileReader.next()
        for line in infileReader:
            self.featuresHash[line[0]] = np.array(map(float, line[1:]))

if __name__ == "__main__":
    testClf = Classifier()
    testClf.loadFeatures("1")
    print testClf.featuresHash['1_1']
