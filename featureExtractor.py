import csv
import matplotlib.pyplot as plt
import numpy as np
import os

class Features:
    baseDriversFolder = "drivers/"
    baseFeatureFolder = "features/"

    # Constructor to initialize the features
    def __init__(self, _id):
        self.driverTripId = _id
        self.distanceList = np.array([])
        self.totalDistance = 0
        self.totalTripTime = 0
        self.tripDisplacement = 0
        self.featuresList = np.array([])

    # Template function to compute all features
    def computeFeatures(self):
        self.computeDistanceFeatures()

    # Function to compute the features
    def computeDistanceFeatures(self):
        # construct the folder and trip file to read as per the driverTripId
        driverId, tripId = self.driverTripId.split('_')
        tripFileName = self.baseDriversFolder + driverId + '/' + tripId + '.csv'

        # read the file
        X, Y = self.readCsv(tripFileName)
        X = np.array(X)
        Y = np.array(Y)

        # computing total trip duration in seconds
        self.totalTripTime = len(X) - 1

        # computing the distance covered every second
        deltaX = np.diff(X)
        deltaY = np.diff(Y)

        sqFunc = np.vectorize(lambda x:x*x)
        sqX = sqFunc(deltaX)
        sqY = sqFunc(deltaY)

        self.distanceList = np.sqrt(sqX+sqY)
        self.totalDistance = np.sum(self.distanceList)

        self.tripDisplacement = np.sqrt( (X[-1]-X[0]) + (Y[-1]-Y[0]) )
        
    # Function to read data from csv and return a list of co-ordinate points.
    # Each point is represented as a list of two values [x, y].
    def readCsv(self, filename):
        X = []
        Y = []
        # open the file in read mode
        infile = open(filename, "r")

        # ignore the first line with label and the last empty line
        temp = infile.read().split('\n')[1:-1]
        for line in temp:
            _x, _y = line.strip().split(',')
            X.append(float(_x))
            Y.append(float(_y))

        # return our list of lists that captures all co-ordinate points for one trip
        return X, Y

    # Function to save the features to the corresponding driver file.
    def writeCsv(self):
        driverId, tripId = self.driverTripId.split('_')
        featureFileName = self.baseFeatureFolder + driverId + '.csv'
        print "Done"


def plotTrip(data):
    X = [x[0] for x in data]
    Y = [x[1] for x in data]
    plt.plot(X, Y, 'ro')
    plt.show()

if __name__ == "__main__":
    # Read the drivers folder to get the driver folder names
    driverData = os.listdir(Features.baseDriversFolder)
    numDrivers = len(driverData)

    for driver in driverData:
        # sanity check to skip '.DS_Store' file in Mac
        if driver.startswith('.'):
            continue
        
        # Read the individual driver folder to process all the trips
        tripData = os.listdir(Features.baseDriversFolder + driver)
        numTrips = len(tripData)

        for trip in tripData:
            # sanity check to skip '.DS_Store' file in Mac
            if trip.startswith('.'):
                continue

            tripId = trip.split('.')[0]
            
            # initialize a feature object
            genFeatures = Features(driver + '_' + tripId)

            # compute all the features
            genFeatures.computeFeatures()

            # save them to feature file
            genFeatures.writeCsv()

