import csv
import matplotlib.pyplot as plt
import numpy as np
<<<<<<< HEAD
import os

class Features:
    baseDriversFolder = "drivers/"
    baseFeatureFolder = "features/"
    # Threshold value to indicate that the vehicle is stopped
    stopThreshold = 1.5

    # Constructor to initialize the features
    def __init__(self, _id):
        self.driverTripId = _id
        self.distanceList = np.array([])
        self.totalDistance = 0
        self.totalTripTime = 0
        self.tripDisplacement = 0
        self.featuresList = np.array([])
        # Speed features
        self.meanSpeed = 0
        self.meanSpeedNotStopped = 0
        self.stdDevSpeed = 0
        self.maxSpeed = 0
        self.speedPercentiles = [] # 5th, 10th, 25th, 50th, 75th, 85th, 90th, 95th, 97th, 98th, 99th, 100th percentiles of speed
        # Acceleration features
        self.meanAcceleration = 0
        self.stdDevAcceleration = 0
        self.accelerationPercentiles = []
        self.meanPosAcceleration = 0
        self.stdDevPosAcceleration = 0
        self.posAccelerationPercentiles = []
        self.meanNegAcceleration = 0
        self.stdDevNegAcceleration = 0
        self.negAccelerationPercentiles = []

    # Template function to compute all features
    def computeFeatures(self):
        self.computeDistanceFeatures()
        self.computeSpeedFeatures()
        self.computeAccelerationFeatures()

    # Function to compute the distance features
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

    # Function to compute the speed features
    def computeSpeedFeatures(self):
        # Average speed
        self.meanSpeed = np.mean(self.distanceList)

        # Average speed when vehicle not stopped
        self.meanSpeedNotStopped = np.mean(self.distanceList[self.distanceList > self.stopThreshold])

        # Calculate standard deviation of instantaneous speed
        self.stdDevSpeed = np.std(self.distanceList)

        # Maximum speed
        self.maxSpeed = max(self.distanceList)

        # Speed percentiles
        self.speedPercentiles = self.computePercentiles(self.distanceList)

    # Function to compute the acceleration features
    def computeAccelerationFeatures(self):
        # Instantaneous acceleration
        instantAcceleration = np.diff(self.distanceList)

        # Mean, std dev, percentiles of overall acceleration
        self.meanAcceleration = np.mean(instantAcceleration)
        self.stdDevAcceleration = np.std(instantAcceleration)
        self.accelerationPercentiles = self.computePercentiles(instantAcceleration)

        # Mean, std dev, percentiles of positive acceleration
        positiveAcceleration = instantAcceleration[instantAcceleration > 0]
        self.meanPosAcceleration = np.mean(positiveAcceleration)
        self.stdDevPosAcceleration = np.std(positiveAcceleration)
        self.posAccelerationPercentiles = self.computePercentiles(positiveAcceleration)

        # Mean, std dev, percentiles of negative acceleration
        negativeAcceleration = instantAcceleration[instantAcceleration < 0]
        self.meanNegAcceleration = np.mean(negativeAcceleration)
        self.stdDevNegAcceleration = np.std(negativeAcceleration)
        self.negAccelerationPercentiles = self.computePercentiles(negativeAcceleration)

    # Function to compute percentiles
    def computePercentiles(self, values):
        percentiles = [ np.percentile(values, 5),
                        np.percentile(values, 10),
                        np.percentile(values, 25),
                        np.percentile(values, 50),
                        np.percentile(values, 75),
                        np.percentile(values, 85),
                        np.percentile(values, 90),
                        np.percentile(values, 95),
                        np.percentile(values, 97),
                        np.percentile(values, 98),
                        np.percentile(values, 99),
                        np.percentile(values, 100) ]
        return percentiles

    # Print feature values
    def printFeatures(self):
        print "Speed"
        print self.meanSpeed
        print self.meanSpeedNotStopped
        print self.stdDevSpeed
        print self.maxSpeed
        print self.speedPercentiles

        print "Acceleration"
        print self.meanAcceleration
        print self.stdDevAcceleration
        print self.accelerationPercentiles
        print self.meanPosAcceleration
        print self.stdDevPosAcceleration
        print self.posAccelerationPercentiles
        print self.meanNegAcceleration
        print self.stdDevNegAcceleration
        print self.negAccelerationPercentiles

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
