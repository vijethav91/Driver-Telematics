from __future__ import division

import csv
import matplotlib.pyplot as plt
import numpy as np
import os

class Features:
    baseDriversFolder = "drivers/"
    baseFeatureFolder = "features/"
    # Threshold value to indicate that the vehicle is stopped
    stopThreshold = 1.5

    # Constructor to initialize the features
    def __init__(self, _id):
        self.driverTripId = _id
        self.featureList = []
        # Distance Features
        self.distanceList = np.array([])
        self.totalDistance = 0
        self.totalTripTime = 0
        self.tripDisplacement = 0
        self.totalStandingTime = 0
        self.stopRatio = 0
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
        self.meanJerk = 0
        self.stdDevJerk = 0
        self.jerkPercentiles = []
        self.meanAngle = 0
        self.stdDevAngle = 0
        self.anglePercentiles = []

    # Template function to compute all features
    def computeFeatures(self):
        self.computeDistanceFeatures()
        self.computeSpeedFeatures()
        self.computeAccelerationFeatures()
        self.composeFeatures()

    # Function to build the featureList for saving to csv
    def composeFeatures(self):
        self.featureList = [self.driverTripId, self.totalDistance, self.totalTripTime, self.tripDisplacement,
                            self.totalStandingTime, self.stopRatio, self.meanSpeed,  self.meanSpeedNotStopped,
                            self.stdDevSpeed, self.maxSpeed
                            ]
        self.featureList.extend(self.speedPercentiles)
        self.featureList.extend([self.meanAcceleration, self.stdDevAcceleration])
        self.featureList.extend(self.accelerationPercentiles)
        self.featureList.extend([self.meanPosAcceleration, self.stdDevPosAcceleration])
        self.featureList.extend(self.posAccelerationPercentiles)
        self.featureList.extend([self.meanNegAcceleration, self.stdDevNegAcceleration])
        self.featureList.extend(self.negAccelerationPercentiles)
        self.featureList.extend([self.meanJerk, self.stdDevJerk])
        self.featureList.extend(self.jerkPercentiles)
        self.featureList.extend([self.meanAngle, self.stdDevAngle])
        self.featureList.extend(self.anglePercentiles)

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
        self.deltaX = np.diff(X)
        self.deltaY = np.diff(Y)

        sqFunc = np.vectorize(lambda x:x*x)
        sqX = sqFunc(self.deltaX)
        sqY = sqFunc(self.deltaY)

        self.distanceList = np.sqrt(sqX+sqY)
        self.totalDistance = np.sum(self.distanceList)
        self.tripDisplacement = np.sqrt(pow(X[-1]-X[0], 2) + pow(Y[-1]-Y[0], 2))

        stopFunc = np.vectorize(lambda x: x < self.stopThreshold)

        self.totalStandingTime = np.sum(stopFunc(self.distanceList))
        self.stopRatio = self.totalStandingTime / self.totalTripTime

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

        # Derivative of acceleration (jerk)
        jerk = np.diff(instantAcceleration)
        self.meanJerk = np.mean(jerk)
        self.stdDevJerk = np.std(jerk)
        self.jerkPercentiles = self.computePercentiles(jerk)

    # Function to compute turning angle features
    def computeAngleFeatures(self):
        turningAngle=(self.deltaY[1:]*self.deltaY[:-1]+self.deltaX[1:]*self.deltaX[:-1])/(self.distanceList[:-1]*self.distanceList[1:])
        self.meanAngle = np.mean(turningAngle)
        self.stdDevAngle = np.mean(turningAngle)
        self.anglePercentiles = self.computePercentiles(turningAngle)

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
        X, Y = np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True)

        # return our list of lists that captures all co-ordinate points for one trip
        return X, Y

    # Function to save the features to the corresponding driver file.
    def writeCsv(self, csvWriter):
        csvWriter.writerow(self.featureList)

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

        print "Processing for driver: ", driver

        # Read the individual driver folder to process all the trips
        tripData = os.listdir(Features.baseDriversFolder + driver)

        # Open a feature file for a driver to write all the trip features as a csv
        featureFileName = Features.baseFeatureFolder + driver + '.csv'
        outFile = open(featureFileName, 'wb')
        featureWriter = csv.writer(outFile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvHeader = ['driverTripId', 'totalDistance', 'totalTripTime', 'tripDisplacement',
                                 'totalStandingTime', 'stopRatio', 'meanSpeed', 'meanSpeedNotStopped',
                                 'stdDevSpeed', 'maxSpeed', 'speedPercentiles5th', 'speedPercentiles10th',
                                 'speedPercentiles25th', 'speedPercentiles50th', 'speedPercentiles75th',
                                 'speedPercentiles85th', 'speedPercentiles90th', 'speedPercentiles95th',
                                 'speedPercentiles97th', 'speedPercentiles98th', 'speedPercentiles99th',
                                 'speedPercentiles100th', 'meanAcceleration', 'stdDevAcceleration',
                                 'accelerationPercentiles5th', 'accelerationPercentiles10th', 'accelerationPercentiles25th',
                                 'accelerationPercentiles50th', 'accelerationPercentiles75th',
                                 'accelerationPercentiles85th', 'accelerationPercentiles90th', 'accelerationPercentiles95th',
                                 'accelerationPercentiles97th', 'accelerationPercentiles98th', 'accelerationPercentiles99th',
                                 'accelerationPercentiles100th', 'meanPosAcceleration', 'stdDevPosAcceleration',
                                 'posAccelerationPercentiles5th', 'posAccelerationPercentiles10th',
                                 'posAccelerationPercentiles25th', 'posAccelerationPercentiles50th', 'posAccelerationPercentiles75th',
                                 'posAccelerationPercentiles85th', 'posAccelerationPercentiles90th', 'posAccelerationPercentiles95th',
                                 'posAccelerationPercentiles97th','posAccelerationPercentiles98th', 'posAccelerationPercentiles99th',
                                 'posAccelerationPercentiles100th', 'meanNegAcceleration', 'stdDevNegAcceleration',
                                 'negAccelerationPercentiles5th', 'negAccelerationPercentiles10th',
                                 'negAccelerationPercentiles25th', 'negAccelerationPercentiles50th', 'negAccelerationPercentiles75th',
                                 'negAccelerationPercentiles85th', 'negAccelerationPercentiles90th', 'negAccelerationPercentiles95th',
                                 'negAccelerationPercentiles97th', 'negAccelerationPercentiles98th', 'negAccelerationPercentiles99th',
                                 'negAccelerationPercentiles100th', 'meanJerk', 'stdDevJerk',
                                 'jerkPercentiles5th', 'jerkPercentiles10th',
                                 'jerkPercentiles25th', 'jerkPercentiles50th', 'jerkPercentiles75th',
                                 'jerkPercentiles85th', 'jerkPercentiles90th', 'jerkPercentiles95th',
                                 'jerkPercentiles97th', 'jerkPercentiles98th', 'jerkPercentiles99th',
                                 'jerkPercentiles100th', 'meanAngle', 'stdDevAngle',
                                 'anglePercentiles5th', 'anglePercentiles10th',
                                 'anglePercentiles25th', 'anglePercentiles50th', 'anglePercentiles75th',
                                 'anglePercentiles85th', 'anglePercentiles90th', 'anglePercentiles95th',
                                 'anglePercentiles97th', 'anglePercentiles98th', 'anglePercentiles99th',
                                 'anglePercentiles100th',
                               ]
        featureWriter.writerow(csvHeader)

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
            genFeatures.writeCsv(featureWriter)

        # Close the feature file after processing
        outFile.close()
    exit()
