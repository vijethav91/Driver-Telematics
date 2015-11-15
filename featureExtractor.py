import csv
import matplotlib.pyplot as plt
import numpy as np

class Features:
    # Threshold value to indicate that the vehicle is stopped
    stopThreshold = 1.5

    def __init__(self):

        self.distanceList = []
        self.totalDistance = 0

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

    def computeSpeedFeatures(self):
        # numpy array for distance list
        npDistList = np.array(self.distanceList)

        # Average speed
        self.meanSpeed = np.mean(npDistList)

        # Average speed when vehicle not stopped
        self.meanSpeedNotStopped = np.mean(npDistList[npDistList > self.stopThreshold])

        # Calculate standard deviation of instantaneous speed
        self.stdDevSpeed = np.std(npDistList)

        # Maximum speed
        self.maxSpeed = max(npDistList)

        # Speed percentiles
        self.speedPercentiles = self.computePercentiles(npDistList)

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

def readCsv(filename):
    data = []
    infile = open(filename, "r")
    temp = infile.read().split('\n')[1:-1]
    for line in temp:
        data.append(line.strip().split(','))

    return data

def plotTrip(data):
    X = [x[0] for x in data]
    Y = [x[1] for x in data]
    plt.plot(X, Y, 'ro')
    plt.show()

if __name__ == "__main__":
    filename = "drivers/1/1.csv"
    data = readCsv(filename)
    print data[0], data[-1]
    plotTrip(data)
