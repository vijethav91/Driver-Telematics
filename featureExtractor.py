import csv
import matplotlib.pyplot as plt

class Features:
    def __init__(self):
        self.distanceList = []
        self.totalDistance = 0

    #def computeFeatures(self, XY):


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
