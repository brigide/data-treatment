import pandas as pd


class Data:
    def __init__(self):
        self.data = pd.read_csv('registro01.csv', header=(0))
        self.data = self.data.dropna().drop_duplicates()

        self.ylabel = self.data.columns[-1]
        self.shape = self.data.shape

        self.nrow, self.ncol = self.shape

    def getData(self):
        return self.data.to_numpy()

    def getYlabel(self):
        return self.ylabel

    def getShape(self):
        return self.shape

    def getX(self):
        return self.data.to_numpy()[:,0:self.ncol-1]

    def getY(self):
        return self.data.to_numpy()[:,-1]
