from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import numpy as np


class LinearRegress:
    def __init__(self, data):
        self.data = data
        self.p = 0.3

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data.getX(), self.data.getY(), test_size=self.p, random_state=42)

        self.lm = LinearRegression()
        self.lm.fit(self.x_train, self.y_train)

        self.y_pred = self.lm.predict(self.x_test)
        self.l = plt.plot(self.y_pred, self.y_test, 'bo')

    def plot(self):
        plt.setp(self.l, markersize=10)
        plt.setp(self.l, markerfacecolor='C0')
        plt.ylabel("y", fontsize=15)
        plt.xlabel("Prediction", fontsize=15)

        xl = np.arange(min(self.y_test), 1.2*max(self.y_test),(max(self.y_test)-min(self.y_test))/10)
        yl = xl

        plt.plot(xl, yl, 'r--')
        plt.show()


