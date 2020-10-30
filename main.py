from Models.Data import Data
from Models.LinearRegress import LinearRegress


def main():
    lr = LinearRegress(Data())
    lr.plot()


if __name__ == '__main__':
    main()