from Models.Data import Data
from Models.LinearRegress import LinearRegress


def main():
    lr = LinearRegress(Data())
    lr.plot()
    r2 = lr.r2()
    print('R2: ', r2)


if __name__ == '__main__':
    main()