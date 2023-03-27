"""Algorithm of regression analysis"""

import lsm.approximation as aprx
import matplotlib.pyplot as plt


def get_average(y: list) -> float:
    """Calculate average of values

    :param y: list of values
    :return: average value (float)
    """
    return sum(y) / len(y)


def get_deviation(y: list) -> float:
    """Calculate deviation values from their mean

    :param y - list of values
    :return - deviation (float)
    """
    avg = get_average(y)
    return sum([i - avg for i in y])


def regression(x: list, y: list, accuracy=0.01):
    """Calculate regression of function y(x)

    :param accuracy: the accuracy with which
    eps1 and eps2 are considered close
    :param x: list of arguments
    :param y: list of values with deviations

    """
    if accuracy <= 0:
        raise ValueError("accuracy must be > 0")

    degree = 1
    eps1 = get_deviation(y)
    eps2 = eps1
    poly = None

    while abs(eps1 - eps2) < accuracy:
        poly = aprx.approx_poly(x, y, degree)
        eps2 = aprx.discrepancy(y, poly(x))
        degree += 1

    return poly


def main():
    """Calculate regression and plotting"""

    x, y = aprx.read_data("reg_data.txt")

    regres = regression(x, y)

    plt.plot(x, y, "ro")
    plt.plot(x, regres(x))
    plt.show()


if __name__ == '__main__':
    main()
