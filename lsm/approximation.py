"""Least-squares function approximation
"""
import numpy as np
import matplotlib.pyplot as plt


def create_system(x: list, y: list, degree: int) -> list:
    """Create system of linear equations for finding
    the coefficients of a polynomial

    :param x - list of objective function arguments
    :param y - list of objective function values
    :param degree - polynomial degree
    :return - augmented matrix of the system (list)
    """

    if len(x) != len(y):
        raise ValueError("x and y length must be the same length")
    if degree < 1:
        raise ValueError("degree must be >= 1")

    augment_matrix = []

    for k in range(degree + 1):
        augment_matrix.append([])
        for j in range(degree + 1):
            augment_matrix[k].append(0)
            for i in range(len(x)):
                augment_matrix[k][j] += x[i] ** (j + k)
        augment_matrix[k].append(0)
        for i in range(len(x)):
            augment_matrix[k][degree + 1] += y[i] * (x[i] ** k)

    return augment_matrix


def solve_system(system: list) -> np.ndarray:
    """Solution of a system of linear equations

    :param system - augmented matrix of the system
    :return - solution vector (np.ndarray)
    """
    system = np.array(system)
    column_count = system.shape[1]
    coefficients = system[:, 0:column_count - 1]
    constant_terms = system[:, column_count - 1]
    return np.linalg.solve(coefficients, constant_terms)


def approx_poly(x: list, y: list, degree: int) -> np.poly1d:
    """Approximation of the function by a polynomial

    :param x - list of objective function arguments
    :param y - list of objective function values
    :param degree - polynomial degree
    :return - polynomial (np.poly1d)
    """

    if len(x) != len(y):
        raise ValueError("x and y length must be the same length")
    if degree < 1:
        raise ValueError("degree must be >= 1")

    system = create_system(x, y, degree)
    coefficients = solve_system(system)
    return np.poly1d(coefficients[::-1])


def discrepancy(target: list, solution: list) -> float:
    """Finding the discrepancy of a function approximation

    :param target - function target values
    :param solution - approximation values

    It is assumed that target and solution values are
    set at the same interval

    :return - discrepancy of solution (float)
    """
    if len(target) != len(solution):
        raise ValueError("target and solution length must be the same length")

    result = 0
    for i in range(len(target)):
        result += (solution[i] - target[i]) ** 2

    return result


def read_data(file_name) -> tuple:
    """Read data in the next format:
    x1 x2 ... xn
    y1 y2 ... yn

    :param - the name of file with data
    """

    with open(file_name, "r") as file:
        read_x = file.readline().split()
        read_y = file.readline().split()
        x = [float(x) for x in read_x]
        y = [float(y) for y in read_y]
    return x, y


def main():
    """Approximating objective function by polynomial
    2nd and 3rd degrees

    Plotting

    Calculation the discrepancy of the approximations
    """
    x, y = read_data("approx_data.txt")

    polynomial2 = approx_poly(x, y, 2)
    polynomial3 = approx_poly(x, y, 3)

    plt.plot(x, y, "ro")
    plt.plot(x, polynomial2(x))
    plt.plot(x, polynomial3(x))
    plt.show()

    print("2nd degrees polynomial discrepancy:")
    print(discrepancy(y, polynomial2(x)))

    print("3rd degrees polynomial discrepancy:")
    print(discrepancy(y, polynomial3(x)))


if __name__ == '__main__':
    main()
