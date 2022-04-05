"""
AUTHOR: Ryan Grindle & Amanda Bertschinger

LAST MODIFIED: Dec 9, 2021

PURPOSE: Generate polynomials with only given exponents with
         only coefficients of one. All possible polynomials
         are equally likely to be generated.

NOTES:

TODO:
"""
import itertools
import os
import json


def get_eq_from_binary_list(b_list, exponents):
    """
    Given a list of exponents and a binary list of the same
    length create the corresponding equation (as a string)
    that is a polynomial in the variable x including only the
    exponents given.

    PARAMETERS
    ----------
    b_list : list containing only 0 or 1
        This list is the same length as exponents because
        the index of each 1 in b_list will be used to raise
        x to the power in the generated equation.
    exponents : list
        The usable exponents. Include 0 for constants.
        This list must be the same length as b_list.

    RETURNS
    -------
    eq_str : str
        The generated equation as a string.
    """
    assert len(b_list) == len(exponents)
    assert all((b == 0 or b == 1) for b in b_list)
    eq_list = []

    for i, b in enumerate(b_list):
        if b == 1:
            if exponents[i] == 0:
                eq_list.append('C')
            elif exponents[i] == 1:
                eq_list.append('C*x1')
            else:
                # eq_list.append('C*x1**{}'.format(exponents[i]))
                current_eq = 'C'
                for _ in range(exponents[i]):
                    current_eq += '*x1'
                eq_list.append(current_eq)

    eq_str = '+'.join(eq_list)
    return eq_str


def get_polynomials(exponents):

    # Get matrix where each row corresponds to a polynomial with
    # given exponents. This matrix contains all possible polynomials/rows.
    poly_matrix = list(itertools.product([0, 1], repeat=len(exponents)))

    # Remove first row because it is all zeros (the zero function).
    poly_matrix = poly_matrix[2:]

    # Convert rows to polynomials (strings).
    polynomials = []
    for row in poly_matrix:
        # each row corresponds to a polynomial
        poly = get_eq_from_binary_list(b_list=row, exponents=exponents)
        polynomials.append(poly)

    return polynomials


if __name__ == '__main__':
    import numpy as np

    # np.random.seed(1234)

    n_points = 30
    n_samples = 10000
    exponents = [0,4,6,8,10]
    polynomials = get_polynomials(exponents=exponents)
    print(polynomials)
    print(len(polynomials))

    # Now, get x and y data.
    eq_list = np.random.choice(polynomials, size=n_samples, replace=True)
    clean_eq_list = []

    print(len(eq_list))

    X = np.random.uniform(-3., 3., size=(n_samples, n_points))
    X.sort(axis=1)
    Y = np.zeros_like(X)
    for i, eq in enumerate(eq_list):
        cleanEQ = ''
        for char in eq:
            if char == 'C':
                char = '1.0'
            cleanEQ += char
        clean_eq_list.append(cleanEQ)

        f = eval('lambda x1: '+cleanEQ)
        Y[i] = f(X[i])
    print(eq_list[0])
    print(clean_eq_list[0])
    # print(X[0])
    # print(Y[0])

    # Next, format data into an actual dataset...
    folder = './Dataset'
    dataPath = folder + '/poly_nox2_x10_val.json'
    fileID = 0
    structure = {'X': [], 'Y': 0.0, 'EQ': '', 'Skeleton': ''}
    X = list(X)
    for i, x_values in enumerate(X):
        x_list = []
        for j in range(len(X[i])):
            value_list = []
            value = X[i][j]
            value_list.append(value)
            x_list.append(value_list)
        X[i] = x_list
    # print(x_list)
    for i, e in enumerate(eq_list):
        structure['X'] = list(X[i])
        structure['Y'] = list(Y[i])
        structure['Skeleton'] = str(e)
        structure['EQ'] = str(clean_eq_list[i])

        outputPath = dataPath
        if os.path.exists(outputPath):
            fileSize = os.path.getsize(outputPath)
            if fileSize > 500000000:  # 500 MB
                fileID += 1
        with open(outputPath, "a", encoding="utf-8") as h:
            json.dump(structure, h, ensure_ascii=False)
            h.write('\n')

    print(structure)
