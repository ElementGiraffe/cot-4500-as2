import numpy as np


# This is a modification of the code provided on WebCourses
def nevilles_method(x_points, y_points, x):
    # must specify the matrix size (this is based on how many columns/rows you want)
    matrix = np.zeros((len(x_points), len(y_points)))

    # fill in value (just the y values because we already have x set)
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]

    # the end of the first loop are how many columns you have...
    num_of_points = len(x_points)

    # populate final matrix (this is the iterative version of the recursion explained in class)
    # the end of the second loop is based on the first loop...
    for i in range(1, num_of_points):
        for j in range(1, i+1):
            first_multiplication = (x - x_points[i - j]) * matrix[i][j - 1]
            second_multiplication = (x - x_points[i]) * matrix[i - 1][j - 1]

            denominator = x_points[i] - x_points[i - j]

            # this is the value that we will find in the matrix
            coefficient = (first_multiplication-second_multiplication)/denominator
            matrix[i][j] = coefficient

    '''
    for i in range(len(x_points)):
        for j in range(i+1):
            print(str(matrix[i][j]) + " ", end='')
        print("")
    '''

    return matrix[num_of_points-1][num_of_points-1]


# This is a modification of the code provided on WebCourses
def divided_difference_table(x_points, y_points):
    # set up the matrix
    size: int = len(x_points)
    matrix: np.array = np.zeros((size, size))

    # fill the matrix
    for index, row in enumerate(matrix):
        row[0] = y_points[index]

    # populate the matrix (end points are based on matrix size and max operations we're using)
    for i in range(1, size):
        for j in range(1, i+1):
            # the numerator are the immediate left and diagonal left indices...
            numerator = matrix[i][j-1] - matrix[i-1][j-1]

            # the denominator is the X-SPAN...
            denominator = x_points[i] - x_points[i-j]

            operation = numerator / denominator

            # cut it off to view it more simpler
             # matrix[i][j] = '{0:.7g}'.format(operation)
            matrix[i][j] = operation

    #print(matrix)
    return matrix


# This is a modification of the code provided on WebCourses
def get_approximate_result(matrix, x_points, value):
    # p0 is always y0 and we use a reoccuring x to avoid having to recalculate x
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]

    # we only need the diagonals...and that starts at the first row...
    for index in range(1, len(x_points)):
        polynomial_coefficient = matrix[index][index]

        # we use the previous index for x_points....
        #print(str(value) + "-" + str(x_points[index-1]))
        reoccuring_x_span *= (value - x_points[index-1])

        # get a_of_x * the x_span
        mult_operation = polynomial_coefficient * reoccuring_x_span

        # add the reoccuring px result
        reoccuring_px_result += mult_operation

    # final result
    return reoccuring_px_result


# This is a modification of the code provided on WebCourses
def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i + 2):
            # skip if value is prefilled (we dont want to accidentally recalculate...)
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            # get left cell entry
            left: float = matrix[i][j-1]

            # get diagonal left entry
            diagonal_left: float = matrix[i-1][j-1]

            # order of numerator is SPECIFIC.
            numerator: float = left - diagonal_left

            # denominator is current i's x_val minus the starting i's x_val....
            denominator = matrix[i][0] - matrix[i - j + 1][0]

            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation

    return matrix


# This is a modification of the code provided on WebCourses
def hermite_interpolation(x_points, y_points, slopes):

    # matrix size changes because of "doubling" up info for hermite
    num_of_points = len(x_points)
    matrix = np.zeros((2*num_of_points, 2*num_of_points))

    # populate x values (make sure to fill every TWO rows)
    for i in range(0, 2 * num_of_points, 2):
        matrix[i][0] = x_points[i//2]
        matrix[i+1][0] = x_points[i//2]

    # prepopulate y values (make sure to fill every TWO rows)
    for i in range(0, 2 * num_of_points, 2):
        matrix[i][1] = y_points[i // 2]
        matrix[i+1][1] = y_points[i // 2]

    # prepopulate with derivates (make sure to fill every TWO rows. starting row CHANGES.)
    for i in range(1, 2 * num_of_points + 1, 2):
        matrix[i][2] = slopes[i // 2]
    #print(matrix)
    filled_matrix = apply_div_dif(matrix)
    #print(filled_matrix)
    return filled_matrix


# This is code implemented from the pseudocode in the textbook
def cubic_spline(x_points, y_points):
    x = x_points
    a = y_points

    n = len(x)

    h = np.zeros(n)
    for i in range(n-1):
        h[i] = x[i+1] - x[i]

    alpha = np.zeros(n)
    for i in range(1, n-1):
        alpha[i] = (3/h[i]) * (a[i+1]-a[i]) - (3/h[i-1]) * (a[i]-a[i-1])

    l = np.zeros(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    l[0] = 1
    mu[0] = 0
    z[0] = 0

    for i in range(1, n-1):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i]-h[i-1]*z[i-1])/l[i]

    l[n-1] = 1
    z[n-1] = 0

    c = np.zeros(n)
    c[n-1] = 0

    b = np.zeros(n)
    d = np.zeros(n)

    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (a[j+1]-a[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3;
        d[j] = (c[j+1]-c[j])/(3*h[j])

    #print(a, b, c, d)

    A_matrix = np.zeros((n, n))
    A_matrix[0][0] = 1
    A_matrix[n-1][n-1] = 1

    for i in range(1, n-1):
        A_matrix[i][i-1] = h[i-1]
        A_matrix[i][i] = 2*(h[i-1]+h[i])
        A_matrix[i][i+1] = h[i]

    print(A_matrix)
    print("")

    b_vector = np.zeros(n)
    b_vector[0] = 0
    b_vector[n-1] = 0
    for i in range(1, n-1):
        b_vector[i] = (3/h[i])*(a[i+1]-a[i]) - (3/h[i-1])*(a[i]-a[i-1])

    print(b_vector)
    print("")

    x_vector = c

    print(x_vector)


if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True, linewidth=100)

    # PART ONE
    # point setup
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    approximating_value = 3.7

    print(nevilles_method(x_points, y_points, approximating_value))
    #####

    print("")

    # PART TWO
    # point setup
    x_points = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    divided_table = divided_difference_table(x_points, y_points)

    answer = [divided_table[1][1], divided_table[2][2], divided_table[3][3]]
    print(answer)
    #####

    print("")

    # PART THREE
    # find approximation
    approximating_x = 7.3
    final_approximation = get_approximate_result(divided_table, x_points, approximating_x)

    print(final_approximation)
    #####

    print("")

    # PART FOUR
    # point setup
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]

    final_matrix = hermite_interpolation(x_points, y_points, slopes)

    print(final_matrix)
    #####

    print("")

    # PART FIVE
    x_points = [2, 5, 8, 10]
    y_points = [3, 5, 7, 9]
    cubic_spline(x_points, y_points)
    #####

    print("")


