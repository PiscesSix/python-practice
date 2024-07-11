import random


class Matrix:
    def __init__(self, rows: int = None, cols: int = None):
        self.rows = rows
        self.cols = cols
        self.matrix = [[0] * cols for _ in range(rows)]

    # accessors
    def get_matrix_value(self, row, col):
        return self.matrix[row][col]

    def get_matrix_row(self):
        return self.rows

    def get_matrix_col(self):
        return self.cols

    def get_matrix_values(self):
        return self.matrix

    def set_matrix_value(self, row, col, value):
        self.matrix[row][col] = value

    # methods
    def random_matrix(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = random.randint(1, 11)

    def print_matrix(self):
        for i in range(self.rows):
            print(self.matrix[i])

    def add_matrix(self, matrix):
        if self.rows != matrix.get_matrix_row() or self.cols != matrix.get_matrix_col():
            raise ValueError("Two matrix must have the same")

        matrix_sum = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                matrix_sum.set_matrix_value(i, j, matrix.get_matrix_value(i, j) + self.matrix[i][j])

        return matrix_sum

    def sub_matrix(self, matrix):
        if self.rows != matrix.get_matrix_row() or self.cols != matrix.get_matrix_col():
            raise ValueError("Two matrix must have the same")

        matrix_sub = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                matrix_sub.set_matrix_value(i, j, matrix.get_matrix_value(i, j) - self.matrix[i][j])

        return matrix_sub

    def multi_matrix(self, matrix):
        if self.cols != matrix.get_matrix_row():
            raise ValueError("Matrix size must be the same")

        mul_matrix_result = type(matrix)(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(matrix.get_matrix_row()):
                for k in range(self.cols):
                    mul_matrix_result.set_matrix_value(i, j, mul_matrix_result.get_matrix_value(i, j) + self.matrix[i][
                        k] * matrix.get_matrix_value(k, j))

        return mul_matrix_result

    def T(self):
        for i in range(self.rows):
            for j in range(i, self.cols):
                tmp = self.matrix[i][j]
                self.matrix[i][j] = self.matrix[j][i]
                self.matrix[j][i] = tmp

    # operators overload
    def __add__(self, matrix):
        return self.add_matrix(matrix)

    def __sub__(self, matrix):
        return self.sub_matrix(matrix)

    def __mul__(self, matrix):
        return self.multi_matrix(matrix)


class SquareMatrix(Matrix):
    def __init__(self, rows: int = None, cols: int = None):
        super().__init__(rows, cols)
        if self.rows != self.cols:
            raise ValueError("The columns and rows of the matrix must be the same")

    @staticmethod
    def split_matrix(matrix, index_col):
        # return [[row[i] for i in range(len(row)) if i != index_col] for row in matrix[1:]]
        return [row[:index_col] + row[index_col + 1:] for row in matrix[1:]]

    @staticmethod
    def det(matrix, size):
        det_result = 0
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        else:
            for i in range(len(matrix)):
                det_result += (-1) ** i * matrix[0][i] * SquareMatrix.det(SquareMatrix.split_matrix(matrix, i), size - 1)

            return det_result


class TriangleMatrix(SquareMatrix):
    def __init__(self, rows: int = None, cols: int = None):
        super().__init__()

    def det(self, size):
        det = 1
        size = self.rows
        for i in range(size):
            det *= self.matrix[i][i]

        return det


def main():
    matrixA = Matrix(3, 3)
    matrixA.random_matrix()
    matrixA.print_matrix()
    print()
    matrixB = Matrix(3, 3)
    matrixB.random_matrix()
    matrixB.print_matrix()
    print()
    matrixC = matrixA + matrixB
    matrixC.print_matrix()
    print()
    matrixD = matrixA - matrixB
    matrixD.print_matrix()
    print()
    matrixE = matrixA * matrixB
    matrixE.print_matrix()
    print()
    matrixE.T()
    matrixE.print_matrix()
    print()

    MatrixSquareA = SquareMatrix(4, 4)
    MatrixSquareA.random_matrix()
    MatrixSquareA.print_matrix()
    print()
    MatrixSquareB = SquareMatrix(4, 4)
    MatrixSquareB.random_matrix()
    MatrixSquareB.print_matrix()
    print()
    MatrixSquareC = MatrixSquareA * MatrixSquareB
    MatrixSquareC.print_matrix()
    print()
    print(MatrixSquareC.det(MatrixSquareC.get_matrix_values(), MatrixSquareC.get_matrix_col()))


if __name__ == "__main__":
    main()
