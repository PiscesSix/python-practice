import random
from typing import List, Self


class Matrix:
    """Represents a matrix with rows and columns.

    It provides methods to set and get values of the matrix,
    its rows, and columns.
    It also includes a method to populate the matrix with random values and
    a method to print the matrix.

    Attributes:
        rows (int): The number of rows in the matrix.
        cols (int): The number of columns in the matrix.
        matrix (list): A 2D list to store the values of the matrix
    """

    def __init__(self, rows: int | None, cols: int | None) -> None:
        """Initialize the matrix with the specified number of rows and columns.

        Args:
            rows (int): The number of rows in the matrix.
            cols (int): The number of columns in the matrix.
        """
        self.rows = rows if rows is not None else 0
        self.cols = cols if cols is not None else 0
        self.matrix = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def get_matrix_value(self, row: int, col: int) -> int:
        """Get the value at a specific position in the matrix.

        Args:
            row (int): The row index of the value.
            col (int): The column index of the value.
        """
        return self.matrix[row][col]

    def get_matrix_row(self) -> int:
        """Get the number of rows in the matrix.

        Returns:
            int: The number of rows in the matrix.
        """
        return self.rows

    def get_matrix_col(self) -> int:
        """Get the number of columns in the matrix.

        Returns:
            int: The number of columns in the matrix.
        """
        return self.cols

    def get_matrix_values(self) -> List[List[int]]:
        """Get the values of the matrix.

        Returns:
            list: A 2D list containing the values of the matrix
        """
        return self.matrix

    def set_matrix_value(self, row: int, col: int, value: int):
        """Set the value at a specific position in the matrix.

        Args:
            row (int): The row index of the value.
            col (int): The column index of the value.
            value (int): The value to be set at the specified
        """
        self.matrix[row][col] = value

    def random_matrix(self):
        """Populate the matrix with random values between 1 and 10.
        Returns:
            Matrix: A matrix object with random values.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = random.randint(1, 11)

    def print_matrix(self):
        """Prints the matrix row by row."""
        for i in range(self.rows):
            print(self.matrix[i])

    def transpose(self):
        """Transpose the matrix in place."""
        for i in range(self.rows):
            for j in range(i, self.cols):
                tmp = self.matrix[i][j]
                self.matrix[i][j] = self.matrix[j][i]
                self.matrix[j][i] = tmp

    def __add__(self, matrix: Self) -> Self:
        """Add two matrices together.
        Args:
            matrix (Matrix): The matrix to be added to the current matrix.
        Returns:
            Matrix: A new matrix object containing the sum of the two matrices.
        Raises:
            ValueError: If the two matrices do not have the same dimensions.
        """
        if (self.rows != matrix.get_matrix_row()
                or self.cols != matrix.get_matrix_col()):
            raise ValueError('Two matrix must have the same')

        matrix_sum = type(self)(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                matrix_sum.set_matrix_value(i, j, matrix.get_matrix_value(i, j)
                                            + self.matrix[i][j])

        return matrix_sum

    def __sub__(self, matrix: Self) -> Self:
        """Subtract one matrix from another.

        Args:
            matrix (Matrix): The matrix to be subtracted from the
            current matrix.
        Returns:
            Matrix: A new matrix object containing the difference of the
            two matrices.
        Raises:
            ValueError: If the two matrices do not have the same dimensions.
        """
        if (self.rows != matrix.get_matrix_row()
                or self.cols != matrix.get_matrix_col()):
            raise ValueError('Two matrix must have the same')

        matrix_sub = type(self)(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                matrix_sub.set_matrix_value(i, j, matrix.get_matrix_value(i, j)
                                            - self.matrix[i][j])

        return matrix_sub

    def __mul__(self, matrix: Self) -> Self:
        """Multiply two matrices together.

        Args:
            matrix (Matrix): The matrix to be multiplied by the
            current matrix.
        Returns:
            Matrix: A new matrix object containing the product of the
            two matrices.
        Raises:
            ValueError: If the two matrices do not have the
            same dimensions.
        """
        if self.cols != matrix.get_matrix_row():
            raise ValueError('Matrix size must be the same')

        mul_matrix_result = type(matrix)(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(matrix.get_matrix_row()):
                for k in range(self.cols):
                    mul_matrix_result.set_matrix_value(
                        i, j,
                        mul_matrix_result.get_matrix_value(i, j)
                        + self.matrix[i][k]
                        * matrix.get_matrix_value(k, j)
                    )

        return mul_matrix_result

    def __eq__(self, other: object) -> bool:
        """ Check if two matrices are equal.

        Args:
            other (object): The object to compare with.
        Returns:
            bool: True if the matrices are equal, False otherwise.
        """
        if not isinstance(other, Matrix):
            return False

        if (self.rows != other.get_matrix_row() or
                self.cols != other.get_matrix_col()):
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                if self.matrix[i][j] != other.get_matrix_value(i, j):
                    return False

        return True

    def __neg__(self):
        """Negate the matrix in place.

        Returns:
            Matrix: A new matrix object with all values negated.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = -self.matrix[i][j]

        return self

    def __iadd__(self, matrix: Self) -> Self:
        """Add another matrix to the current matrix with operator +=.

        Args:
            matrix (Matrix): The matrix to be added to the current matrix.
        Returns:
            Matrix: The current matrix object after the addition.
        Raises:
            ValueError: If the two matrices do not have the same dimensions.
        """
        if (self.rows != matrix.get_matrix_row()
                or self.cols != matrix.get_matrix_col()):
            raise ValueError('Two matrix must have the same')

        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] += matrix.get_matrix_value(i, j)

        return self

    def __isub__(self, matrix: Self) -> Self:
        """Subtract another matrix from the current matrix with operator -=.

        Args:
            matrix (Matrix): The matrix to be subtracted from
            the current matrix.
        Returns:
            Matrix: The current matrix object after the subtraction.
        Raises:
            ValueError: If the two matrices do not have the same dimensions.
        """
        if (self.rows != matrix.get_matrix_row()
                or self.cols != matrix.get_matrix_col()):
            raise ValueError('Two matrix must have the same')

        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] -= matrix.get_matrix_value(i, j)

        return self

    def __imul__(self, matrix: Self) -> Self:
        """Multiply the current matrix by another matrix with operator *=.

        Args:
            matrix (Matrix): The matrix to be multiplied by the current matrix.
        Returns:
            Matrix: The current matrix object after the multiplication.
        Raises:
            ValueError: If the two matrices do not have the same dimensions.
        """
        if self.cols != matrix.get_matrix_row():
            raise ValueError('Matrix size must be the same')

        for i in range(self.get_matrix_row()):
            for j in range(matrix.get_matrix_row()):
                for k in range(self.cols):
                    self.set_matrix_value(
                        i, j,
                        self.get_matrix_value(i, j)
                        + self.matrix[i][k]
                        * matrix.get_matrix_value(k, j)
                    )

        return self

    def __repr__(self) -> str:
        """Return a string representation of the matrix.

        Returns:
            str: A string representation of the matrix.
        """
        result = ''
        for row in range(self.get_matrix_row()):
            result += ' '.join(['{} '.format(col)
                                for col in range(self.get_matrix_col())])
            result += '\n'
        return result

    def __rmul__(self, other: int) -> Self:
        """Multiply the matrix by a scalar from the right.

        Args:
            other (int): The scalar value to multiply the matrix by.
        Returns:
            Matrix: A new matrix object containing the result
            of the multiplication.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] *= other
        return self


class SquareMatrix(Matrix):
    def __init__(self, rows: int | None = None, cols: int | None = None):
        super().__init__(rows, cols)
        if self.rows != self.cols:
            raise ValueError('The columns and rows of the matrix '
                             'must be the same')

    @staticmethod
    def split_matrix(
            matrix: List[List[int]],
            index_col: int
    ) -> List[List[int]]:

        return [row[:index_col] + row[index_col + 1:] for row in matrix[1:]]

    @staticmethod
    def det(matrix: List[List[int]], size: int) -> int:
        """Calculate the determinant of a square matrix.

        Args:
            matrix (list): A 2D list containing the values of the matrix.
            size (int): The size of the square matrix.
        Returns:
            int: The determinant of the square matrix.
        """
        det_result = 0
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        else:
            for i in range(len(matrix)):
                det_result += ((-1) ** i * matrix[0][i] * SquareMatrix.det(
                    SquareMatrix.split_matrix(matrix, i),
                    size - 1))

            return det_result


class TriangleMatrix(SquareMatrix):
    def __init__(self, rows: int | None = None, cols: int | None = None):
        super().__init__(rows, cols)

    @staticmethod
    def det(matrix: List[List[int]], size: int) -> int:
        det = 1
        for i in range(len(matrix)):
            det *= matrix[i][i]

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
    matrixE.transpose()
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
    print(
        MatrixSquareC.det(MatrixSquareC.get_matrix_values(),
                          MatrixSquareC.get_matrix_col())
    )
    print()
    print('Negative Matrix')
    (-MatrixSquareC).print_matrix()
    print()
    print('Add Matrix A with Matrix B +=')
    MatrixSquareA += MatrixSquareB
    MatrixSquareA.print_matrix()
    print()
    print('Sub Matrix A with Matrix B -=')
    MatrixSquareA -= MatrixSquareB
    MatrixSquareA.print_matrix()
    print()
    print('Mul Matrix A with Matrix B *=')
    MatrixSquareA *= MatrixSquareB
    MatrixSquareA.print_matrix()


if __name__ == "__main__":
    main()
