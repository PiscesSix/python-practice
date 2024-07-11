from __future__ import annotations

import random
from typing import List


class Matrix:
    """
    A class used to represent a matrix

    ...
    Attributes
    ----------
        rows : int
            the number of rows of the matrix
        cols : int
            the number of columns of the matrix

        matrix : list
            a 2D list that represents the matrix
    
    Methods
    -------
        get_matrix_value(row, col)
            Get the value of the index (row, col) of the matrix
        
        get_matrix_row()
            Get the number of rows of the matrix
        
        get_matrix_col()
            Get the number of columns of the matrix

        get_matrix_values()
            Get all values of the matrix

        set_matrix_value(row, col, value)
            Set the value of the index (row, col) of the matrix
        
        random_matrix()
            Randomly generate the matrix from 1 to 10
        
        print_matrix()
            Print the matrix to the screen

        add_matrix(matrix)
            Add two matrix together
        
        sub_matrix(matrix)
            Subtract two matrix together
        
        multi_matrix(matrix)
            Multiply two matrix together
        
        T()
            transpose the matrix
        
        __add__(matrix)
            overload operator + to add two matrix
        
        __sub__(matrix)
            overload operator - to subtract two matrix
        
        __mul__(matrix)
            overload operator * to multiply two matrix
    """
    rows: int
    cols: int
    matrix: List[List[int]]

    def __init__(self, rows: int = None, cols: int = None):
        """
        Parameters
        ----------
            rows : int
                the number of rows of the matrix (default is None)
            cols : int
                the number of columns of the matrix (default is None)
        """

        self.rows = rows
        self.cols = cols
        self.matrix = [[0] * cols for _ in range(rows)]

    def get_matrix_value(self, row: int, col: int) -> int:
        """
        Get the value of the index (row, col) of the matrix

        Parameters
        ----------
            row : int
                the row index of the matrix
            col : int
                the column index of the matrix

        Returns
        -------
            int: the value of the index (row, col) of the matrix
        """
        return self.matrix[row][col]

    def get_matrix_row(self) -> int:
        """
        Get the number of rows of the matrix

        Returns
        -------
            int: the number of rows of the matrix
        """
        return self.rows

    def get_matrix_col(self) -> int:
        """
        Get the number of columns of the matrix

        Returns
        -------
            int: the number of columns of the matrix
        """
        return self.cols

    def get_matrix_values(self) -> List[List[int]]:
        """
        Get all values of the matrix

        Returns
        -------
            list: a 2D list that represents the matrix
        """
        return self.matrix

    def set_matrix_value(self, row: int, col: int, value: int):
        """
        Set the value of the index (row, col) of the matrix

        Parameters
        ----------
            row : int
                the row index of the matrix
            col : int
                the column index of the matrix
            value : int
                the value that you want to set
        """
        self.matrix[row][col] = value

    def random_matrix(self):
        """
        Randomly generate the matrix from 1 to 10

        """
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = random.randint(1, 11)

    def print_matrix(self):
        """
        Print the matrix to the screen

        """
        for i in range(self.rows):
            print(self.matrix[i])

    def add_matrix(self, matrix: Matrix) -> Matrix:
        """
        Add two matrix together

        Parameters
        ----------
            matrix : Matrix
                the matrix that you want to add
        
        Returns
        -------
            Matrix: the result of adding two matrix together
        
        Raises
        ------
            ValueError: if the two matrix have different size
        """
        if self.rows != matrix.get_matrix_row() or self.cols != matrix.get_matrix_col():
            raise ValueError("Two matrix must have the same")

        matrix_sum = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                matrix_sum.set_matrix_value(i, j, matrix.get_matrix_value(i, j) + self.matrix[i][j])

        return matrix_sum

    def sub_matrix(self, matrix: Matrix) -> Matrix:
        """
        Subtract two matrix together

        Parameters
        ----------
            matrix : Matrix
                the matrix that you want to subtract
            
        Returns
        -------
            Matrix: the result of subtracting two matrix together

        Raises
        ------
            ValueError: if the two matrix have different size
        """

        if self.rows != matrix.get_matrix_row() or self.cols != matrix.get_matrix_col():
            raise ValueError("Two matrix must have the same")

        matrix_sub = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                matrix_sub.set_matrix_value(i, j, matrix.get_matrix_value(i, j) - self.matrix[i][j])

        return matrix_sub

    def multi_matrix(self, matrix: Matrix) -> Matrix:
        """
        Multiply two matrix together

        Parameters
        ----------
            matrix : Matrix
                the matrix that you want to multiply
        
        Returns
        -------
            Matrix: the result of multiplying two matrix together
        
        Raises
        ------
            ValueError: if the two matrix have different size
        """
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
        """
        Transpose the matrix
        """

        for i in range(self.rows):
            for j in range(i, self.cols):
                tmp = self.matrix[i][j]
                self.matrix[i][j] = self.matrix[j][i]
                self.matrix[j][i] = tmp

    def __add__(self, matrix: Matrix) -> Matrix:
        """
        Overload operator + to add two matrix

        Parameters
        ----------
            matrix : Matrix
                the matrix that you want to add
        """
        return self.add_matrix(matrix)

    def __sub__(self, matrix: Matrix) -> Matrix:
        """
        Overload operator - to subtract two matrix

        Parameters
        ----------
            matrix : Matrix
                the matrix that you want to subtract
        """
        return self.sub_matrix(matrix)

    def __mul__(self, matrix: Matrix) -> Matrix:
        """
        Overload operator * to multiply two matrix

        Parameters
        ----------
            matrix : Matrix
                the matrix that you want to multiply
        """
        return self.multi_matrix(matrix)


class SquareMatrix(Matrix):
    """
    A class used to represent a square matrix

    ...
    Attributes
    ----------
        rows : int
            the number of rows of the matrix
        cols : int
            the number of columns of the matrix

    Methods
    -------
        split_matrix(matrix, index_col)
            Split the matrix by the index column
        
        det(matrix, size)
            Calculate the determinant of the matrix
    """

    def __init__(self, rows: int = None, cols: int = None):
        """
        Parameters
        ----------
            rows : int
                the number of rows of the matrix (default is None)
            cols : int
                the number of columns of the matrix (default is None)
        
        Raises
        ------
            ValueError: if the columns and rows of the matrix are not the same
        """
        super().__init__(rows, cols)
        if self.rows != self.cols:
            raise ValueError("The columns and rows of the matrix must be the same")

    @staticmethod
    def split_matrix(matrix: Matrix, index_col: int) -> List[List[int]]:
        """
        Split the matrix by the index column
        
        Parameters
        ----------
            matrix : list
                a 2D list that represents the matrix
            index_col : int
                the column index that you want to split
        """
        return [row[:index_col] + row[index_col + 1:] for row in matrix[1:]]

    @staticmethod
    def det(matrix: List[List[int]], size: int) -> int:
        """
        Calculate the determinant of the matrix

        Parameters
        ----------
            matrix : list
                a 2D list that represents the matrix
            size : int
                the size of the matrix
        
        Returns
        -------
            int: the determinant of the matrix
        """
        det_result = 0
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        else:
            for i in range(len(matrix)):
                det_result += (-1) ** i * matrix[0][i] * SquareMatrix.det(SquareMatrix.split_matrix(matrix, i),
                                                                          size - 1)

            return det_result


class TriangleMatrix(SquareMatrix):
    """
    A class used to represent a triangle matrix

    ...

    Attributes
    ----------
        rows : int
            the number of rows of the matrix
        cols : int
            the number of columns of the matrix

    Methods
    -------
        det(matrix, size)
            Calculate the determinant of the matrix
    """

    def __init__(self, rows: int = None, cols: int = None):
        """
        Parameters
        ----------
            rows : int
                the number of rows of the matrix (default is None)
            cols : int
                the number of columns of the matrix (default is None)
        """
        super().__init__(rows, cols)

    def det(self, size: int) -> int:
        """
        Calculate the determinant of the matrix

        Parameters
        ----------
            size : int
                the size of the matrix
        
        Returns
        -------
            int: the determinant of the matrix
        """
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
