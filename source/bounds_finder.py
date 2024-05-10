import numpy as np
from color import Color


class BoundsFinder:
    """
    Class which holds functions to find to first and last black values \n
    to determine where to clip the image.
    """
    found_bounds = (620, 1579, 1445, 3053)

    def __init__(self, image_array: np.array):
        self.image_array = image_array
        self.shape = self.image_array.shape

    def first_black_row(self):
        for i in range(self.shape[0]):  # Iterate over rows
            for j in range(self.shape[1]):  # Iterate over columns
                if np.array_equal(self.image_array[i, j], Color.BLACK):
                    return i

    def first_black_col(self):
        for j in range(self.shape[1]):  # Iterate over columns
            for i in range(self.shape[0]):  # Iterate over rows
                if np.array_equal(self.image_array[i, j], Color.BLACK):
                    return j

    def last_black_row(self):
        for i in range(self.shape[0] - 1, 0, -1):  # Iterate over rows in reverse
            for j in range(self.shape[1]):  # Iterate over columns
                if np.array_equal(self.image_array[i, j], Color.BLACK):
                    return i

    def last_black_col(self):
        for j in range(self.shape[1] - 1, 0, -1):  # Iterate over columns in reverse
            for i in range(self.shape[0]):  # Iterate over rows
                if np.array_equal(self.image_array[i, j], Color.BLACK):
                    return j

    def get_bounds(self):
        return self.first_black_row(), self.first_black_col(), self.last_black_row(), self.last_black_col()
