
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
from source.color import Color


class Img:
    """
    Container class for image handling functions.
    """
    @staticmethod
    def load_from_png(filename):
        with PIL.Image.open(filename) as img:
            ans = np.asarray(img)
        return ans

    @staticmethod
    def load_from_csv(filename):
        return np.genfromtxt(filename, dtype=np.uint8, delimiter=',')

    @staticmethod
    def show(image_array):
        plt.imshow(image_array)
        plt.show()

    @staticmethod
    def save_as_png(image_array, file_name):
        ans = PIL.Image.fromarray(image_array)
        ans.save(file_name)

    @staticmethod
    def save_as_csv(image_array, file_name):
        np.savetxt(file_name, image_array, delimiter=',', fmt='%d')

    @staticmethod
    def _test15(array, first_pixel):     # _ means only used within this class
        if np.array_equal(array, Color.BLACK):
            return 0
        elif np.array_equal(array, Color.WHITE):
            return 1
        elif np.array_equal(array, first_pixel):
            return 3
        else:
            return 2

    @staticmethod
    def d3_to_d2_old(image_array: np.array):
        # shape = image_array.shape
        # reshaped_array = image_array.reshape(-1, 3)     # sort of flatten
        # remapped = map(Img._test15, reshaped_array)
        # remapped_list = list(remapped)
        # remapped_array = np.array(remapped_list)
        # return remapped_array.reshape(shape[:2])
        first_pixel = image_array[0, 0]
        shape = image_array.shape
        reshaped_array = image_array.reshape(-1, 3)  # Flatten the image_array
        remapped_array = Img._test15(reshaped_array, first_pixel)  # Apply _test15 function directly
        return remapped_array.reshape(shape[:2])  # Reshape back to 2D array

    @staticmethod
    def d3_to_d2(array, first_pixel=None):
        """Perform element-wise operation on input array."""
        ans = np.zeros(array.shape[:2], dtype=int)  # Create an array to store the result
        # Apply the logic element-wise using vectorized operations
        # ans[np.array_equal(array, Color.BLACK)] = 0
        ans[np.all(array == Color.WHITE, -1)] = 1
        if first_pixel is not None:
            ans[np.all(array == first_pixel, axis=-1)] = 3
            ans[~(np.all(array == Color.BLACK, -1) | np.all(array == Color.WHITE, -1) |
                  np.all(array == first_pixel, -1))] = 2
        else:
            ans[~(np.all(array == Color.BLACK, -1) | np.all(array == Color.WHITE, -1))] = 2
        return ans

    @staticmethod
    def d2_to_d3(array):
        """Trying to export a csv to a png"""
        old_shape = array.shape
        new_shape = (*old_shape, 3)
        ans = np.zeros(new_shape, dtype=np.uint8)   # Color.WHITE
        ans[np.where(array == 1)] = [255, 255, 255]     # Color.BLACK
        ans[np.where(array == 2)] = [254, 1, 1]     # Color.RED
        # for row in range(new_shape[0]):
        #     for col in range(new_shape[1]):
        #         if 1 == array[row, col]:
        #             ans[row, col] = [255, 255, 255]
        #         elif 2 == array[row, col]:
        #             ans[row, col] = [254, 1, 1]
        return ans

    @staticmethod
    def clip(image_array, bounds):
        ub, lb, db, rb = bounds
        return image_array[ub:db, lb:rb]

    @staticmethod
    def replace_first_pixel(image_array):
        """Replace all red (first pixel) with black."""
        image_array = image_array.copy()
        first_pixel = image_array[0, 0]  # [237  28  36]
        mask = np.all(np.array_equal(image_array, first_pixel), axis=-1)
        image_array[mask] = Color.BLACK
        return image_array
