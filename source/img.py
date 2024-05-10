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
    def _test15(array):     # _ means only used within this class
        if np.array_equal(array, Color.BLACK):
            return 0
        elif np.array_equal(array, Color.WHITE):
            return 1
        else:
            return 2

    @staticmethod
    def d3_to_d2(image_array):
        shape = image_array.shape
        reshaped_array = image_array.reshape(-1, 3)     # sort of flatten
        remapped = map(Img._test15, reshaped_array)
        remapped_list = list(remapped)
        remapped_array = np.array(remapped_list)
        return remapped_array.reshape(shape[:2])

    @staticmethod
    def d2_to_d3(array):
        """Trying to export a csv to a png"""
        old_shape = array.shape
        new_shape = (*old_shape, 3)
        ans = np.zeros(new_shape, dtype=np.uint8)     # Color.WHITE
        for row in range(new_shape[0]):
            for col in range(new_shape[1]):
                if 1 == array[row, col]:
                    ans[row, col] = [255, 255, 255]     # Color.BLACK
                elif 2 == array[row, col]:
                    ans[row, col] = [254, 1, 1]     # Color.RED
        return ans


