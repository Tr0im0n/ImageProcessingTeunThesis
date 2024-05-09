import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import scipy.stats


bounds = (620, 1579, 1445, 3053)

BLACK = np.array((0, 0, 0))
WHITE = np.array((255, 255, 255))
# RED = image_array[0, 0]  # [237  28  36]
RED = np.array((255, 0, 0))
YELLOW = np.array((255, 255, 0))
DARKGRAY = np.array((17, 17, 17))
LIGHTGRAY = np.array((238, 238, 238))


class Img:
    """
    Container class for image handling functions.
    """
    @staticmethod
    def load(filename):
        with PIL.Image.open(filename) as img:
            ans = np.asarray(img)
        return ans[:, :, :]

    @staticmethod
    def show(image_array):
        plt.imshow(image_array)
        plt.show()

    @staticmethod
    def save(image_array, file_name):
        img = PIL.Image.fromarray(image_array)
        img.save(file_name)


def get_color_set(image_array: np.array):
    """Get a set will all the different colors in the image"""
    reshaped_array = image_array.reshape(-1, 3)
    return set(map(tuple, reshaped_array))


class BoundsFinder:
    """
    Class which holds functions to find to first and last black values \n
    to determine where to clip the image.
    """
    def __init__(self, image_array: np.array):
        self.image_array = image_array
        self.shape = self.image_array.shape

    def first_black_row(self):
        for i in range(self.shape[0]):  # Iterate over rows
            for j in range(self.shape[1]):  # Iterate over columns
                if np.array_equal(self.image_array[i, j], BLACK):
                    return i

    def first_black_col(self):
        for j in range(self.shape[1]):  # Iterate over columns
            for i in range(self.shape[0]):  # Iterate over rows
                if np.array_equal(self.image_array[i, j], BLACK):
                    return j

    def last_black_row(self):
        for i in range(self.shape[0] - 1, 0, -1):  # Iterate over rows in reverse
            for j in range(self.shape[1]):  # Iterate over columns
                if np.array_equal(self.image_array[i, j], BLACK):
                    return i

    def last_black_col(self):
        for j in range(self.shape[1] - 1, 0, -1):  # Iterate over columns in reverse
            for i in range(self.shape[0]):  # Iterate over rows
                if np.array_equal(self.image_array[i, j], BLACK):
                    return j

    def get_bounds(self):
        return self.first_black_row(), self.first_black_col(), self.last_black_row(), self.last_black_col()


def clip(image_array):
    return image_array[620:1445, 1579:3053, :3]


def replace1(image_array):
    """Replace all red (first pixel) with black."""
    image_array = image_array.copy()
    RED_here = image_array[0, 0]  # [237  28  36]
    mask = np.all(image_array == RED_here, axis=-1)
    image_array[mask] = BLACK
    return image_array


def replace2(image_array):
    """Replace all yellow (not black or white) with bright yellow."""
    image_array = image_array.copy()
    mask = np.isin(image_array, [DARKGRAY, LIGHTGRAY], invert=True).any(axis=-1)
    # mask = np.all((image_array != BLACK) & (image_array != WHITE), axis=-1)
    image_array[mask] = RED
    return image_array


def replace3(image_array):
    """Replace all darkgray with bright black."""
    image_array = image_array.copy()
    mask = np.all(image_array == BLACK, axis=-1)
    image_array[mask] = DARKGRAY
    mask = np.all(image_array == WHITE, axis=-1)
    image_array[mask] = LIGHTGRAY
    return image_array


def majority_color(image_array, n):
    height, width, _ = image_array.shape
    result_array = np.empty((height // n + 1, width // n + 1, 3), dtype=np.uint8)  # New array to store majority colors

    for i in range(0, height, n):
        for j in range(0, width, n):
            block = image_array[i:i + n, j:j + n]  # Extract block of pixels
            majority = scipy.stats.mode(block.reshape(-1, 3), axis=0)[0][0]  # Calculate majority color
            result_array[i // n, j // n] = majority  # Store majority color in result array

    return result_array


def test15(color):
    """Used as map function in following function"""
    if np.array_equal(color, DARKGRAY):
        return 0
    elif np.array_equal(color, LIGHTGRAY):
        return 1
    else:
        return 2


def to_csv(image_array):
    """Turn image into a CSV"""
    shape = image_array.shape
    reshaped_array = image_array.reshape(-1, 3)
    remapped = map(test15, reshaped_array)
    remapped_list = list(remapped)
    remapped_array = np.array(remapped_list)
    reshaped_array = remapped_array.reshape(shape[:2])
    np.savetxt('data.csv', reshaped_array, delimiter=',', fmt='%d')


def csv_to_show():
    """Import csv to numpy array, then imshow"""
    data = np.genfromtxt('data.csv', delimiter=',')
    plt.imshow(data)
    plt.show()


def majority_in_csv(n: int = 8, fraction: float = 0.7):
    data = np.genfromtxt('data.csv', delimiter=',')
    height, width = data.shape
    result_array = np.empty((height // n + 1, width // n + 1), dtype=np.uint8)

    for i in range(0, height, n):
        for j in range(0, width, n):
            block = data[i:i + n, j:j + n]  # Extract block of pixels
            majority = int(sum(block.flatten())/pow(n, 2) + fraction)      # Calculate majority color
            result_array[i // n, j // n] = majority  # Store majority color in result array

    plt.imshow(result_array)
    plt.show()
    np.savetxt('result.csv', result_array, delimiter=',', fmt='%d')
    return result_array


def csv_to_png():
    """Trying to export a csv to a png"""
    data = np.genfromtxt('data.csv', delimiter=',')
    new_shape = (*data.shape, 3)
    ans = np.zeros(new_shape, dtype=np.uint8)
    for row in range(new_shape[0]):
        for col in range(new_shape[1]):
            if 1 == data[row, col]:
                ans[row, col] = [255, 255, 255]
            elif 2 == data[row, col]:
                ans[row, col] = [254, 1, 1]
    Img.save(ans, "data1.png")


def main():
    # file_name = "test5.PNG"
    # image_array = Img.load(file_name)
    # to_csv(image_array)
    # mc = majority_color(image_array, 4)
    # print(image_array[400, 400])
    # clipped_array = replace2(image_array)
    # Img.save(mc, "test6.PNG")
    # print(image_array.dtype)
    # print(*get_color_set(image_array), sep="\n")
    # my_bf = BoundsFinder(image_array)
    # print(my_bf.get_bounds())
    # show_image(image_array)
    csv_to_png()


if __name__ == '__main__':
    main()

