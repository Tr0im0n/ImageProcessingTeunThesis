import os

from source.bounds_finder import BoundsFinder
from source.img import Img


def main():
    os.chdir(r"../old_data")
    import_file_name = "test.PNG"
    image_array = Img.load_from_png(import_file_name)
    rgb_array = image_array[:, :, :3]
    first_pixel = rgb_array[0, 0]
    d2_array = Img.d3_to_d2(rgb_array, first_pixel)

    boundsfinder = BoundsFinder(d2_array, 0)
    bounds = boundsfinder.get_bounds()

    clipped_array = Img.clip(d2_array, bounds)
    d3_array = Img.d2_to_d3(clipped_array)
    Img.show(d3_array)
    export_file_name = "test7.png"
    Img.save_as_png(d3_array, export_file_name)

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
    # csv_to_png()


if __name__ == '__main__':
    main()

