from source.bounds_finder import BoundsFinder
from source.img import Img


def main():
    import_file_name = "test5.PNG"
    image_array = Img.load_from_png(import_file_name)
    d2_array = Img.d3_to_d2(image_array)

    boundsfinder = BoundsFinder(image_array, 3)
    bounds = boundsfinder.get_bounds()

    clipped_array = Img.clip(d2_array, bounds)


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

