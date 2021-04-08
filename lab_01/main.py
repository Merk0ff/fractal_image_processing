import cv2
import numpy as np

from utils.loaders import ImageLoader, load_default, log


def box_count(img, size):
    boxes = np.add.reduceat(
        np.add.reduceat(img, np.arange(0, img.shape[0], size), axis=0),
        np.arange(0, img.shape[1], size),
        axis=1
    )

    return len(np.where((boxes > 0) & (boxes < size * size))[0])


def fractal_dimension(loader: ImageLoader):
    loader.grayscale_img()
    img_bin = cv2.threshold(loader.img, 127, 255, cv2.THRESH_BINARY)[1] / 255

    min_dimension = min(img_bin.shape)
    greatest_power = 2 ** np.floor(np.log(min_dimension) / np.log(2))
    exponent = int(np.log(greatest_power) / np.log(2))

    sizes = 2 ** np.arange(exponent, 0, -1)

    cnt = [box_count(img_bin, size) for size in sizes]
    coeffs = np.polyfit(np.log(1 / sizes), np.log(cnt), 1)

    return coeffs[0]


if __name__ == '__main__':
    data = []

    with load_default() as loaders:
        for load in loaders:
            data.append(
                '{:13}: {}'.format(load.name, fractal_dimension(load))
            )

    log('\n'.join(data))



