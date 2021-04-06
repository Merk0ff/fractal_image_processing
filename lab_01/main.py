import cv2
import numpy as np

from utils.loaders import ImageLoader, load_default, log


def fractal_dimension(loader: ImageLoader):
    loader.grayscale_img()
    img_bw = cv2.threshold(loader.img, 127, 255, cv2.THRESH_BINARY)[1] / 255

    min_dimension = min(img_bw.shape)
    greatest_power = 2 ** np.floor(np.log(min_dimension) / np.log(2))
    exponent = int(np.log(greatest_power) / np.log(2))

    scales = 2 ** np.arange(exponent, 0, -1)

    n = []
    for scale in scales:
        boxes = np.add.reduceat(
            np.add.reduceat(img_bw, np.arange(0, img_bw.shape[0], scale), axis=0),
            np.arange(0, img_bw.shape[1], scale),
            axis=1,
        )
        non_empty_boxes_number = len(
            np.where((boxes > 0) & (boxes < scale ** 2))[0]
        )

        n.append(non_empty_boxes_number)

    coeffs = np.polyfit(np.log(1 / scales), np.log(n), 1)

    return coeffs[0]


if __name__ == '__main__':
    data = []

    with load_default() as loaders:
        for load in loaders:
            data.append(
                '{:13}: {}'.format(load.name, fractal_dimension(load))
            )

    log('\n'.join(data))



