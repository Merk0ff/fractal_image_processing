import cv2
import numpy as np

from utils.loaders import ImageLoader, load_default


def get_ub(img, call_func, max_d=2):
    w, h = img.shape
    u = np.zeros((max_d + 1, w + 1, h + 1))

    for i in range(w):
        for j in range(h):
            u[0][i][j] = img[i][j]
            for d in range(1, max_d + 1):
                u[d][i][j] = call_func(
                    u[d-1][i][j] + 1,
                    call_func(
                        u[d-1][i+1][j+1],
                        u[d-1][i-1][j+1],
                        u[d-1][i+1][j-1],
                        u[d-1][i-1][j-1]
                    )
                )
    return u


def get_u(img, max_d=2):
    return get_ub(img, max, max_d)


def get_b(img, max_d=2):
    return get_ub(img, min, max_d)


def get_vol(img, max_d=2):
    w, h = img.shape
    u = get_u(img, max_d)
    b = get_b(img, max_d)
    vol = np.zeros(max_d + 1)

    for d in range(1, max_d + 1):
        summ = 0

        for i in range(w):
            for j in range(h):
                summ += u[d][i][j] - b[d][i][j]

        vol[d] = summ

    return vol


def get_a(img, max_d=2):
    vol = get_vol(img, max_d)
    a = np.zeros(max_d + 1)

    for d in range(1, max_d + 1):
        a[d] = (vol[d] - vol[d-1]) / 2

    return a[max_d - 1]


def segment_img(loader: ImageLoader, cell_size=50, max_delta=2):
    img = loader.img
    width, height = img.shape
    cell_width = int(width / cell_size)
    cell_height = int(height / cell_size)
    big_a = np.zeros((cell_width, cell_height))

    for i in range(cell_width):
        for j in range(cell_height):
            cell_img = img[i * cell_size:cell_size * (i + 1), j * cell_size:cell_size * (j + 1)]
            big_a[i][j] = get_a(cell_img, max_delta)

    threshold = np.mean(big_a)
    for i in range(cell_width):
        for j in range(cell_height):
            if big_a[i][j] > threshold:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)

            cv2.rectangle(
                img,
                (j * cell_size, i * cell_size),
                (cell_size * (j + 1), cell_size * (i + 1)),
                color,
                -1
            )

    loader.set_img(img)


if __name__ == '__main__':
    data = []

    with load_default() as loaders:
        for load in loaders:
            load.grayscale_img()
            segment_img(load, 5, 2)


