import numpy as np

from utils.loaders import ImageLoader, load_default, log


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

    return a


def get_d(loader: ImageLoader, cell_size=50, max_d=2):
    img = loader.img
    w, h = img.shape
    cell_width = int(w / cell_size)
    cell_height = int(h / cell_size)
    big_a = np.zeros((w, h, max_d + 1))

    for i in range(cell_width):
        for j in range(cell_height):
            cell_img = img[j * cell_size:cell_size * (j + 1), i * cell_size:cell_size * (i + 1)]

            a = get_a(cell_img, max_d)
            for d in range(1, max_d + 1):
                big_a[i][j][d] = a[d]

    sum_a = []
    for d in range(1, max_d + 1):
        summ = 0
        for i in range(0, cell_width):
            for j in range(0, cell_height):
                summ += big_a[i][j][d]
        sum_a.append(summ)

    return 2 + np.polyfit(np.log(sum_a), np.log(np.arange(1, max_d + 1)), 1)[0]


if __name__ == '__main__':
    data = []

    with load_default() as loaders:
        for load in loaders:
            load.grayscale_img()
            data.append(
                '{:13}: {}'.format(load.name, get_d(load))
            )

    log('\n'.join(data))