import numpy as np

from utils.loaders import ImageLoader, load_default


def grayscale_alg(loader: ImageLoader):
    img = loader.img

    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


if __name__ == '__main__':
    with load_default() as loaders:
        for load in loaders:
            load.set_img(grayscale_alg(load))

