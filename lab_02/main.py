from utils.loaders import ImageLoader, load_default


def get_blue(loader: ImageLoader):
    img = loader.img

    img[:, :, 1] = 0
    img[:, :, 2] = 0

    return img


def get_green(loader: ImageLoader):
    img = loader.img

    img[:, :, 0] = 0
    img[:, :, 2] = 0

    return img


def get_red(loader: ImageLoader):
    img = loader.img

    img[:, :, 0] = 0
    img[:, :, 1] = 0

    return img


if __name__ == '__main__':
    func = [
        get_blue,
        get_green,
        get_red,
    ]

    with load_default() as loaders:
        for load in loaders:
            load.set_img(func.pop()(load))

