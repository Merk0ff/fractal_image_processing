import cv2
import os

from contextlib import contextmanager


OUT_DIR = './out'
INPUT_DIR = './in'
ACCEPTABLE_EXT = (
    'png',
    'jpg',
    'jpeg',
    'bmp'
)


class ImageLoader:
    _img = ...
    _name = ...

    def print_resolution(self):
        print("{}x{}".format(self._img.shape[0], self._img.shape[1]))

    def _resize(self, w, h):
        self._img = cv2.resize(self._img, (w, h), cv2.INTER_NEAREST)

    def grayscale_img(self):
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

    def __init__(self, name):
        self._name = name
        self._img = cv2.imread('{}/{}'.format(INPUT_DIR, name))

    @property
    def img(self):
        return self._img

    @property
    def name(self):
        return self._name

    def set_img(self, img):
        self._img = img

    def save(self):
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR, exist_ok=True)

        self._resize(512, 512)
        cv2.imwrite('{}/out_{}'.format(OUT_DIR, self._name), self._img)


@contextmanager
def load_default():
    loaders = []

    for file in os.listdir(INPUT_DIR):
        if any(file.endswith('.{}'.format(ext)) for ext in ACCEPTABLE_EXT):
            loaders.append(ImageLoader(file.split('/')[-1]))

    yield loaders

    for loader in loaders:
        loader.save()


def log(lines):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

    with open('{}/log.txt'.format(OUT_DIR), 'w') as f:
        f.write(lines)


