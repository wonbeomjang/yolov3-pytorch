import os.path
from glob import glob
from multiprocessing import Pool
from threading import Thread
import zipfile

import torch
from tqdm import tqdm


def download_and_unzip(url, path):
    print(f"Download {url} to {path}")

    torch.hub.download_url_to_file(url, path, None, True)
    base, file = os.path.splitext(path)
    directory = os.path.join(base, os.path.splitext(os.path.basename(file))[1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    print(f"Unzip {path} to {directory}")

    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(directory)


def convert_label(path):
    train_label_path = glob(os.path.join(path, "coco2017labels", "coco", "labels", "train2017", "*.txt"))
    test_label_path = glob(os.path.join(path, "coco2017labels", "coco", "labels", "test2017", "*.txt"))
    if not os.path.exists(os.path.join(path, "coco2017labels", "coco", "labels", "train")):
        os.makedirs(os.path.join(path, "coco2017labels", "coco", "labels", "train"))
        os.makedirs(os.path.join(path, "coco2017labels", "coco", "labels", "test"))

    for p in tqdm(train_label_path, total=len(train_label_path)):
        with open(p, "r") as f:
            line = f.readlines()

        with open(os.path.join(path, "coco2017labels", "coco", "labels", "train", os.path.basename(p)), "w") as f:
            for l in line:
                convert_label_format(f, l)

    for p in tqdm(test_label_path, total=len(test_label_path)):
        with open(p, "r") as f:
            line = f.readlines()

        with open(os.path.join(path, "coco2017labels", "coco", "labels", "test", os.path.basename(p)), "w") as f:
            for l in line:
                convert_label_format(f, l)


def convert_label_format(f, l):
    c, x, y, w, h = l[:-1].split(" ")
    x, y, w, h = map(float, (x, y, w, h))
    xmin, xmax = x - w / 2, x + w / 2
    ymin, ymax = y - h / 2, y + h / 2
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(1, xmax)
    ymax = min(1, ymax)
    x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
    w, h = xmax - xmin, ymax - ymin
    f.write(f"{c} {x} {y} {w} {h}")


if __name__ == "__main__":
    path = os.path.join(os.curdir, "COCO")

    if not os.path.exists(path):
        os.makedirs(path)

    # Download data
    urls = [
        'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip',
        # 'http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
        # 'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
        # 'http://images.cocodataset.org/zips/test2017.zip'
    ]  # 7G, 41k images (optional)

    threads = []
    print("Download images")

    for url in urls:
        thr = Thread(target=download_and_unzip,
                     args=(url, os.path.join(path, url.split("/")[-1])))
        thr.start()
        threads += [thr]

    for thr in threads:
        thr.join()

    convert_label(path)
