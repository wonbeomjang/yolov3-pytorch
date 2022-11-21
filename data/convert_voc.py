import shutil
import xml.etree.ElementTree as ET
import glob
import os
import yaml
import zipfile

import tqdm
import kaggle


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    xmin = bbox["xmin"]
    xmax = bbox["xmax"]
    ymin = bbox["ymin"]
    ymax = bbox["ymax"]

    x_center = ((xmin + xmax) / 2) / w
    y_center = ((ymin + ymax) / 2) / h
    width = (xmax - xmin) / w
    height = (ymax - ymin) / h
    assert x_center >= 0 and y_center >= 0 and width >= 0 and height >= 0

    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width height
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


if __name__ == "__main__":
    path = "./VOC2012"
    input_dir = os.path.join(path, "Annotations/")
    output_dir = os.path.join(path, "labels/")
    image_dir = os.path.join(path, "JPEGimages/")

    data_path = "./PASCAL_VOC"
    train_dir = "train"
    val_dir = "val"
    out_yaml = {"names": [], "path": data_path, "train": os.path.join(train_dir, "images"),
                "val": os.path.join(val_dir, "images")}

    kaggle.api.dataset_download_files("huanghanchina/pascal-voc-2012", quiet=False, unzip=True)

    train_dir = os.path.join(data_path, train_dir)
    val_dir = os.path.join(data_path, val_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # create the labels folder (output directory)
    os.mkdir(output_dir)

    # identify all the xml files in the annotations folder (input directory)
    files = glob.glob(os.path.join(input_dir, '*.xml'))
    # loop through each 
    for fil in tqdm.tqdm(sorted(files)):
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
            print(f"{filename} image does not exist!")
            continue

        result = []

        # parse the content of the xml file
        tree = ET.parse(fil)
        path = tree.getroot()
        width = int(path.find("size").find("width").text)
        height = int(path.find("size").find("height").text)

        for obj in path.findall('object'):
            label = obj.find("name").text
            # check for new classes and append to list
            if label not in out_yaml["names"]:
                out_yaml["names"].append(label)
            index = out_yaml["names"].index(label)

            pil_bbox = {x.tag: int(float(x.text)) for x in obj.find("bndbox")}
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
            # convert data to string
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")

        if result:
            # generate a YOLO format text file for each xml file
            with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))

    out_yaml["nc"] = len(out_yaml["names"])

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)

    os.makedirs(os.path.join(train_dir, "images"))
    os.makedirs(os.path.join(train_dir, "labels"))
    os.makedirs(os.path.join(val_dir, "images"))
    os.makedirs(os.path.join(val_dir, "labels"))

    filename = os.listdir(image_dir)
    train_len = int(len(filename) * 0.8)

    for directory, files in zip([train_dir, val_dir], [filename[:train_len], filename[train_len:]]):
        for file in tqdm.tqdm(files):
            file = os.path.basename(file)
            name, extension = file.split(".")

            shutil.copy(os.path.join(image_dir, f"{name}.{extension}"),
                        os.path.join(directory, "images", f"{name}.{extension}"))

            shutil.copy(os.path.join(output_dir, f"{name}.txt"),
                        os.path.join(directory, "labels", f"{name}.txt"))

    if os.path.exists(os.path.join("..", data_path)):
        shutil.rmtree(os.path.join("..", data_path))
    shutil.move(data_path, os.path.join("..", data_path))

    # generate the classes file as reference
    with open('voc.yaml', 'w', encoding='utf8') as f:
        yaml.dump(out_yaml, f)

