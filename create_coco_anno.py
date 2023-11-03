import os
import cv2
import json
import random


def generate_categories(filename: str) -> list:
    categories = []
    with open(filename) as f:
        lines_of_class = f.readlines()
    # num_categories = len(lines_of_class)
    for idx, line in enumerate(lines_of_class):
        name = line.strip()
        category = {
            "id": idx,
            "name": name,
            "supercategory": "None"
        }
        categories.append(category)
    return categories


def generate_images(dirpath: str) -> list:
    images = []
    images_list = os.listdir(dirpath)
    images_list.sort()
    num_images = len(images_list)
    for idx, image_path in enumerate(images_list):
        img = cv2.imread(os.path.join(dirpath, image_path))
        height, width = img.shape[0], img.shape[1]
        image = {
            "id": idx,
            "width": width,
            "height": height,
            "file_name": image_path
        }
        images.append(image)
        print(f"{idx + 1} of {num_images} image done.")
    return images


def generate_annotations(dirpath: str, images: list) -> list:
    annotations = []
    width, height = 1920, 1080
    anno_list = os.listdir(dirpath)
    anno_list.sort()
    instance_id = 0
    num_images = len(anno_list)
    for idx, anno_path in enumerate(anno_list):
        # 1.read instances of one image
        with open(os.path.join(dirpath, anno_path)) as f:
            lines_of_annos = f.readlines()
        # 2.testify the idx
        img_name = images[idx]["file_name"].split(".")[0]
        anno_name = anno_path.split(".")[0]
        assert img_name == anno_name, \
            "wrong image id"
        for line in lines_of_annos:
            line = line.strip().split(" ")
            category_id = int(line[0])
            bbox = cxcywh_to_xywh([float(item) for item in line[1:]], [width, height])
            polygon = xywh_to_polygon(bbox)
            area = calculate_area(bbox)
            annotation = {
                "id": instance_id,
                "image_id": idx,
                "category_id": category_id,
                "segmentation": [polygon],  # need change
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            }
            annotations.append(annotation)
            instance_id += 1
        print(f"{idx + 1} of {num_images} image done.")
    return annotations


def cxcywh_to_xywh(box: list, img_shape: list) -> list:
    [cx, cy, w, h] = box
    center = [cx*img_shape[0], cy*img_shape[1]]
    width, height = w*img_shape[0], h*img_shape[1]
    pt1 = [center[0] - width/2, center[1] - height/2]
    return pt1 + [width, height]


def xywh_to_polygon(box: list):
    [w, h] = box[2:]
    pt1 = box[0:2]
    pt2 = [pt1[0] + w, pt1[1]]
    pt3 = [pt2[0], pt2[1] + h]
    pt4 = [pt1[0], pt1[1] + h]
    return pt1 + pt2 + pt3 + pt4


def calculate_area(box: list) -> float:
    return box[-1]*box[-2]


def split_train_val_data(filename: str) -> None:
    train_data = {"images": [], "annotations": [], "categories": []}
    val_data = {"images": [], "annotations": [], "categories": []}
    val_num = 41
    # read total data
    with open(filename) as f:
        full_data = json.load(f)
    total_num = len(full_data["images"])
    # record categories
    train_data["categories"] = full_data["categories"]
    val_data["categories"] = full_data["categories"]
    # select train ids and val ids
    train_num = total_num - val_num
    train_ids = random.sample(list(range(total_num)), train_num)
    train_ids.sort()
    # record images
    for image in full_data["images"]:
        if image["id"] in train_ids:
            train_data["images"].append(image)
        else:
            val_data["images"].append(image)
    # record annotations
    for item in full_data["annotations"]:
        if item["image_id"] in train_ids:
            train_data["annotations"].append(item)
        else:
            val_data["annotations"].append(item)
    # save as json file
    train_file = "dataset/annotations/train_det_data.json"
    val_file = "dataset/annotations/val_det_data.json"
    with open(train_file, "w") as f:
        json.dump(train_data, f)
        print("train set saved!")
    with open(val_file, "w") as f:
        json.dump(val_data, f)
        print("val set saved!")


def rename_files(dirpath: str, augtype="h"):
    filelist = os.listdir(dirpath)
    for file in filelist:
        [old_name, file_type] = file.split(".")
        new_name = old_name + "_" + augtype
        print(f"rename {old_name} to {new_name}")
        os.rename(os.path.join(dirpath, file), os.path.join(dirpath, new_name + "." + file_type))


def pick_flip_images(base_file, horizon_file, vertical_file):
    with open(base_file) as f:
        base_det_data = json.load(f)
    with open(horizon_file) as f:
        horizon_data = json.load(f)
    with open(vertical_file) as f:
        vertical_data = json.load(f)
    # change the ids of horizon data
    for img in horizon_data["images"]:
        img['id'] += 291
    for item in horizon_data["annotations"]:
        item["image_id"] += 291
        item["id"] += 1527
    # change the ids of vertical data
    for img in vertical_data["images"]:
        img['id'] += 582
    for item in vertical_data["annotations"]:
        item["image_id"] += 582
        item["id"] += 3054
    # collect names of train images
    train_img_names = []
    for img in base_det_data["images"]:
        train_img_names.append(img["file_name"].split(".")[0])
    # pick the horizon images
    horizon_img_ids = []
    for img in horizon_data["images"]:
        img_name = img["file_name"].split(".")[0]
        img_name = img_name.split("_h")[0]
        if img_name in train_img_names:
            base_det_data["images"].append(img)
            horizon_img_ids.append(img["id"])
            print(f"{img_name} picked and saved to base_det_data.")
    # pick the horizon annotations
    for item in horizon_data["annotations"]:
        if item["image_id"] in horizon_img_ids:
            base_det_data["annotations"].append(item)
            print(f"annotation:{item['id']} picked and saved to base_det_data.")
    # pick the vertical images
    vertical_img_ids = []
    for img in vertical_data["images"]:
        img_name = img["file_name"].split(".")[0]
        img_name = img_name.split("_v")[0]
        if img_name in train_img_names:
            base_det_data["images"].append(img)
            vertical_img_ids.append(img["id"])
            print(f"{img_name} picked and saved to base_det_data.")
    # pick the vertical annotations
    for item in vertical_data["annotations"]:
        if item["image_id"] in vertical_img_ids:
            base_det_data["annotations"].append(item)
            print(f"annotation:{item['id']} picked and saved to base_det_data.")
    # save the augmented train data
    augmented_train_file = "/share/zlh/cv_task/annotations/aug_train_det_data.json"
    with open(augmented_train_file, "w") as f:
        json.dump(base_det_data, f)
        print(f"{augmented_train_file} saved.")
    # save the horizon data with new ids
    horizon_file = "/share/zlh/cv_task_hor/horizon_det_data.json"
    with open(horizon_file, "w") as f:
        json.dump(horizon_data, f)
        print(f"{horizon_file} saved.")
    # save the vertical data with new ids
    vertical_file = "/share/zlh/cv_task_ver/vertical_det_data.json"
    with open(vertical_file, "w") as f:
        json.dump(vertical_data, f)
        print(f"{vertical_file} saved.")


def pick_flip_test_images(base_train_file, base_val_file, horizon_file, vertical_file):
    with open(base_train_file) as f:
        base_train_data = json.load(f)
    with open(base_val_file) as f:
        base_val_data = json.load(f)
    with open(horizon_file) as f:
        horizon_data = json.load(f)
    with open(vertical_file) as f:
        vertical_data = json.load(f)
    # collect names of train images
    train_img_names = []
    for img in base_train_data["images"]:
        train_img_names.append(img["file_name"].split(".")[0])
    # pick the horizon images
    horizon_img_ids = []
    for img in horizon_data["images"]:
        img_name = img["file_name"].split(".")[0]
        img_name = img_name.split("_h")[0]
        if img_name not in train_img_names:
            base_val_data["images"].append(img)
            horizon_img_ids.append(img["id"])
            print(f"{img_name} picked and saved to base_det_data.")
    # pick the horizon annotations
    for item in horizon_data["annotations"]:
        if item["image_id"] not in horizon_img_ids:
            base_val_data["annotations"].append(item)
            print(f"annotation:{item['id']} picked and saved to base_det_data.")
    # pick the vertical images
    vertical_img_ids = []
    for img in vertical_data["images"]:
        img_name = img["file_name"].split(".")[0]
        img_name = img_name.split("_v")[0]
        if img_name not in train_img_names:
            base_val_data["images"].append(img)
            vertical_img_ids.append(img["id"])
            print(f"{img_name} picked and saved to base_det_data.")
    # pick the vertical annotations
    for item in vertical_data["annotations"]:
        if item["image_id"] not in vertical_img_ids:
            base_val_data["annotations"].append(item)
            print(f"annotation:{item['id']} picked and saved to base_det_data.")
    # save the augmented train data
    augmented_val_file = "/share/zlh/cv_task/annotations/aug_val_det_data.json"
    with open(augmented_val_file, "w") as f:
        json.dump(base_val_data, f)
        print(f"{augmented_val_file} saved.")


if __name__ == "__main__":
    class_file = "/share/zlh/cv_task_ver/classes.txt"
    images_path = "/share/zlh/cv_task_ver/images"
    anno_path = "/share/zlh/cv_task_ver/labels"

    # categories = generate_categories(class_file)
    # for item in categories:
    #     print(item)

    # images = generate_images(images_path)
    # print(images[0])

    # annotations = generate_annotations(anno_path, images)
    # print(annotations[0])

    # det_data = {
    #     "images": images,
    #     "annotations": annotations,
    #     "categories": categories
    # }

    json_path = "/share/zlh/cv_task_ver/det_data.json"
    # with open(json_path, "w") as f:
    #     json.dump(det_data, f)
    # print(f"{json_path} saved!")

    # split_train_val_data(json_path)
    # val_path = "dataset/annotations/val_det_data.json"
    # with open(val_path) as f:
    #     val_data = json.load(f)
    # val_images = val_data["images"]
    # with open("val.txt", "w") as f:
    #     for img in val_images:
    #         f.write(img["file_name"] + "\n")

    # horizon = "/share/zlh/cv_task_ver/"
    # image_dir = horizon + "images"
    # label_dir = horizon + "labels"
    # rename_files(image_dir, "v")
    # rename_files(label_dir, "v")

    base_train = "/share/zlh/cv_task/annotations/train_det_data.json"
    base_val = "/share/zlh/cv_task/annotations/val_det_data.json"
    horizon = "/share/zlh/cv_task_hor/horizon_det_data.json"
    vertical = "/share/zlh/cv_task_ver/vertical_det_data.json"
    aug = "/share/zlh/cv_task/annotations/aug_train_det_data.json"
    # with open(aug) as f:
    #     data = json.load(f)
    # print(len(data["annotations"]))
    # print(len(data["images"]))

    # pick_flip_images(base, horizon, vertical)
    pick_flip_test_images(base_train, base_val, horizon, vertical)
