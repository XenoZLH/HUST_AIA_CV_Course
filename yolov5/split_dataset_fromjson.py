import os
import random
import shutil
import json

def label_to_file(label_path, class_file):
    # get class name
    with open(class_file, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    # class_len = len(class_names)
    class_num = [0 for _ in class_names]  # all number of each class

    each_file = dict()
    
    label_list = os.listdir(label_path)
    for label in label_list:
        ann_path = os.path.join(label_path, label)
        with open(ann_path, encoding='utf-8') as f:
            annotations = f.readlines()
        annotations = [c.strip().split(' ') for c in annotations]

        each_file_class = [0 for _ in class_names]
        for ann in annotations:
            class_num[int(ann[0])] += 1
            each_file_class[int(ann[0])] += 1
        
        file_class = label.split('_')[0]
        if each_file.get(file_class) is  None:
            each_file[file_class] = dict()
            # each_file[file_class] = {'file_class': file_class}

            for index, cla in enumerate(class_names):
                each_file[file_class][cla] = each_file_class[index]
        else:
            for index, cla in enumerate(class_names):
                each_file[file_class][cla] += each_file_class[index]
    
    all = dict()
    for index, cla in enumerate(class_names):
        all[cla] = dict()
        all[cla]['all_number'] = class_num[index]
        all[cla]['files'] = dict()
        for each in each_file:
            all[cla]['files'][each] = each_file[each][cla]

    print(all)

def find_soilcompactor(label_path):

    label_list = os.listdir(label_path)
    files = []
    for label in label_list:
        ann_path = os.path.join(label_path, label)
        with open(ann_path, encoding='utf-8') as f:
            annotations = f.readlines()
        annotations = [c.strip().split(' ') for c in annotations]

        for ann in annotations:
            if ann[0] == '3':
                files.append([label])

    print(files)

def split_dataset(dataset_path):
    ori_image_path = os.path.join(dataset_path, 'images')
    ori_label_path = os.path.join(dataset_path, 'labels')
    train_image_path = os.path.join(ori_image_path, 'train')
    if train_image_path is not None:
        os.makedirs(train_image_path, exist_ok=True)
    train_label_path = os.path.join(ori_label_path, 'train')
    if train_label_path is not None:
        os.makedirs(train_label_path, exist_ok=True)
    valid_image_path = os.path.join(ori_image_path, 'valid')
    if valid_image_path is not None:
        os.makedirs(valid_image_path, exist_ok=True)
    valid_label_path = os.path.join(ori_label_path, 'valid')
    if valid_label_path is not None:
        os.makedirs(valid_label_path, exist_ok=True)


    train_json_path = os.path.join(dataset_path, 'train_det_data.json')
    valid_json_path = os.path.join(dataset_path, 'val_det_data.json')
    with open(train_json_path) as f:
        train_json = json.load(f)
    with open(valid_json_path) as f:
        valid_json = json.load(f)
    
    for image in train_json['images']:
        img_name = image['file_name']
        lab_name = img_name.replace('jpg', 'txt')

        old_img = os.path.join(ori_image_path, img_name)
        new_img = os.path.join(train_image_path, img_name)
        shutil.move(old_img, new_img)

        old_lab = os.path.join(ori_label_path, lab_name)
        new_lab = os.path.join(train_label_path, lab_name)
        shutil.move(old_lab, new_lab)

        print('move ', old_img, ' to ', new_img, ' successful!')

    for image in valid_json['images']:
        img_name = image['file_name']
        lab_name = img_name.replace('jpg', 'txt')

        old_img = os.path.join(ori_image_path, img_name)
        new_img = os.path.join(valid_image_path, img_name)
        shutil.move(old_img, new_img)

        old_lab = os.path.join(ori_label_path, lab_name)
        new_lab = os.path.join(valid_label_path, lab_name)
        shutil.move(old_lab, new_lab)

        print('move ', old_img, ' to ', new_img, ' successful!')



if __name__ == "__main__":
    # label_path = 'dataset/labels'
    # class_file = '/home/kdzhang/CVcourse/dataset/classes.txt'
    # label_to_file(label_path, class_file)
    # find_soilcompactor(label_path)
    dataset_path = 'data/dataset'
    split_dataset(dataset_path)