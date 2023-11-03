import os
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
from create_coco_anno import *
import shutil

def save_img_ann(img_dir, ann_dir, img_name, img, dst_ann, cropno=None):
    '''
    this function is to save images and annotations
    img_dir:    the parent directory to save image
    ann_dir:    the parent directory to save annotation
    img_name:   the name of image
    img:        the images we processed
    dst_ann:    the annotations we processed
    cropno:     number of crops
    '''
    if cropno is not None:
        img_name = img_name.split('.')
        img_name[0] = img_name[0] + '_' + str(cropno)   # add suffix to image name
        img_name = '.'.join(img_name)

    # the real path to save image and corresponding annotations
    new_img = os.path.join(img_dir, img_name)
    new_anno = os.path.join(ann_dir, img_name.replace('jpg', 'txt'))

    # save image and annotations
    img.save(new_img)
    with open(new_anno, 'w') as f:
        for anno in dst_ann:
            tmp_ann = [str(x) for x in anno]
            dealt_anno = ' '.join(tmp_ann) + '\n'
            f.writelines(dealt_anno)
    
    print('  Deal image: '
            , img_name
            , '\tsaved to\t'
              , '/'.join(new_img.split('/')[-3:-1]))

def process_image(sor_path, dst_path, horizontalflip=False, verticalflip=False, cropno=None):
    '''
    this function is to process image with choices of horizontal, vertical or crop
    sor_path:       the sorce parent path with images and annotations to deal with
    dst_path:       the target parent path to save processed images and annotations
    horizontalflip: flip image flag horizontally
    veriticalflip:  flip image flag vertically
    cropno:         number of crops
    '''
    # create output folder
    img_dst_path = os.path.join(dst_path, 'images')
    if img_dst_path is not None:
        os.makedirs(img_dst_path, exist_ok=True)
    ann_dst_path = os.path.join(dst_path, 'labels')
    if ann_dst_path is not None:
        os.makedirs(ann_dst_path, exist_ok=True)

    # process images
    img_sor_path = os.path.join(sor_path, 'images')
    img_list = os.listdir(img_sor_path)
    for i in img_list:
        # read images
        old_img = os.path.join(img_sor_path, i)
        img = Image.open(old_img)
        img = img.convert('RGB')
        width, height = img.size

        # assert annotations
        old_ann = old_img.replace('images', 'labels').replace('jpg', 'txt')
        if os.path.exists(old_ann) is False:
            print('no annotation!!!!!!!!!!!!!!!!!!!!!!')
            break

        # read instances of one image
        with open(old_ann) as f:
            raw_annos = f.readlines()
        old_list = []
        for anno in raw_annos:
            anno = anno.strip().split(' ')
            label, center_x, center_y, w, h = map(lambda i: float(i), anno)
            label = int(label)
            old_list.append([label, center_x, center_y, w, h])
        dst_ann = old_list      # save annotations

        # deal with picture
        if horizontalflip == True:  # flip image horizontally
            img = TF.hflip(img)
            for index, anno in enumerate(old_list):
                [label, center_x, center_y, w, h] = anno
                center_x = 1 - center_x
                dst_ann[index][1] = center_x
        if verticalflip == True:    # flip image vertically
            img = TF.vflip(img)
            for index, anno in enumerate(old_list):
                [label, center_x, center_y, w, h] = anno
                center_y = 1 - center_y
                dst_ann[index][2] = center_y
        if cropno is not None:      # crop image
            x_min = width
            x_max = 0
            y_min = height
            y_max = 0
            for anno in old_list:
                [label, center_x, center_y, w, h] = anno
                if center_x*width < (x_min + w/2):
                    x_min = center_x*width - w/2
                if (center_x*width + w/2) > x_max:
                    x_max = center_x*width + w/2
                if center_y*height < (y_min + h/2):
                    y_min = center_y*height - h/2
                if (center_y*height + h/2) > y_max:
                    y_max = center_y*height + h/2
            for time in range(cropno):
                # Randomly select a random top left corner
                if x_min >= 0:
                    crop_x_min = np.random.uniform(0, x_min)
                else:
                    crop_x_min = 0
                if y_min >= 0:
                    crop_y_min = np.random.uniform(0, y_min)
                else:
                    crop_y_min = 0

                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = x_max - crop_x_min
                crop_height_min = y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                assert (crop_height_max>crop_height_min and crop_height_max>crop_height_min), \
                    'wrong image crop size'
                
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                for index, anno in enumerate(old_list):
                    [label, center_x, center_y, w, h] = anno
                    center_x = (center_x*width - offset_x)/crop_width
                    center_y = (center_y*height - offset_y)/crop_height
                    dst_ann[index][1] = center_x
                    dst_ann[index][2] = center_y

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
                # save image and annotation
                save_img_ann(img_dst_path, ann_dst_path, i, img, dst_ann, time)
        else:
            # save image and annotation
            save_img_ann(img_dst_path, ann_dst_path, i, img, dst_ann, cropno)
        
        # copy class file
        dst_class_path = os.path.join(dst_path, 'classes.txt')
        if os.path.exists(dst_class_path) is False:
            sor_class_path = os.path.join(sor_path, 'classes.txt')
            shutil.copy(sor_class_path, dst_class_path)

def convert_to_coco(path):
    '''
    This function is to transfer labels to coco format
    path:   dataset's path
    '''
    # read file
    class_file = os.path.join(path, 'classes.txt')
    images_path = os.path.join(path, 'images')
    anno_path = os.path.join(path, 'labels')

    # read catgeories
    categories = generate_categories(class_file)
    # for item in categories:
    #     print(item)

    # read_images
    images = generate_images(images_path)
    # print(images[0])

    # read annotations
    annotations = generate_annotations(anno_path, images)
    print(annotations[0])

    # construct coco format
    det_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # save json file
    json_path = os.path.join(path, 'det_data.json')
    with open(json_path, "w") as f:
        json.dump(det_data, f)
    print(f"{json_path} saved!")

def split_train_valid(path, valrate_or_valnum=True, value=41):
    '''
    this image is to split trian and valid data and save their annotations
    path:               dataset's path
    valrate_or_valnum:  flag of valation choice
                        True: use valid data number     False: use valid data rate
    value:              valid data number or rate
    '''
    # read total data
    origin_json = os.path.join(path, 'det_data.json')
    with open(origin_json) as f:
        full_data = json.load(f)
    total_num = len(full_data["images"])
    dst_json_path = os.path.join(path, 'annotations')
    if dst_json_path is not None:
        os.makedirs(dst_json_path, exist_ok=True)

    # split to train or val
    train_data = {"images": [], "annotations": [], "categories": []}
    val_data = {"images": [], "annotations": [], "categories": []}
    if valrate_or_valnum == True:
        val_num = value
    else:
        val_num = int(total_num*value)
    
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
    train_file = os.path.join(dst_json_path, 'train_det_data.json')
    val_file = os.path.join(dst_json_path, 'val_det_data.json')
    with open(train_file, "w") as f:
        json.dump(train_data, f)
        print("train set saved!")
    with open(val_file, "w") as f:
        json.dump(val_data, f)
        print("val set saved!")

if __name__ == "__main__":
    '''
    you should place this file , dataset folder and create_coco_anno.py in the same level directory
    '''
    sor_path = 'dataset'
    dst_path = 'dataset_H'
    horizontalflip = True
    verticalflip = False
    cropno = None # !!!!choose from {None, 1, 2, 3, ...}
    process_image(sor_path, dst_path, horizontalflip, verticalflip, cropno)
    convert_to_coco(dst_path)
    split_train_valid(dst_path)
