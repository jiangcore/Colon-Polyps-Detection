
from PIL import Image # (pip install Pillow)
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%matplotlib inline
import imageio
import os
import time, sys
from IPython.display import clear_output
import pickle
from shutil import copyfile
import sys
import random

MAX_PROCESSS_FILES =2000

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def create_sub_masks(mask_image):
#     print(mask_image.shape, mask_image.size)
    width, height = mask_image.size
#     print(type(mask_image))
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))

            # If the pixel is not black...
            if pixel != 0:
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks


def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    min_poly_filter = 10 # filter out noises
    segmentations = []
    polygons = []
    num_polys = 0
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)
        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        if(poly.exterior):
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    num_polys = len(multi_poly.geoms)
    # print(num_polys)
    if len(multi_poly.bounds) == 4:
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area

        annotation = {
            'segmentation': segmentations,
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': bbox,
            'area': area
        }

        return annotation, num_polys
    else:
        return None, None

ori_data_train = "5000imageori/ori1/"
gt_data_train = "5000imageori/gt1/"
jason_train ="5000imageori/annotations/coco_format_full_train_3k.json"

ori_data_eval = "5000imageori/evori1/"
gt_data_eval = "5000imageori/evgt1/"
jason_eval ="5000imageori/annotations/coco_format_full_eval_3k.json"

ori_data_test = "testimage/ori1/"
gt_data_test = "testimage/gt1/"
jason_test ="testimage/annotations/coco_format_full_test.json"
dataSets = {'train':(ori_data_train,gt_data_train,jason_train ), 'eval':(ori_data_eval,gt_data_eval,jason_eval), 'test':(ori_data_test,gt_data_test,jason_test)}

def convertTIFToJPG(dataset_type="test"):
    ori_data_root = dataSets[dataset_type][0]
    gt_data_root =   dataSets[dataset_type][1]

    for filename in  os.listdir(gt_data_root):
        if filename.split('.')[1] =='tif':
            image = Image.open(os.path.join(gt_data_root,filename))
            newfile = os.path.join(gt_data_root,filename.split('.')[0][1:]+'.jpg')
            print('convert file={} to {}'.format(filename, newfile))
            image.save(newfile)
            image.close()

from shutil import copyfile
def ranSelectTest():
    fileList = os.listdir(ori_data_eval)
    while len(fileList) > 196:
        r = random.randint(0,len(fileList)-1)
        del fileList[r]
    os.system("rm "+ ori_data_test +'*')
    os.system("rm "+ gt_data_test +'*')
    for i, fileName in enumerate(fileList):
        copyfile(os.path.join(ori_data_eval, fileName), os.path.join(ori_data_test, str(i+1)+'.jpg'))
        copyfile(os.path.join(gt_data_eval, 'gt'+fileName[3:] ), os.path.join(gt_data_test,  str(i+1)+'.jpg'))




def make_coco_file(dataset_type="train"):
    final_anns = []
    mean = 0
    std = 0
    ori_data_root = dataSets[dataset_type][0]
    gt_data_root =   dataSets[dataset_type][1]
    # copy_root = "/Users/michaeljiang/DEV/uml/CVCOMP5230/ColonPolypsDetection/data/Polyp_Aug"
    # Define which colors match which categories in the images
    polyp_id = [1]
    is_crowd = 0

    # These ids will be automatically increased as we go
    annotation_id = 1
    image_id = 1
    # Create the annotations
    annotations_main = []
    imgs_without_polyp=[]
    img_eval = []
    image_info_list = []

    filelist = os.listdir(ori_data_root)
    imgnamelist = []
    masknamelist = []
    for filename in filelist:
        imgnamelist.append(filename)
    img_num = len(imgnamelist)
    mask_num = len(masknamelist)
    prob_poly_img_ids = []
    count = 0
    maxfiles = MAX_PROCESSS_FILES if dataset_type != 'eval' else MAX_PROCESSS_FILES/5

    while len(imgnamelist) > maxfiles:
        r = random.randint(0, len(imgnamelist)-1)
        del imgnamelist [r]

    for img_name in imgnamelist:
        if count >= maxfiles :break
        # copyfile(Gt_data_root + img_name, copy_root + img_name)
        tmp = img_name.split('ori')  if dataset_type != 'test' else ['', img_name]

        if img_name.split('.')[1] not in ['jpg','tif']: continue

        if len(tmp) <2: continue
        count += 1

        mask_name = "gt"+tmp[1] if dataset_type!='test' else tmp[1]
        try:
            # orig_image = Image.open(ori_data_root + img_name)
            mask_image = Image.open(gt_data_root + mask_name)
        except FileNotFoundError:
            print ("File not found:" + gt_data_root + mask_name)
            continue
        pil_image = mask_image.convert('L') 
        open_cv_image = np.array(pil_image) 
        new_mask_image = open_cv_image>123
        mask_image = Image.fromarray((new_mask_image*255).astype(np.uint8))
        width, height = mask_image.size
        sub_masks = create_sub_masks(mask_image)
        color = '255'
        if(len(sub_masks)):
            sub_mask = sub_masks[color]
            annotation, num_poly = create_sub_mask_annotation(sub_mask, image_id, 1, annotation_id, is_crowd)
            if annotation is not None:
                annotations_main.append(annotation)
                annotation_id += 1
                image_info = {
                    'license': 1,
                    'file_name': img_name,
                    'coco_url': "",
                    'height': height,
                    'width': width,
                    "date_captured": "2019-04-24 23:12:33",
                    "flickr_url": "",
                    "id": image_id
                }
                image_id += 1
                image_info_list.append(image_info)
                if num_poly > 1:
                    prob_poly_img_ids.append(img_name)



            else:
                imgs_without_polyp.append(img_name)
        else:
            imgs_without_polyp.append(img_name)
        # update_progress(image_id / 5000)
        if count% 100 ==0:
            print ("\r",  str(count) + ' file:'+img_name, end="", flush=True)
    # update_progress(1)
    return annotations_main, imgs_without_polyp, image_info_list, prob_poly_img_ids



coco_base_template = {
    "info": {
        "description": "CVC-ClinicDB",
        "url": "https://giana.grand-challenge.org/PolypDetection/",
        "version": "1.0",
        "year": 2015,
        "contributor": "Jorge Bernal del Nozal <Jorge.Bernal@uab.cat>, Aymeric.Histace@ensea.fr <Aymeric.Histace@ensea.fr>, Shaozhi Jiang, UMass Lowell",
        "date_created": "2019/04/04"
    },
    "licenses": [    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    },
    {
        "url": "http://creativecommons.org/licenses/by-nc/2.0/",
        "id": 2,
        "name": "Attribution-NonCommercial License"
    }],
    "images": [],
    "annotations": [],
    "categories": [{"supercategory": "polyp","id": 1,"name": "polyp"}]
}

#convertTIFToJPG(dataset_type="test")


train_json = coco_base_template.copy()
print("\n =====================")
print("start train..")
train_json["annotations"], imgs_without_polyp, train_json["images"], prob_poly_img_ids = make_coco_file(dataset_type="train")
with open(dataSets['train'][2], 'w') as outfile:
    json.dump(train_json, outfile)
print('write file: ',dataSets['train'][2])
 
print("\n =====================")
print("start eval..")
eval_json = coco_base_template.copy()
eval_json["annotations"], imgs_without_polyp, eval_json["images"], prob_poly_img_ids = make_coco_file(dataset_type="eval")
with open(dataSets['eval'][2], 'w') as outfile:
    json.dump(eval_json, outfile)
# pickle.dump( imgs_without_polyp, open( "/Users/michaeljiang/DEV/uml/CVCOMP5230/ColonPolypsDetection/data/Polyp_Aug/CVC-ClinicDB/annotations/imgs_without_polyp.pkl", "wb" ) )
print('write file: ',dataSets['eval'][2])


'''
print ("rand test")
ranSelectTest()

print("=============")
print("start test..")
test_json = coco_base_template.copy()
test_json["annotations"], imgs_without_polyp, test_json["images"], prob_poly_img_ids = make_coco_file(dataset_type="test")
with open(dataSets['test'][2], 'w') as outfile:
    json.dump(test_json, outfile)

print('write file: ',dataSets['test'][2])


'''