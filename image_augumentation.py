import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A
import os
import glob
import math
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def yolo2bbox(bboxes, w, h):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    ## Desnormalizar coordenadas
    xmin = int(xmin*w)
    ymin = int(ymin*h)
    xmax = int(xmax*w)
    ymax = int(ymax*h)
    return xmin, ymin, xmax, ymax

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    h, w, _ = img.shape
    x_min, y_min , x_max, y_max = yolo2bbox(bbox, w, h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img

def visualize_orig_aug_image(aug_image_df):
    total_images = len(aug_image_df) + 1
    print('total_images:', total_images)
    n_colums = 4 if total_images >= 4 else total_images
    n_rows =  math.ceil(total_images/n_colums) if total_images > n_colums else 1 
    fig, axes = plt.subplots(nrows= n_rows, ncols= n_colums, figsize = (12,10))
    for i in range(n_rows):
        for j in range(n_colums):
            if n_rows > 1:
                ax = axes[i,j]
            else:
                ax = axes[j]
            ax.axis("off")
            if i * n_colums + j >= total_images:
                continue
                  
            if i == 0 and j ==0:
                row = aug_image_df.iloc[0]
                image_path = row["image_path"]
                label_path = row["label_path"]
                ax.set_title("Imagen Original")
            else:
                row = aug_image_df.iloc[i * n_colums + j - 1]
                image_path = row["image_output_path"]
                label_path = row["label_output_path"]
                ax.set_title(f"Imagen Trasformada {i * n_colums + j}")
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           
            bboxes, labels = read_labels(label_path)

            visualize(image, bboxes, labels, category_id_to_name, axis = ax)
            
    
    plt.tight_layout()
    plt.show()


def visualize(image, bboxes, category_ids, category_id_to_name, axis = None):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    if axis is None:
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        plt.imshow(img)
    else:
        axis.imshow(img)

def yolo2albumentations(bbox):
    xmin, ymin = bbox[0]-bbox[2]/2, bbox[1]-bbox[3]/2
    xmax, ymax = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2

    xmax = min(xmax,1.0)
    ymax = min(ymax,1.0)

    return [xmin, ymin, xmax, ymax]

def albumentations2yolo(bbox):
    xmin, ymin, xmax, ymax = bbox
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    w = xmax - ymin
    h = ymax - ymin
    return [xc,yc,w, h]

def read_labels(label_path):
    with open(label_path) as f:
        bboxes = []
        labels = []
        label_lines = f.readlines()
        for label_line in label_lines:
            label, x_c, y_c, w, h= label_line.split(' ')
            x_c = min(float(x_c),1.0) 
            y_c = min(float(y_c),1.0) 
            w = min(round(float(w),4),1.0)
            h = min(round(float(h),4),1.0)
            bboxes.append([x_c, y_c, w, h])
            labels.append(label)
    return bboxes, labels


# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {
    "0": "rice leaf roller",
    "1": "rice leaf caterpillar",
    "2": "paddy stem maggot",
    "3": "asiatic rice borer",
    "4": "yellow rice borer",
    "5": "rice gall midge",
    "6": "Rice Stemfly",
    "7": "brown plant hopper",
    "8": "white backed plant hopper",
    "9": "small brown plant hopper",
    "10": "rice water weevil",
    "11": "rice leafhopper",
    "12": "grain spreader thrips",
    "13": "rice shell pest",
    "14": "grub",
    "15": "mole cricket",
    "16": "wireworm",
    "17": "white margined moth",
    "18": "black cutworm",
    "19": "large cutworm",
    "20": "yellow cutworm",
    "21": "red spider",
    "22": "corn borer",
    "23": "army worm",
    "24": "aphids",
    "25": "Potosiabre vitarsis",
    "26": "peach borer",
    "27": "english grain aphid",
    "28": "green bug",
    "29": "bird cherry-oataphid",
    "30": "wheat blossom midge",
    "31": "penthaleus major",
    "32": "longlegged spider mite",
    "33": "wheat phloeothrips",
    "34": "wheat sawfly",
    "35": "cerodonta denticornis",
    "36": "beet fly",
    "37": "flea beetle",
    "38": "cabbage army worm",
    "39": "beet army worm",
    "40": "Beet spot flies",
    "41": "meadow moth",
    "42": "beet weevil",
    "43": "sericaorient alismots chulsky",
    "44": "alfalfa weevil",
    "45": "flax budworm",
    "46": "alfalfa plant bug",
    "47": "tarnished plant bug",
    "48": "Locustoidea",
    "49": "lytta polita",
    "50": "legume blister beetle",
    "51": "blister beetle",
    "52": "therioaphis maculata Buckton",
    "53": "odontothrips loti",
    "54": "Thrips",
    "55": "alfalfa seed chalcid",
    "56": "Pieris canidia",
    "57": "Apolygus lucorum",
    "58": "Limacodidae",
    "59": "Viteus vitifoliae",
    "60": "Colomerus vitis",
    "61": "Brevipoalpus lewisi McGregor",
    "62": "oides decempunctata",
    "63": "Polyphagotars onemus latus",
    "64": "Pseudococcus comstocki Kuwana",
    "65": "parathrene regalis",
    "66": "Ampelophaga",
    "67": "Lycorma delicatula",
    "68": "Xylotrechus",
    "69": "Cicadella viridis",
    "70": "Miridae",
    "71": "Trialeurodes vaporariorum",
    "72": "Erythroneura apicalis",
    "73": "Papilio xuthus",
    "74": "Panonchus citri McGregor",
    "75": "Phyllocoptes oleiverus ashmead",
    "76": "Icerya purchasi Maskell",
    "77": "Unaspis yanonensis",
    "78": "Ceroplastes rubens",
    "79": "Chrysomphalus aonidum",
    "80": "Parlatoria zizyphus Lucus",
    "81": "Nipaecoccus vastalor",
    "82": "Aleurocanthus spiniferus",
    "83": "Tetradacus c Bactrocera minax",
    "84": "Dacus dorsalis(Hendel)",
    "85": "Bactrocera tsuneonis",
    "86": "Prodenia litura",
    "87": "Adristyrannus",
    "88": "Phyllocnistis citrella Stainton",
    "89": "Toxoptera citricidus",
    "90": "Toxoptera aurantii",
    "91": "Aphis citricola Vander Goot",
    "92": "Scirtothrips dorsalis Hood",
    "93": "Dasineura sp",
    "94": "Lawana imitata Melichar",
    "95": "Salurnis marginella Guerr",
    "96": "Deporaus marginatus Pascoe",
    "97": "Chlumetia transversa",
    "98": "Mango flat beak leafhopper",
    "99": "Rhytidodera bowrinii white",
    "100": "Sternochetus frigidus",
    "101": "Cicadellidae"
  }

def save_transformed(transformed, output_path, image_name, image_id, prefix = "aug"):
    image = transformed['image']
    bboxes = transformed['bboxes']
    labels = transformed['category_ids']
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    path_new_image = f"{output_path}/images/{prefix}_{image_id:06d}_{image_name}.jpg"
    output_path_label = f"{output_path}/labels/{prefix}_{image_id:06d}_{image_name}.txt"
    cv2.imwrite(path_new_image, image)
    save_labels(output_path_label, bboxes, labels)

def save_labels(output_path_label, bboxes, category_ids):
    bbox_coordinates = []
    for bbox, id in zip(bboxes, category_ids):
        xc, yc, wo, ho = bbox
        line = str(id) + " " + str(xc) + " " + str(yc) + " " + str(wo) + " " + str(ho)
        bbox_coordinates.append(line)
    
    with open(output_path_label, 'w') as f:
        f.write("\n".join(bbox_coordinates))

from tqdm.auto import tqdm


def create_imagedf(data_path):
    image_df = pd.DataFrame()
    images_path = f"{data_path}/images"
    all_images = []
    all_images.extend(glob.glob(images_path+'/*.jpg'))
    all_images.extend(glob.glob(images_path+'/*.JPG'))
    all_images.extend(glob.glob(images_path+'/*.jpeg'))
    image_df["image_path"] = all_images
    image_df["image_root"] = image_df["image_path"].apply(os.path.dirname)
    image_df["label_path"] = image_df["image_path"].apply(lambda x: x.replace("images","labels").replace(".jpg",".txt"))
    ## Radom images
    output_path = f"{os.path.dirname(data_path)}/aug"
    os.makedirs(f"{output_path}/images", exist_ok= True)
    os.makedirs(f"{output_path}/labels", exist_ok= True)

    image_aug_df = image_df.sample(frac=2, replace=True, random_state=1).copy()
    image_aug_df.reset_index(inplace=True)
    image_aug_df['IMAGE_ID'] = image_aug_df.index
    image_aug_df["image_output_path"] = image_aug_df['IMAGE_ID'].apply(lambda image_id: f"{output_path}/images/AUG_{image_id:06d}.jpg")
    image_aug_df["label_output_path"] = image_aug_df['IMAGE_ID'].apply(lambda image_id: f"{output_path}/labels/AUG_{image_id:06d}.txt")
    print("Total de imagenes a aumentar:", len(image_aug_df))
    return image_aug_df

def aug_image(row, transform):
    image_path = row["image_path"]
    label_path = row["label_path"]
    image_output_path = row["image_output_path"]
    label_output_path = row["label_output_path"]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes, labels = read_labels(label_path)
    #bboxes = [ yolo2albumentations(bbox) for bbox in bboxes]
    transformed = transform(image=image, bboxes=bboxes, category_ids=labels)
    image = transformed['image']
    bboxes = transformed['bboxes']
    labels = transformed['category_ids']
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(image_output_path, image)
    #bboxes = [ albumentations2yolo(bbox) for bbox in bboxes]
    save_labels(label_output_path, bboxes, labels)


def augmentation_images(data_path, transform):
    image_aug_df = create_imagedf(data_path)
    image_aug_df.apply(lambda row: aug_image(row, transform), axis= 1)
    original_image_path = random.choice(image_aug_df["image_path"].unique().tolist())
    visualize_orig_aug_image(image_aug_df[image_aug_df["image_path"] == original_image_path].copy())
    original_image_path = random.choice(image_aug_df["image_path"].unique().tolist())
    visualize_orig_aug_image(image_aug_df[image_aug_df["image_path"] == original_image_path].copy())
    original_image_path = random.choice(image_aug_df["image_path"].unique().tolist())
    visualize_orig_aug_image(image_aug_df[image_aug_df["image_path"] == original_image_path].copy())
    

data_path = "./datasets/lyromiza/data"

transform = A.Compose(
    [A.HorizontalFlip(p=0.7),
     A.VerticalFlip(p=0.7),
     A.RandomBrightnessContrast(p=0.7),
     A.GaussNoise(p=1.0),
     A.MotionBlur(p=1.0)],
    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
)

augmentation_images(data_path, transform)
#create_imagedf(data_path)

#augmentation_images(data_path, transform)  