import cv2
import numpy as np
import matplotlib.pyplot as plt


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.shape[0:2][::-1]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.zeros((size[1], size[0], 3), np.uint8)
    new_image.fill(128)
    dx = (w-nw)//2
    dy = (h-nh)//2
    new_image[dy:dy+nh, dx:dx+nw,:] = image
    return new_image

image = cv2.imread("./datasets/lyromiza/data/images/IMG00005_jpg.rf.6b275a6a17ca4ad4f787223b31a307cf.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

new_size = (640, 640)

fig, axes = plt.subplots(1,3, figsize = (12,10))

image_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
image_resized_letterbox = letterbox_image(image, new_size)
axes[0].imshow(image)
axes[0].set_title("Tamaño Original")
axes[1].imshow(image_resized)
axes[1].set_title("Redimensionada")
axes[2].imshow(image_resized_letterbox)
axes[2].set_title("Redimensionada Letterbox")
plt.tight_layout()
plt.show()

#def letterbox_image(image, size):
