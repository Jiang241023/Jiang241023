import gin
import tensorflow as tf
from tensorflow import keras as k
import numpy as np
from PIL import Image
from multiprocessing import Pool
import warnings
from tqdm import tqdm
import os
import cv2

@gin.configurable
def preprocess(image, label, img_height = 256, img_width = 256):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.
    # Try to normalize with mean and standard deviation
    mean = tf.constant([0.485, 0.456, 0.406])
    std_dev = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std_dev

    return image, label

def augment(image, label):
    """Data augmentation"""
    image = tf.image.random_flip_left_right(image) # image will flip left right randomly
    image = tf.image.random_flip_up_down(image)    # image will flip up down randomly
    image = tf.image.random_brightness(image, max_delta = 0.25)      # image will change it brightness randomly
    random_rotation_layer_image = k.layers.RandomRotation(factor = (-0.5, 0.5)) #A preprocessing layer which randomly rotates images from -180 degrees to 180 degreesaduring training.
    image = random_rotation_layer_image(image, training = True)
    return image, label


def load_image(file_path, label, img_height, img_width, save_path):

    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels = 3)
    image, label = preprocess(image, label, img_height, img_width)
    image, label = augment(image, label)
    os.makedirs(save_path, exist_ok = True)
    base_name = os.path.basename(file_path)
    save_path = os.path.join(save_path, f"revised_{base_name}")
    encoded_image = tf.io.encode_jpeg(tf.cast(image * 255, tf.uint8))
    #Write the encoded image data to the specified path save_path
    tf.io.write_file(save_path, encoded_image)

    return image, label


#selected_images = [
#    "IDRiD_021", "IDRiD_079", "IDRiD_105", "IDRiD_194", "IDRiD_198",
#    "IDRiD_203", "IDRiD_256", "IDRiD_265", "IDRiD_266", "IDRiD_279",
#    "IDRiD_290", "IDRiD_291", "IDRiD_301", "IDRiD_302", "IDRiD_304",
#    "IDRiD_379", "IDRiD_388", "IDRiD_402", "IDRiD_403", "IDRiD_408"
#]

#for n in selected_images:
##    file_path = fr"F:\学校\课程文件\dl lab\idrid\IDRID_dataset\images\train\{n}.jpg"
#    save_path = r"F:\学校\课程文件\dl lab\idrid\IDRID_dataset\images_revized\train"
#    image, label = load_image(file_path, label = n, img_height = 256, img_width = 256, save_path = save_path)



sample_data_path = r"F:\学校\课程文件\dl lab\IDRID_dataset\images\train"
output_folder = r"F:\学校\课程文件\dl lab\IDRID_dataset\images_revized"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def trim(image):

    percentage = 0.02
    img = np.array(image)
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # Convert to grayscale to simply the process
    # create the binary mask , to get the background from actual content
    img_gray = img_gray > 0.1 * np.mean(img_gray[img_gray!=0])
    # calculate the row wise and column wise sums to find where the significant content exists
    row_sums = np.sum(img_gray, axis = 1)
    col_sums = np.sum(img_gray, axis = 0)
    rows = np.where(row_sums > img.shape[1] * percentage)[0] # return the rows index of rows which contain atleast 2% of its content
    cols = np.where (col_sums > img.shape[0] * percentage)[0]
    # find the min and max rows and columns for croping
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    im_crop = img[min_row : max_row +1 , min_col : max_col+1]
    return Image.fromarray(im_crop)

def resize_main_aspect(image, desired_size):
    old_size = image.size
    ratio = float(desired_size)/ max(old_size) # resize ratio
    new_size = tuple([int(x * ratio) for x in old_size]) # (N,M) N,M are new size
    im = image.resize(new_size, Image.LANCZOS) # a filter to smooth image when resize, helps to reduce artifacts in the reduced image
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0])//2 , (desired_size - new_size[1])//2)) # paster the image on the new square background
    return new_im

def save_single(args): # helpfull for multiprocessing
    img_file, input_path_folder, output_path_folder, output_size = args
    image_org = Image.open(os.path.join(input_path_folder, img_file))
    image = trim(image_org)
    image = resize_main_aspect(image, desired_size= output_size[0])
    image.save(os.path.join(output_path_folder , img_file))



def multi_image_resize(input_path_folder, output_path_folder, output_size=None):
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=100")
        exit()

    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)

    jobs = [
        (file, input_path_folder, output_path_folder, output_size)
        for file in os.listdir(input_path_folder)
        if os.path.isfile(os.path.join(input_path_folder,file))
    ]

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))
if __name__ == '__main__':
    multi_image_resize(sample_data_path, output_folder, output_size = (256,256))
