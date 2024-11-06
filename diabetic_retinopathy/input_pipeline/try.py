import tensorflow as tf
import os

file_path = r'F:\学校\课程文件\dl lab\IDRID_dataset\images\test\*'
images_ds = tf.data.Dataset.list_files(r'F:\学校\课程文件\dl lab\IDRID_dataset\images\test\*', shuffle = False)
images_ds = images_ds.shuffle(buffer_size = 200)
for file in images_ds.take(5):
  print(file.numpy().decode('utf-8'))
image_count = len(images_ds)
print(image_count)
train_size = int(image_count * 0.8)
train_ds = images_ds.take(train_size)
test_ds = images_ds.skip(train_size)
print(len(train_ds))
print(len(test_ds))

#get the label IDRiD_027.jpg
s = r"F:\学校\课程文件\dl lab\IDRID_dataset\images\test\IDRiD_027.jpg"
s = s.split("\\")[-1]
print(s)


def get_label(file_path):
  x = tf.strings.split(file_path,'\\')
  label = x[-1]
  return label
def process_image(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img)
  img = tf.image.resize(img, [256, 256])

  return img, label
#for t in train_ds.take(4):
 # print(t.numpy().decode('utf-8'))
for label in train_ds.map(get_label):
  print(label.numpy().decode('utf-8'))

train_ds = train_ds.map(process_image)
for img, label in train_ds.take(3):
  print("image: ", img)
  print("label: ", label.numpy().decode('utf-8'))

def scale(image, label):
  return image/255, label

train_ds = train_ds.map(scale)
for image, label in train_ds.take(5):
  print('resized_image: ', image.numpy())
  print('label: ', label.numpy().decode('utf-8'))