import tensorflow as tf
import os
import re
import hashlib
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile

FLAGS = None
LOCAL_FILE_PATH = os.getcwd()

MAX_NUM_IMAGES_PER_CLASS = 100000

IMAGE_CHANNEL_COUNT = 3
INPUT_IMAGE_WIDTH = 64
INPUT_IMAGE_HEIGHT = 64
LABEL_CLASSES = 2
MINI_BATCH_SIZE = 100

def parse_function(filename, label):
    image_string = tf.read_file(filename)
    decoded_image = tf.image.decode_jpeg(image_string, channels=IMAGE_CHANNEL_COUNT)
    image = tf.image.convert_image_dtype(decoded_image, tf.float32)
    resized_image = tf.image.resize_images(image, [INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT])

    label = tf.one_hot(label, LABEL_CLASSES, axis=0)
    return resized_image, label

def create_image_dataset(filenames, labels):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(buffer_size=len(filenames))
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(MINI_BATCH_SIZE)
    dataset = dataset.prefetch(1)
    return dataset

def create_image_lists(image_dir, testing_percentage, validation_percentage):
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
  if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None
  result = {}
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    print("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))
    if not file_list:
      print('No files found')
      continue
    if len(file_list) < 20:
      print('WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      print('WARNING: Folder {} has more than {} images. Some images will '
            'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(file_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(file_name)
      else:
        training_images.append(file_name)
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result

def get_image_dataset(image_dir, testing_percentage, validation_percentage):
    images = create_image_lists(image_dir, testing_percentage, validation_percentage)
    class_count = len(images.keys())
    if class_count == 0:
        print('No valid folders of images found at ' + image_dir)
        return -1
    if class_count == 1:
        print('Only one valid folder of images found at ' + image_dir +
              ' - multiple classes are needed for classification.')
        return -1

    curr_class = 0
    result = {'testing': {'filenames': [], 'labels': []},
              'training': {'filenames': [], 'labels': []},
              'validation': {'filenames': [], 'labels': []}}

    for cls in images.keys():
        for k in result.keys():
            result[k]['filenames'] += images[cls][k]
            result[k]['labels'] += [curr_class for _ in images[cls][k]]
        curr_class += 1
    
    for k in result.keys():
        result[k] = create_image_dataset(result[k]['filenames'], result[k]['labels'])

    return curr_class, result