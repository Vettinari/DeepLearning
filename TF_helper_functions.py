import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import zipfile

def dataset_to_numpy(dataset, batched = False):
  """
  Converts dataset to numpy array. 
  If batched = True it returns numpy array with dataset batch split.
  """
  data_out, labels_out = [], []
  for images, labels in dataset:
    data_out.append(images.numpy())
    labels_out.append(labels.numpy())
    
  if not batched:
    data_out = np.concatenate(data_out, axis = 0)
    labels_out = np.concatenate(labels_out, axis = 0)

  return data_out, labels_out

def load_and_prep_image(filename, img_shape = 224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).
  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img

def unzip_data(filename):
  """
  Unzips filename into the current working directory.
  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

