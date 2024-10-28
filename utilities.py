import os
import pandas as pd
import tensorflow as tf

def get_imgs_path(path:str, labels:list):
    """get the path of images and their labels

    Args:
        path (str): the directory where the images are stored
        labels (list): the labels of the images (should be the name of the subdirectories)

    Returns:
        pd.DataFrame: a DataFrame containing the path of the images and their labels
    """
    # initialize the dictionary to store the path of the images and their labels
    imgs_path = {"img_path": [], "label": []}
    # get the path of the images and their labels from the directory
    for label in labels:
        imgs_path["img_path"] += [os.path.join(path, label, file) for file in os.listdir(os.path.join(path, label))]
        imgs_path["label"] += [label for _ in range(len(os.listdir(os.path.join(path, label))))]
    # convert the dictionary to a DataFrame
    imgs_path = pd.DataFrame(imgs_path)
    # convert the labels to integers
    imgs_path["label"] = imgs_path["label"].map({label: i for i, label in enumerate(labels)})
    return imgs_path

def load_img(path:str, label:str, image_size:tuple):
    """load and preprocess the image

    Args:
        path (str): the path of the image
        label (str): the label name of the image
        image_size (tuple): the size of the image

    Returns:
        tf.Tensor: the image
        tf.Tensor: the label
    """
    img = tf.io.read_file(path)                 # read the image file
    img = tf.image.decode_jpeg(img, channels=3) # decode the jpeg/jpg image
    img = tf.image.resize(img, image_size)      # resize the image
    img = tf.cast(img, tf.float32) / 255.       # normalize the image
    return img, label

def create_dataset(imgs_path:pd.DataFrame, image_size:tuple, batch_size:int):
    """create a tf.data.Dataset object

    Args:
        imgs_path (pd.DataFrame): the path of the images and their labels
        image_size (tuple): the size of the image
        batch_size (int): the batch size

    Returns:
        tf.data.DataFrame: the tf.data.Dataset object
    """
    # create the dataset in the form of (image, label)
    dataset = tf.data.Dataset.from_tensor_slices((imgs_path["img_path"].values, imgs_path["label"].values))
    # implement the load_img function on the dataset
    dataset = dataset.map(lambda path, label: load_img(path, label, image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(imgs_path))
    # make the dataset into batches
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset