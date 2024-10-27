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
    imgs_path = {"img_path": [], "label": []}
    for label in labels:
        imgs_path["img_path"] += [os.path.join(path, label, file) for file in os.listdir(os.path.join(path, label))]
        imgs_path["label"] += [label for _ in range(len(os.listdir(os.path.join(path, label))))]
    imgs_path = pd.DataFrame(imgs_path)
    imgs_path["label"] = imgs_path["label"].map({label: i for i, label in enumerate(labels)})
    return pd.DataFrame(imgs_path)

def load_img(path:str, label:str, image_size:tuple):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.
    return img, label

def create_dataset(imgs_path:pd.DataFrame, image_size:tuple, batch_size:int):
    dataset = tf.data.Dataset.from_tensor_slices((imgs_path["img_path"].values, imgs_path["label"].values))
    dataset = dataset.map(lambda path, label: load_img(path, label, image_size))
    dataset = dataset.shuffle(buffer_size=len(imgs_path))
    dataset = dataset.batch(batch_size)
    return dataset