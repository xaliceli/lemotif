"""
data.py
"""
from collections import Counter
import glob
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans
import scipy.stats as st
import tensorflow as tf

from colors import colors_dict


def get_dominant_color(image, k=6, image_processing_size=(200, 200), thresh=0.0):
    """
    Adapted from https://adamspannbauer.github.io/2018/03/02/app-icon-dominant-colors/
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # resize image if new dims provided
    if image_processing_size is not None and \
            image.shape[0] > image_processing_size[0] or image.shape[1] > image_processing_size[1]:
        image = cv2.resize(image, image_processing_size,
                           interpolation=cv2.INTER_AREA)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels).most_common(k)

    output, mapped = [], []
    for count in label_counts:
        if count[1] > thresh*image.shape[0]:
            color = clt.cluster_centers_[count[0]]
            output.append(color)

    return np.array(output)


def convert_to_lab(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return image.reshape((image.shape[0] * image.shape[1], 3))


def map_to_colors(lab_in, lemotif_colors, sim_thresh=1500, freq_thresh=0.2, one_hot=True):
    all_colors = []
    for color, info in lemotif_colors.items():
        all_colors.append(lemotif_colors[color]['lab'])
    targets = np.array(all_colors)

    diff = np.sum(np.square(lab_in[:, None, ...] - targets), axis=(2))

    best_match = np.argsort(-diff, axis=1)[:,diff.shape[1]-1::]

    if one_hot:
        matches = np.zeros(diff.shape[1], dtype='float32')
        for target in range(diff.shape[1]):
            pixel_idx = np.where(best_match == target)[0]
            if len(pixel_idx) > 0:
                pixel_diffs = diff[pixel_idx, np.ones((len(pixel_idx)), dtype='int32')*target]
                n_below_thresh = np.where(pixel_diffs < sim_thresh)[0].shape[0]
                if n_below_thresh > freq_thresh*diff.shape[0]:
                    matches[target] = 1.0
        return matches
    else:
        matches = []
        for match in range(best_match.shape[0]):
            if diff[match, best_match[match][0]] < sim_thresh:
                matches.append((list(lemotif_colors)[best_match[match][0]],
                                lemotif_colors[list(lemotif_colors)[best_match[match][0]]]['color']))
        return set(matches)


def generate_crops(source_dir, out_dir, crop_size, up_factor):
    img_files = []
    for e in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png']:
        img_files += glob.glob(os.path.join(source_dir, e))
    for img_file in img_files:
        img = cv2.imread(img_file)
        if img.shape[0] <= crop_size or img.shape[1] <= crop_size:
            max_factor = up_factor*crop_size/min(img.shape[0], img.shape[1])
            img = cv2.resize(img, (int(max_factor*img.shape[1]), int(max_factor*img.shape[0])))
        for c in range(int(img.shape[0]/crop_size)):
            rand_row, rand_col = np.random.randint(0, img.shape[0]-256), np.random.randint(0, img.shape[1]-256)
            crop = img[rand_row:rand_row+256, rand_col:rand_col+256]
            lab = convert_to_lab(crop)
            mapped = map_to_colors(lab, colors_dict, sim_thresh=1250, freq_thresh=0.3)

            if np.sum(mapped) > 0:
                extension = img_file.split('.')[-1]
                name = img_file.replace('.' + extension, '')\
                    .replace(source_dir, out_dir)
                new_file = name + '_' + str(c) + '.jpg'

                cv2.imwrite(new_file, crop)
                np.savetxt(new_file.replace('jpg', 'txt'), mapped)
                np.save(new_file.replace('.jpg', ''), mapped)


class DataGenerator():

    def __init__(self, **kwargs):
        self.source_dir = kwargs['source_dir']
        self.shape = kwargs['shape']
        self.batch_size = kwargs['batch_size']
        self.tags = kwargs['tag_dict']

    def extract_colors(self, img):
        dominant = get_dominant_color(img, thresh=0.2)
        mapped = map_to_colors(dominant, self.tags)
        return mapped

    def parse_cond(self, file, label_file):
        img = tf.image.decode_jpeg(tf.io.read_file(file), channels=3)
        img = (tf.image.resize(img, self.shape, align_corners=True, preserve_aspect_ratio=False) - 127.5)/127.5
        label = tf.py_func(np.load, [label_file], tf.float32)

        return img, label

    def parse_blur(self, file):
        img = tf.image.decode_jpeg(tf.io.read_file(file), channels=3)
        img = (tf.image.resize(img, self.shape, align_corners=True, preserve_aspect_ratio=False) - 127.5)/127.5

        return img

    def gen_dataset(self, parser):
        img_files = []
        for e in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
            img_files += glob.glob(os.path.join(self.source_dir, e))

        if parser == 'parse_cond':
            labels = [img.replace(img.split('.')[-1], 'npy') for img in img_files]
            dataset_source = (img_files, labels)
        else:
            dataset_source = img_files

        dataset = tf.data.Dataset.from_tensor_slices(dataset_source).repeat().\
            shuffle(len(img_files)).map(getattr(self, parser)).batch(self.batch_size)

        return dataset
