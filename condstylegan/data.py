"""
data.py
"""
import os

import numpy as np
import tensorflow as tf

class DataGenerator():

    def __init__(self, **kwargs):
        self.source_dir = kwargs['source_dir']
        self.shape = kwargs['shape']
        self.batch_size = kwargs['batch_size']
        self.tags = kwargs['tag_dict']

    def parse(self, file, label):
        img = (tf.cast(tf.image.decode_jpeg(tf.io.read_file(file)), tf.float32) - 127.5)/127.5
        img = tf.image.resize(img, self.shape)

        return img, label

    def gen_dataset(self):
        empty_label_vector = np.zeros(len(self.tags.keys()))
        all_imgs = []
        all_labels = []
        for idx, (concept, labels) in enumerate(self.tags.items()):
            concept_samples = []
            if labels['annotation_dir']:
                for annotation in labels['annotation']:
                    with open(os.path.join(labels['annotation_dir'], annotation + '.txt'), 'r') as f:
                        samples = f.readlines()
                    concept_samples += samples
            # TODO: User tags in MIRFLICKR25K take different format.

            # Append images to full list
            concept_samples = list(set([os.path.join(self.source_dir, 'im' + s.replace('\n', '') + '.jpg')
                                        for s in concept_samples]))
            all_imgs += concept_samples

            # Append label vectors to full list
            label_vector = empty_label_vector.copy()
            label_vector[idx] = 1
            all_labels += [label_vector] * len(concept_samples)


        dataset = tf.data.Dataset.from_tensor_slices((all_imgs, all_labels)).\
            shuffle(len(all_imgs)).repeat().map(self.parse).batch(self.batch_size)

        return dataset
