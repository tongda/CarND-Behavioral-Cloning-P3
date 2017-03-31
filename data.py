import csv
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def read_data(path):
    correction = [0, -0.2, 0.2]
    images = []
    measurements = []
    with open(os.path.join(path, 'driving_log.csv'), 'r') as csvfile:
        data_reader = csv.reader(csvfile)
        for line in data_reader:
            for i in range(3):
                img_path = os.path.join(path, 'IMG', line[i].split('/')[-1])
                image = cv2.imread(img_path)
                images.append(image)
                measurements.append(float(line[3]) + correction[i])
                images.append(cv2.flip(image, 1))
                measurements.append(-float(line[3]) - correction[i])
    return np.array(images), np.array(measurements)

def read_data_generator(path, header=True):
    correction = [0, -0.2, 0.2]
    records = []
    with open(os.path.join(path, 'driving_log.csv'), 'r') as csvfile:
        data_reader = csv.reader(csvfile)
        if header:
            print(next(data_reader))
        for line in data_reader:
            records.append(line)
    train_records, valid_records = train_test_split(records, test_size=0.2)

    def generator(samples, batch_size=16):
        num_samples = len(samples)
        while 1:
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                measurements = []
                for sample in batch_samples:
                    image_tmp = []
                    # for i in range(3):
                    #     img_path = os.path.join(path, 'IMG', sample[i].split('/')[-1])
                    #     image = cv2.imread(img_path)
                    #     image_tmp.append(image)
                    # triple_image = np.concatenate(image_tmp, axis=2)

                    img_path = os.path.join(path, 'IMG', sample[0].split('/')[-1])
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    images.append(image)
                    measurements.append(float(sample[3]))

                    # for i in range(3):
                    #     img_path = os.path.join(path, 'IMG', sample[i].split('/')[-1])
                    #     image = cv2.imread(img_path)
                    #     images.append(image)
                    #     measurements.append(float(sample[3]) + correction[i])
                    #     images.append(cv2.flip(image, 1))
                    #     measurements.append(-float(sample[3]) - correction[i])
                yield np.array(images), np.array(measurements)
    return generator(train_records, 128), len(train_records), generator(valid_records, 128), len(valid_records)
