import keras
import keras.preprocessing.image
from keras_retinanet.utils import eval
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
import matplotlib.pyplot as plt
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.eval import draw_detections
import cv2
import numpy as np
import time
import glob
import os

import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = ''


def get_session():
    config = tf.ConfigProto(
        # device_count={'GPU': 0}
    )
    return tf.Session(config=config)



def analyse_images(model, val_generator):
    for index in val_generator.size():
        # for index in range(11, 12):
        # load image
        image = val_generator.load_image(index)

        # copy to draw on
        draw = image.copy()
        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = val_generator.preprocess_image(image)
        height, width, channels = image.shape
        font_size = int(height / 1000)
        if font_size < 1:
            font_size = 1
        image, scale = val_generator.resize_image(image)

        start = time.time()
        _, _, detections = get_predictions(model, image)
        print("processing time: ", time.time() - start)

        # compute predicted labels and scores
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

        # correct for image scale
        detections[0, :, :4] /= scale

        # visualize detections
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score < 0.5:
                continue

            b = detections[0, idx, :4].astype(int)
            cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
            caption = "{} {:.3f}".format(val_generator.label_to_name(label), score)
            cv2.putText(draw, caption, (b[0] - 50, b[1]), cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 0, 0), 3)
            cv2.putText(draw, caption, (b[0] - 50, b[1]), cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 255), 2)

        cv2.imwrite('c:/test/' + str(index) + '.png', draw)


def get_predictions(model, image):
    return model.predict_on_batch(np.expand_dims(image, axis=0))


def main1():
    # for file in glob.glob("./snapshots/*_05.h5"):
    file = './snapshots/resnet50_csv_05.h5'
    # file = 'C:/Projects/OLD-keras-retinanet-master/snapshots/resnet50_csv_01.h5'
    map_total = 0

    for i in range(50, 100, 5):
        i = i / 100

        keras.backend.tensorflow_backend.set_session(get_session())
        model = keras.models.load_model(file, custom_objects=custom_objects)

        val_generator = CSVGenerator(
            csv_data_file='c:/MTSD/Updated/test - copy.csv',
            csv_class_file='c:/MTSD/Updated/classes.csv',
            base_dir='c:/MTSD/Updated/detection/',
            image_min_side=1440,
            image_max_side=2560,
            min_size=25
        )
        # analyse_images(val_generator)

        my_eval = eval.evaluate(val_generator, model, score_threshold=0.5, iou_threshold=0.5, save_path='C:/video-out/', ground_truth=False)

        print(my_eval)

        # total_keys = 0
        # total_val = 0
        # for key, value in my_eval.items():
        #     if value > 0:
        #         total_keys = total_keys + 1
        #         total_val = total_val + value
        #
        # print(total_val/total_keys)
        print(sum(my_eval.values())/39)
        map_total = map_total + sum(my_eval.values())/39
        keras.backend.clear_session()
        break

def vid_main():
    generator = CSVGenerator(
        csv_data_file='c:/MTSD/Updated/test - Copy.csv',
        csv_class_file='c:/MTSD/Updated/classes.csv',
        base_dir='c:/MTSD/Updated/detection/',
        image_min_side=750,
        image_max_side=1200,
        max_size=680
    )
    directory = 'c:/videos/images/kuala/'
    score_threshold = 0.5
    max_detections = 100
    save_path = 'c:/video-out/kuala/'
    file = './snapshots/resnet50_csv_05.h5'
    keras.backend.tensorflow_backend.set_session(get_session())
    model = keras.models.load_model(file, custom_objects=custom_objects)

    for subdir in os.listdir(directory):
        decteted_folder = save_path + subdir + '/'
        if not os.path.exists(decteted_folder):
            os.makedirs(decteted_folder)
        for filename in glob.glob(directory + subdir + '/*.png'):
            image = read_image_bgr(filename)
            _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

            # select scores from detections
            scores = detections[0, :, 4:]

            # clip to image shape
            detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
            detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
            detections[:, :, 2] = np.minimum(image.shape[1], detections[:, :, 2])
            detections[:, :, 3] = np.minimum(image.shape[0], detections[:, :, 3])

            # correct boxes for image scale
            # detections[0, :, :4] /= scale

            # select indices which have a score above the threshold
            indices = np.where(detections[0, :, 4:] > score_threshold)

            # select those scores
            scores = scores[indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            # # select detections
            # image_boxes = detections[0, indices[0][scores_sort], :4]
            # image_scores = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
            # image_detections = np.append(image_boxes, image_scores, axis=1)
            # image_predicted_labels = indices[1][scores_sort]

            if decteted_folder is not None:
                draw_detections(image, detections[0, indices[0][scores_sort], :], generator=generator)
                cv2.imwrite(os.path.join(decteted_folder, '{}'.format(os.path.basename(filename))), image)


if __name__ == '__main__':
    main1()
