import os
import pathlib
import argparse

from instalooter.looters import InstaLooter, HashtagLooter, ProfileLooter, PostLooter

import cv2
import numpy as np
import PIL
from PIL import ImageDraw
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from scipy.misc import imread
import matplotlib.pyplot as plt
from ssd.ssd import SSD300
from ssd.ssd_utils import BBoxUtility


def _profile_images(username, destination='.instagram'):
    # if not InstaLooter._logged_in():
    #     InstaLooter._login(username, password)
    #
    # looter = ProfileLooter(username)
    # n = looter.download(destination=destination, new_only=True)

    path = pathlib.Path(destination)
    return list(path.glob('*'))


def _init_keras_session():
    if tf.get_default_session() is None:
        K.set_session(tfu.session.debug_session())


def _load_images(paths, target_size=(300, 300)):
    inputs = []
    images = []
    for img_path in paths:
        img = image.load_img(img_path, target_size=target_size)
        img = image.img_to_array(img)
        images.append(np.asarray(PIL.Image.open(img_path)))
        inputs.append(img.copy())

    inputs = preprocess_input(np.array(inputs))
    return images, inputs


def _load_model(n_classes, model_path='weights_SSD300.hdf5'):
    input_shape=(300, 300, 3)
    model = SSD300(input_shape, num_classes=n_classes)
    model.load_weights(model_path, by_name=True)
    return model


def _main():
    parser = argparse.ArgumentParser(description='instagram post describer')
    parser.add_argument('--login.user', dest='username',
                        help='instagram username')
    parser.add_argument('--login.password', dest='password',
                        help='instagram password')
    # parser.add_argument('--post_id', dest='post_id',
    #                     help='post id (click post then see after the url "https://www.instagram.com/p/")')

    args = parser.parse_args()
    username, password = args.username, args.password
    image_paths = _profile_images(username)
    images, inputs = _load_images(image_paths)

    voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
    NUM_CLASSES = len(voc_classes) + 1
    bbox_util = BBoxUtility(NUM_CLASSES)
    model = _load_model(NUM_CLASSES)
    model.summary()

    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        for j in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[j] * img.shape[1]))
            ymin = int(round(top_ymin[j] * img.shape[0]))
            xmax = int(round(top_xmax[j] * img.shape[1]))
            ymax = int(round(top_ymax[j] * img.shape[0]))
            score = top_conf[j]
            label = int(top_label_indices[j])
            label_name = voc_classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            color = (int(255*color[0]), int(255*color[1]), int(255*color[2]))
            # draw rectangle
            image = PIL.Image.fromarray(np.uint8(img))
            draw = ImageDraw.Draw(image)
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color)
            draw.text((xmin, ymin), display_txt)
            image.save('{}.png'.format(i))

if __name__ == '__main__':
    _main()
