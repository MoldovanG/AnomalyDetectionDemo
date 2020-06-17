import os
import random
import sys
from multiprocessing.pool import ThreadPool
from os import path

import cv2
import time

import easygui as easygui
import numpy as np
import pickle
import cvlib as cv
import boto3
import requests
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import normalize

class AutoEncoderModel:
    """
    Class used for creating autoencoders with the needed architecture .

    Attributes
    ----------
    autoencoder = the full autoencoder model, containing both the encoder and the decoder
    encoder = the encoder part of the autoencoder, sharing the weights with the autoencoder
    """

    def __init__(self, name, s3_client):
        self.s3_client = s3_client
        self.autoencoder, self.encoder = self.__generate_autoencoder()
        self.__load_autoencoder(name)

    def __generate_autoencoder(self):
        input_img = Input(shape=(64, 64, 1))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), strides=2, padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        # compiling the models using Adam optimizer and mean squared error as loss
        optimizer = Adam(lr=10 ** -3)
        encoder.compile(optimizer=optimizer, loss='mse')
        autoencoder.compile(optimizer=optimizer, loss='mse')
        return autoencoder, encoder

    def __load_autoencoder(self, name):
        """Method used for loading the weights from the S3 bucket"""
        # set_session(sess)
        result = self.s3_client.download_file("moldovan.newanomalymodels", name + ".hdf5",
                                              "/tmp/" + name + ".hdf5")
        self.autoencoder.load_weights("/tmp/" + name + ".hdf5")
        self.autoencoder._make_predict_function()
        self.encoder._make_predict_function()

    def get_encoded_state(self, image):
        """
        Parameters
        ----------
        images - np.array containing the image that need to be encoded

        Returns
        -------
        np.array containing the encoded images, predicted by the encoder.
        """
        input = np.expand_dims(image, axis=0)
        encodings = self.encoder.predict(input)
        return encodings[0]


class ObjectDetector:
    """
    Class used for detecting objects inside a given image

     Parameters
    ----------
    image = np.array - the image for which we want to extract the detections

    Attributes
    ----------
    net : pretrained-model from cvlib, using yolov3-tiny architecture trained on the coco dataset.
    threshold : int - the threshold for the detections to be considered positive.
    """

    def __init__(self, image):
        self.image = image
        self.threshold = 0.25
        self.bounding_boxes, self.class_IDs, self.scores = cv.detect_common_objects(image, confidence=self.threshold,
                                                                                    model='yolov3-tiny')

    def __get_cropped_detections(self, frame):
        cropped_images = []
        for idx, score in enumerate(self.scores):
            try:
                c1, l1, c2, l2 = self.bounding_boxes[idx]
                image = frame[l1:l2, c1:c2]
                image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                cropped_images.append(image)
            except cv2.error as e:
                continue

        return np.array(cropped_images)

    def get_object_detections(self):
        """
        Method used for cropping the detected objects from the given image and returning the images
        reshaped to (64x64) and converted to grayscale for further processing by the autoencoder.

        Returns
        ----
        np.array of size (NxWixHix1) where :
        N = number of detections.
        Wi = 64
        Hi = 64
        """
        detections = self.__get_cropped_detections(self.image)
        return detections

    def get_detections_and_cropped_sections(self, frame_d3, frame_p3):
        """
        Method that will return the detections for the image allready present in the ObjectDetector, and using the
        existent bounding-boxes, will also cropp the frames given as parameters.
        :param frame_d3: np.array
        :param frame_p3: np.array
        :return: A pair formed of :
                        - np.array containg detected object appearence of the t frame.
                        - np.array containg cropped image of the t-3 frame of the corresponding detected object
                        - np.array containg cropped image of the t+3 frame of the corresponding detected object
        """
        detections = self.__get_cropped_detections(self.image)
        cropped_d3 = self.__get_cropped_detections(frame_d3)
        cropped_p3 = self.__get_cropped_detections(frame_p3)

        return detections, cropped_d3, cropped_p3


class GradientCalculator:

    def __init__(self) -> None:
        super().__init__()

    def calculate_gradient(self, image):
        # Get x-gradient in "sx"
        sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        # Get y-gradient in "sy"
        sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        # Get square root of sum of squares
        sobel = np.hypot(sx, sy)
        sobel = sobel.astype(np.float32)
        sobel = cv2.normalize(src=sobel, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)
        return sobel

    def calculate_gradient_bulk(self, images):
        gradients = []
        for image in images:
            gradient = self.calculate_gradient(image)
            gradients.append(gradient)
        return np.array(gradients)


class FramePredictor:
    threshold = 1.5
    def __init__(self, s3_client) -> None:
        super().__init__()
        self.s3_client = s3_client
        self.autoencoder_images = AutoEncoderModel("image_autoencoder", self.s3_client)
        self.autoencoder_gradients = AutoEncoderModel("gradient_autoencoder", self.s3_client)
        self.num_clusters = 10
        self.ovr_model = self.__load_model()

    def __load_model(self):
        file_name = "model.sav"
        result = self.s3_client.download_file("moldovan.newanomalymodels", file_name,
                                              path.join("/tmp/", file_name))
        model = pickle.load(open(path.join("/tmp/", file_name), 'rb'))
        return model

    def get_inference_score(self,feature_vector):
        scores = self.ovr_model.decision_function([feature_vector])[0]
        return -max(scores)

def prepare_data_for_CNN(array):
    transformed = []
    for i in range(array.shape[0]):
        transformed.append(array[i] / 255)
    return np.array(transformed)

def normalize_features(feature_vectors, l):
    if l ==0 :
        return feature_vectors
    else :
        return normalize(feature_vectors, 'l' + str(l))


def get_feature_vectors_and_bounding_boxes(frame_predictor, frame, frame_d3, frame_p3):
    start_time = time.time()
    object_detector = ObjectDetector(frame)
    cropped_detections, cropped_d3, cropped_p3 = object_detector.get_detections_and_cropped_sections(frame_d3, frame_p3)
    end_time = time.time()
    class_ids = object_detector.class_IDs
    gradient_calculator = GradientCalculator()
    gradients_d3 = prepare_data_for_CNN(gradient_calculator.calculate_gradient_bulk(cropped_d3))
    gradients_p3 = prepare_data_for_CNN(gradient_calculator.calculate_gradient_bulk(cropped_p3))
    cropped_detections = prepare_data_for_CNN(cropped_detections)
    list_of_feature_vectors = []
    for i in range(cropped_detections.shape[0]):
        if class_ids[i] != 'person':
            continue
        apperance_features = frame_predictor.autoencoder_images.get_encoded_state(
            np.resize(cropped_detections[i], (64, 64, 1)))
        motion_features_d3 = frame_predictor.autoencoder_gradients.get_encoded_state(
            np.resize(gradients_d3[i], (64, 64, 1)))
        motion_features_p3 = frame_predictor.autoencoder_gradients.get_encoded_state(
            np.resize(gradients_p3[i], (64, 64, 1)))
        feature_vector = np.concatenate((motion_features_d3.flatten(), apperance_features.flatten(),
                                         motion_features_p3.flatten()))
        list_of_feature_vectors.append(feature_vector)
    return np.array(list_of_feature_vectors), object_detector.bounding_boxes


def push_back(frames, frame):
    for idx in range(len(frames)-1):
        frames[idx] = frames[idx+1]
    frames[len(frames)-1] = frame


def upload_image(argument_tuple):
    path = argument_tuple[0]
    key = argument_tuple[1]
    s3_client = argument_tuple[2]
    s3_client.upload_file(path, 'moldovan.inferenceframes', key)


def show_alert(bounding_box,frame):
    c1, l1, c2, l2 = bounding_box
    image = frame[l1:l2, c1:c2]
    alert_directory = '/../alerts/'
    if not os.path.exists(alert_directory):
        os.mkdir(alert_directory)
    img_path = os.path.join(alert_directory,"alert.jpg")
    cv2.imwrite(img_path, image)
    easygui.boolbox('Alertă, anomalie detectată', image=img_path, choices=['[O]k', '[C]ancel'], default_choice='OK',
                    cancel_choice='Cancel')


def show_boxes(boxes, scores, frame):
    for idx, bounding_box in enumerate(boxes):
        score = scores[idx]
        top_corner = (int(bounding_box[0]), int(bounding_box[1]))
        bottom_corner = (int(bounding_box[2]), int(bounding_box[3]))
        cv2.rectangle(frame, top_corner, bottom_corner, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, str(round(score, 2)), top_corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return False
    return True


