
import csv
import fnmatch
import os
import cv2
import numpy as np
import tensorflow as tf
import handshape_feature_extractor
import frameextractor

# global variables
BASE = os.path.dirname(os.path.abspath(__file__))


IMAGE_MODE = cv2.IMREAD_GRAYSCALE
TRAININGDATA = os.path.join(BASE, 'traindata')
TESTINGDATA = os.path.join(BASE, 'test')
TRAINOUT = os.path.join(BASE, 'trainout')
TESTOUT = os.path.join(BASE, 'testout')

# CNN model
hfe = handshape_feature_extractor.HandShapeFeatureExtractor().get_instance()


table = {
    'Num0': 0, 'num0': 0,
    'Num1': 1, 'num1': 1,
    'Num2': 2, 'num2': 2,
    'Num3': 3, 'num3': 3,
    'Num4': 4, 'num4': 4,
    'Num5': 5, 'num5': 5,
    'Num6': 6, 'num6': 6,
    'Num7': 7, 'num7': 7,
    'Num8': 8, 'num8': 8,
    'Num9': 9, 'num9': 9,
    'FanDown': 10, 'decreasefanspeed': 10,'FanSpeedDown':10,
    'FanOff': 11, 'fanoff': 11,
    'FanOn': 12, 'fanon': 12,
    'FanUp': 13, 'increasefanspeed': 13,'FanSpeedUp': 13,
    'LightOff': 14, 'lightoff': 14,
    'LightOn': 15, 'lighton': 15,
    'SetThermo': 16, 'setthermo': 16, 'Sethermostat':16
}


class Gesture:
    """Holds the data pertaining to gesture like the name,video file name, image file name etc"""
    def __init__(self, name=None, video_file=None, image_file=None, feature_score=None,
                 true_label=float('inf'), predicted_label=float('inf')):
        """
        : name -Gesture name
        : video_file -video file name
        : image_file -image file name
        : feature_score -  extract_feature() score
        : true_label - true label for training data
        : predicted_label -predicted label after cosine similarity
        """
        self.name = name
        self.image_file = image_file
        self.feature_score = feature_score
        self.video_file = video_file

        self.true_label = true_label
        self.predicted_label = predicted_label
        if self.video_file:
            self.video_name = self.video_file.replace('_SaiMadhuriMolleti', '').replace('.mp4', '')

    def is_identified(self):

        return self.predicted_label == self.true_label


def features_test():
    """ extracts the features from each testdata video provided  """
    test_gesture = dict()
    for test_count, test_file in enumerate(fnmatch.filter(os.listdir(TESTINGDATA), '*.mp4')):
        te_gesture = feature_gesture(TESTINGDATA, TESTOUT, test_file, test_count)
        test_gesture[te_gesture.video_name] = te_gesture
    return test_gesture


def features_train():
    """ extracts the features from each train data video provided """
    train_gesture = dict()
    for train_count, train_file in enumerate(fnmatch.filter(os.listdir(TRAININGDATA), '*.mp4')):
        tr_gesture = feature_gesture(TRAININGDATA, TRAINOUT, train_file, train_count)
        train_gesture[tr_gesture.video_name] = tr_gesture
    return train_gesture


def get_gesture_name(file_path):
    name = file_path
    if '-' in file_path:
        name = '-'.join(file_path.split('-')[1:])
    elif '_' in file_path:
        name = file_path.split('_')[0]
    return name.replace(".mp4", "")


def feature_gesture(data_dir, out_dir, video_file, count):
    """Extract frame to a file and calculate its feature vector
    """
    video_path = os.path.join(data_dir, video_file)
    frameextractor.frameExtractor(video_path, out_dir, count)
    image_path = os.path.join(out_dir, '{0:05}.png'.format(count+1))

    gesture_name = get_gesture_name(video_file)
    image_file = os.path.basename(image_path)
    image_array = cv2.imread(image_path, IMAGE_MODE)
    vector = hfe.extract_feature(image_array)
    try:
        # get the known true label for train data
        true_label = table[gesture_name]
    except KeyError:
        # guess the expected true label for test data
        true_label = count % 17

    gesture = Gesture(gesture_name, video_file, image_file,
                      vector, true_label)
    return gesture

def print_result(test_gesture, results_matrix):

    correct = np.sum([1 for x in test_gesture if test_gesture[x].is_identified()])
    total = results_matrix.sum()
    percentage = correct/total*100
    print("Matched {0} out of {1} => {2:0.2f} %".format(correct, total, percentage))
    with open(os.path.join(BASE, 'Results.csv'), mode='w') as results_csv:
        results_writer = csv.writer(results_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in sorted(test_gesture, key=lambda x: test_gesture[x].video_name):
            results_writer.writerow([test_gesture[i].predicted_label])


def main():
    test_gesture = features_test()
    train_gesture = features_train()
    print("Test Data","Prediction","True", "Match" )

    # calculate cosine similarity
    results_matrix = np.zeros((17, 17), dtype=int)
    for k in test_gesture:
        test_proba = test_gesture[k].feature_score

        cosine_similarity = float('inf')
        tr_video_file = None
        for key in train_gesture:
            trained = train_gesture[key].feature_score

            cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
            y = cosine_loss(trained, test_proba).numpy()

            if y < cosine_similarity:
                cosine_similarity = y
                test_gesture[k].predicted_label = train_gesture[key].true_label
                tr_video_file = train_gesture[key].video_name

        results_matrix[test_gesture[k].predicted_label][test_gesture[k].true_label] += 1

        print(test_gesture[k].video_name,test_gesture[k].predicted_label,
            test_gesture[k].true_label, test_gesture[k].is_identified())

    print_result(test_gesture, results_matrix)



if __name__ == '__main__':
    main()
