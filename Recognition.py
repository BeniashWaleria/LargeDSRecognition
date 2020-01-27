import insightface
import cv2
import os
import numpy as np
from numpy.linalg import norm
from  sklearn.metrics import classification_report
from  sklearn.metrics import accuracy_score

def count_files(basedir):
    dir_names_ = []
    num_files_ = []
    labels_ = []
    for directory in os.listdir(basedir):

        full_dir = os.path.join(basedir, directory)
        dir_files = len([item for item in os.listdir(full_dir) if os.path.isfile(os.path.join(full_dir, item))])
        if dir_files >= 2:
            labels_.append(directory)
            dir_names_.append(full_dir)
            num_files_.append(dir_files)

    return labels_, dir_names_, num_files_


def read_image(path_, array):
    img = cv2.imread(path_, cv2.IMREAD_COLOR)
    array.append(img)
    return img, array


def get_embedding(image_, embeddings, model_):
    faces_ = model_.get(image_)
   # faces_db.append(faces_[0])
    embeddings.append(faces_[0].embedding)
    return  embeddings, faces_[0].embedding


def get_label(filename):
    basename = os.path.basename(filename)
    name = '.'.join(basename.split('.')[:-1])
    return name


def compute_sim(db_matr, test_vect=None):
    distances = norm(db_matr - test_vect, 2, 1)
    angles = np.arccos(db_matr @ test_vect.T / (norm(db_matr, 2, 1) * norm(test_vect, 2))) * 180 / np.pi
    return angles, distances


model = insightface.app.FaceAnalysis()
ctx_id = -1
model.prepare(ctx_id=ctx_id, nms=0.4)
threshold = 75
main_dir = r'C:\Users\User\Desktop\lfw-deepfunneled'
embeddings_base = list()
embeddings_test = list()
faces_base = list()
faces_test = list()
images_base = list()
images_test = list()
unrecognized = list()
################################################################
# extract 2  photos from the folders and create  test and base datasets

labels, dir_names, num_files = count_files(main_dir)

for path in dir_names:
    (_, _, filenames) = next(os.walk(path))
    img_base = read_image(os.path.join(path, filenames[0]), images_base)
    print("read:{}".format(filenames[0]))
    img_test = read_image(os.path.join(path, filenames[1]), images_test)

img_database = zip(labels, images_base)

################################################################
# calculate embeddings for image databasе
print("GETTING TRAINING EMBEDDINGS ")

for i,image in enumerate(images_base):
        print(i)
        get_embedding(image, embeddings_base, model)
##################################################################
# save embeddings and labels to file
print("SAVING   TRAINING EMBEDDINGS ")
np.savez_compressed('Database.npz', labels, embeddings_base)
###################################################################
# get embeddings for unknown and perform face recognition
min_angles = []

predicted_labels = []
print("READING TRAINING EMBEDDINGS ")
database = np.load('Database.npz')['arr_1']
for tested in images_test:
    t_embeddings, test_vect_ = get_embedding(tested, embeddings_test, model)
  #  cv2.imshow("UNKNOWN", tested)
   # cv2.waitKey(0)
    print("COMPUTING ANGLES ")
    angles_, distances_ = compute_sim(database, test_vect_)
    min_angle_ = angles_.min()
    print(min_angle_)
    index = angles_.argmin()
    if min_angle_ < threshold:
        print("ANGLE:{}".format(min_angle_))
        print("PREDICTED:{}".format(labels[index]))
        predicted_labels.append(labels[index])
    else:
        predicted_labels.append("Unknown")
        unrecognized.append(labels[index])

result = classification_report(labels, predicted_labels,labels)

print(result)


accuracy = accuracy_score(labels,predicted_labels)
print(accuracy)
print("UNRECOGNIZED: ")
print(unrecognized)

