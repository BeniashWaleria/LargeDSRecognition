import insightface
import cv2
import os
import numpy as np
from numpy.linalg import norm


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image


def get_label(filename):
    basename = os.path.basename(filename)
    name = '.'.join(basename.split('.')[:-1])
    return name


def compute_sim(emb1, emb2):
    return np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))


model = insightface.app.FaceAnalysis()
ctx_id = -1
model.prepare(ctx_id=ctx_id, nms=0.4)
directory = r'C:\Users\User\Desktop\Samples'
labels = list()
embeddings = list()
faces = list()
images = list()

################################################################
# get labels and embeddings for known photos

for filename_ in os.listdir(directory):
    img = read_image(directory + "\\" + filename_)
    images.append(img)
    faces_im = model.get(img)
    faces.append(faces_im[0])
    embeddings.append(faces_im[0].embedding)
    labels.append(get_label(filename_))

# cv2.resizeWindow("{}".format(get_label(filename_)), 200, 200)
# cv2.imshow("{}".format(get_label(filename_)), img)
# cv2.waitKey(0)

for i, label in enumerate(labels):
    print(i, label)
##################################################################
# save embeddings and labels to file

np.savez_compressed('database.npz', labels, embeddings)


###################################################################
# get embedding for unknown
test_path = r'C:\Users\User\Desktop\UNKNOWN.jpg'
test_img = read_image(test_path)
# cv2.imshow("UNKNOWN", test_img)
# cv2.waitKey()
faces_test = model.get(test_img)
test_emb = faces_test[0].embedding
np.save('Test.npy', test_emb)

######################################################################
# perform face  recognition

database = np.load('database.npz')['arr_1']
names = np.load('database.npz')['arr_0']
test_sample = np.load('Test.npy')
diff = (database - test_sample)
ssq = np.sum(diff ** 2, axis=1)
index = ssq.argmin()
print("PREDICTION:{}".format(labels[index]))

####################################################################
# similarity
print(compute_sim(test_emb, embeddings[index]))
