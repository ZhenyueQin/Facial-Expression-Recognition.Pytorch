import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()


class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Contempt']

acc = 72.0

# Compute confusion matrix
# private saliency
# matrix = np.array([[193, 0, 64, 74, 83, 17, 60],
#  [9, 30, 5, 4, 3, 0, 4],
#  [50, 0, 216, 64, 100, 39, 59],
#  [48, 0, 41, 609, 96, 15, 70],
#  [48, 0, 79, 127, 222, 19, 99],
#  [20, 0, 41,  34, 26, 263, 32],
#  [61, 0, 58, 129, 104, 12, 262]])

# public saliency
# matrix = np.array([[158, 1, 65, 62, 81, 18, 82],
#                    [8, 22, 4, 8, 8, 1, 5],
#                    [53, 0, 182, 64, 90, 33, 74],
#                    [42, 0, 41, 609, 91, 18, 94],
#                    [63, 0, 81, 129, 236, 17, 127],
#                    [21, 0, 31, 29, 30, 269, 35],
#                    [62, 1, 46, 123, 116, 12, 247]])

# CK saliency
# matrix = np.array([[41, 47, 0, 8, 14, 10, 0],
#                    [24, 122, 0, 23, 10, 1, 0],
#                    [4, 30, 6, 17, 6, 21, 0],
#                    [0, 29, 0, 184, 1, 2, 0],
#                    [13, 15, 2, 0, 47, 13, 0],
#                    [6, 7, 0, 2, 10, 215, 0],
#                    [6, 32, 2, 19, 1, 0, 0]])

# CK original
# matrix = np.array(
#     [[107, 7, 0, 0, 0, 0, 6],
#      [9, 168, 0, 0, 0, 0, 3],
#      [0, 0, 54, 6, 9, 3, 12],
#      [0, 0, 0, 216, 0, 0, 0],
#      [10, 0, 1, 0, 70, 0, 9],
#      [2, 0, 7, 0, 2, 229, 0],
#      [5, 3, 1, 0, 3, 0, 48]]
# )

# public original
# matrix = np.array(
#     [[298, 4, 38, 18, 72, 7, 30],
#      [13, 38, 0, 0, 5, 0, 0],
#      [38, 1, 280, 6, 105, 26, 40],
#      [16, 0, 13, 790, 21, 18, 37],
#      [50, 0, 63, 23, 428, 7, 82],
#      [6, 0, 29, 17, 13, 344, 6],
#      [30, 2, 26, 61, 111, 5, 372]]
# )

# private original
# matrix = np.array(
#     [[302, 7, 51, 17, 59, 9, 46],
#      [11, 39, 2, 1, 1, 1, 0],
#      [47, 1, 297, 24, 79, 41, 39],
#      [11, 0, 9, 812, 13, 19, 15],
#      [49, 0, 68, 33, 346, 5, 93],
#      [3, 0, 32, 16, 7, 352, 6],
#      [31, 0, 35, 32, 75, 12, 441]]
# )

# ck_saliency_top_left
matrix = np.array(
    [[110, 3, 3, 0, 4, 0, 0, ],
          [8, 168, 0, 0, 3, 0, 1, ],
          [0, 3, 60, 3, 2, 7, 9, ],
          [0, 0, 3, 213, 0, 0, 0, ],
          [10, 0, 0, 0, 68, 6, 6, ],
          [0, 0, 5, 0, 1, 233, 1, ],
          [4, 0, 5, 0, 1, 0, 50]]
)

# fer2013_top_left_public
# matrix = np.array(
#     [[323, 4, 34, 17, 51, 6, 32],
#      [14, 36, 0, 1, 4, 1, 0],
#      [50, 0, 256, 8, 107, 24, 51],
#      [22, 0, 7, 780, 21, 10, 55],
#      [65, 0, 55, 22, 412, 4, 95],
#      [11, 0, 30, 18, 11, 333, 12],
#      [48, 0, 20, 38, 86, 3, 412]]
# )

# # fer2013_top_left private
# matrix = np.array(
#     [[286, 9, 51, 15, 76, 7, 47],
#      [7, 41, 3, 2, 1, 0, 1],
#      [42, 1, 264, 17, 107, 46, 51],
#      [10, 0, 6, 798, 25, 17, 23],
#      [34, 1, 36, 28, 397, 4, 94],
#      [6, 0, 22, 16, 9, 354, 9],
#      [24, 0, 14, 28, 99, 8, 453]]
# )

acc = 91.4
a_tyep = 'ck+'

if a_tyep == 'fer2013_public': 
  class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
  save_path = 'fer2013_public_composition.png'
  plot_title = 'FER2013 Public Test Composition Confusion Matrix (Accuracy: %0.1f%%)' % acc
elif a_tyep == 'fer2013_private': 
  class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
  save_path = 'fer2013_private_composition.png'
  plot_title = 'FER2013 Private Test Composition Confusion Matrix (Accuracy: %0.1f%%)' % acc
elif a_tyep == 'ck+': 
  class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Contempt']
  save_path = 'ck+_composition.png'
  plot_title = 'CK+ Composition Confusion Matrix (Accuracy: %0.1f%%)' % acc

print('confusion matrix: ', matrix)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                      title=plot_title)
plt.savefig(save_path)
plt.close()
