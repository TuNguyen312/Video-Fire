import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randrange
import seaborn as sns
import tensorflow as tf


def show_video_frames(frames, label):
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"{label}")
    for i in range(len(frames)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(frames[i])
        plt.axis("off")
        plt.title(f"Frame {i + 1}")

    plt.tight_layout()
    plt.show()


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2)

    fig.set_size_inches(18.5, 10.5)

    # Plot loss
    ax1.set_title("Loss")
    ax1.plot(history.history["loss"], label="train")
    ax1.plot(history.history["val_loss"], label="val")
    ax1.set_ylabel("Loss")

    # Determine upper bound of y-axis
    max_loss = max(history.history["loss"] + history.history["val_loss"])

    ax1.set_ylim([0, np.ceil(max_loss)])
    ax1.set_xlabel("Epoch")
    ax1.legend(["Train", "Validation"])

    # Plot accuracy
    ax2.set_title("Accuracy")
    ax2.plot(history.history["accuracy"], label="train")
    ax2.plot(history.history["val_accuracy"], label="val")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1])
    ax2.set_xlabel("Epoch")
    ax2.legend(["Train", "Validation"])
    plt.savefig("images/history.jpg")
    plt.clf()


def plot_confusion_matrix(actual, predicted, labels):
    cm = tf.math.confusion_matrix(actual, predicted)
    ax = sns.heatmap(cm, annot=True, fmt="g", annot_kws={"fontsize": 10})
    sns.set(rc={"figure.figsize": (len(labels), len(labels))})
    ax.set_title("Confusion matrix on test set", size=10)
    ax.set_xlabel("Predicted Action", size=10)
    ax.set_ylabel("Actual Action", size=10)
    plt.xticks(rotation=90, size=10)
    plt.yticks(rotation=0, size=10)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.savefig("images/confusion_matrix.jpg")
    plt.clf()
