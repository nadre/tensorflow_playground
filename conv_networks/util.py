import numpy as np
import matplotlib.pyplot as plt
import os, shutil

from tensorflow.examples.tutorials.mnist import input_data

# Utility Functions
def get_mnist():
    return input_data.read_data_sets('../data/MNIST_data', one_hot=True)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def plot_log_data(log_data):
    plt.ylim(-5,110)

    plt.plot(log_data['validation_acc'], color="red", label='Validation Acc')
    plt.plot(log_data['mini_batch_acc'], color="blue", label='Mini Batch Acc')
    plt.plot(log_data['mini_batch_loss'], color="green", label='Mini Batch Loss (Scaled)')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def delete_folder_content(path_to_folder):
    for the_file in os.listdir(path_to_folder):
        file_path = os.path.join(path_to_folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
