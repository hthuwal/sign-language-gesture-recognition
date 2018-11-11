"""
Given a saved output of predictions or pooled features from our CNN,
train an RNN (LSTM) to examine temporal dependencies.
"""
from rnn_utils import get_network_wide, get_data
import argparse
import tensorflow as tf
import tflearn
import os


def main(filename, frames, batch_size, num_classes, input_length):
def load_labels(label_file):
    label = {}
    count = 0
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label[l.strip()] = count
        count += 1
    return label

    """From the blog post linked above."""
    # Get our data.
    X_train, X_test, y_train, y_test = get_data(
        filename, frames, num_classes, input_length, True)

    # Get sizes.
    print "y_train[0] : ", y_train
    num_classes = len(y_train[0])
    print "Num classes : - ", num_classes
    # print "Y train : ", y_train[0]

    # Get our network.
    # net = get_network_deep(frames, input_length, num_classes)
    net = get_network_wide(frames, input_length, num_classes)

    # Train the model.
    if os.path.exists('checkpoints/rnn.tflearn'):
        print "Model already exists! Loading it"
        model.load('checkpoints/rnn.tflearn')
        print "Model Loaded"
    else:
        model = tflearn.DNN(net, tensorboard_verbose=0)
    
    model.fit(X_train, y_train, validation_set=(X_test, y_test),
              show_metric=True, batch_size=batch_size, snapshot_step=100,
              n_epoch=10)

    # Save it.
    print "Do you wanna save the model and overwrite? y or n"
    x = raw_input()
    if(x == "y"):
        model.save('checkpoints/rnn.tflearn')

if __name__ == '__main__':
    # filename = 'data/cnn-features-frames-1.pkl'
    # input_length = 2048
    filename = 'data/predicted-frames-1.pkl'
    input_length = 64
    frames = 201
    batch_size = 32
    num_classes = 64

    main(filename, frames, batch_size, num_classes, input_length)
