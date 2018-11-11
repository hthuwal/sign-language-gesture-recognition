from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import os
import pickle
import sys
import tensorflow as tf
from tqdm import tqdm


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
    input_name = "file_reader"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def predict(graph, image_tensor, input_layer, output_layer):
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    with tf.Session(graph=graph) as sess:
        results = sess.run(
            output_operation.outputs[0],
            {input_operation.outputs[0]: image_tensor}
        )
    results = np.squeeze(results)
    return results


def predict_on_frames(frames_folder, model_file, input_layer, output_layer):
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    graph = load_graph(model_file)

    labels_in_dir = os.listdir(frames_folder)
    frames = [each for each in os.walk(frames_folder) if os.path.basename(each[0]) in labels_in_dir]

    predictions = []
    for each in frames:
        label = each[0]
        print("Predicting on frame of %s\n" % (label))
        for frame in tqdm(each[2], ascii=True):
            try:
                image_path = os.path.join(label, frame)
                image_tensor = read_tensor_from_image_file(image_path, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std)
                pred = predict(graph, image_tensor, input_layer, output_layer)
                predictions.append([pred, label])

            except KeyboardInterrupt:
                print("You quit with ctrl+c")
                sys.exit()

            except Exception as e:
                print("Error making prediction: %s" % (e))
                x = input("\nDo You Want to continue on other samples: y/n")
                if x.lower() == 'y':
                    continue
                else:
                    sys.exit()
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", help="graph/model to be executed")
    parser.add_argument("frames_folder", help="'Path to folder containing folders of frames of different gestures.'")
    parser.add_argument("--input_layer", help="name of input layer", default='Placeholder')
    parser.add_argument("--output_layer", help="name of output layer", default='final_result')
    parser.add_argument('--test', action='store_true', help='Use labeled frames')
    args = parser.parse_args()

    model_file = args.graph
    frames_folder = args.frames_folder

    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer
    if args.test:
        train_or_test = "test"
    else:
        train_or_test = "train"

    # reduce tf verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    predictions = predict_on_frames(frames_folder, model_file, input_layer, output_layer)

    out_file = 'predicted-frames-%s-%s.pkl' % (output_layer, train_or_test)
    print("Dumping predictions to: %s" % (out_file))
    with open(out_file, 'wb') as fout:
        pickle.dump(predictions, fout)

    print("Done.")
