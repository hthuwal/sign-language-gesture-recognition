import pickle
import sys
import tensorflow as tf
from tqdm import tqdm
import argparse


def get_labels():
    """Return a list of our trained labels so we can
    test our training accuracy. The file is in the
    format of one label per line, in the same order
    as the predictions are made. The order can change
    between training runs."""
    with open("retrained_labels.txt", 'r') as fin:
        labels = [line.rstrip('\n') for line in fin]
    return labels


def predict_on_frames(frames):
    """Given a list of frames, predict all their classes."""
    # Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as fin:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fin.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        input_layer = sess.graph.get_operations()[0].name + ':0'

        frame_predictions = []

        pbar = tqdm(total=len(frames))
        for i, frame in enumerate(frames):
            label = frame[1]
            frameCount = frame[2]

            # Get the image path.
            image = frame[0]

            # Read in the image_data
            image_data = tf.gfile.FastGFile(image, 'rb').read()

            try:
                predictions = sess.run(
                    softmax_tensor,
                    {input_layer: image_data}
                )
                prediction = predictions[0]

            except KeyboardInterrupt:
                print("You quit with ctrl+c")
                sys.exit()

            except Exception as e:
                print("Error making prediction: %s" % (e))
                print("\nContinuing.")
                continue

            # Save the probability that it's each of our classes.
            frame_predictions.append([prediction, label, frameCount])

            if i > 0 and i % 10 == 0:
                pbar.update(10)

        pbar.close()

        return frame_predictions


def get_accuracy(predictions, labels):
    """After predicting on each batch, check that batch's
    accuracy to make sure things are good to go. This is
    a simple accuracy metric, and so doesn't take confidence
    into account, which would be a better metric to use to
    compare changes in the model."""
    correct = 0
    for frame in predictions:
        # Get the highest confidence class.
        this_prediction = frame[0].tolist()
        # print(this_prediction)
        this_label = frame[1]
        # print(this_label)

        max_value = max(this_prediction)
        max_index = this_prediction.index(max_value)
        predicted_label = labels[max_index]
        # print(predicted_label)

        # Now see if it matches.
        print(predicted_label, this_label)
        if predicted_label.lower() == this_label.lower():
            correct += 1
        print(correct, len(predictions))

    print(correct, len(predictions))
    accuracy = correct / float(len(predictions))
    return accuracy


def main(train=True):
    labels = get_labels()
    # print(labels)
    train_or_test = "train" if train else "test"
    with open('data/labeled-frames-' + train_or_test + '.pkl', 'rb') as fin:
        frames = pickle.load(fin)

    predictions = predict_on_frames(frames)
    
    accuracy = get_accuracy(predictions, labels)
    print("Batch accuracy: %.5f" % accuracy)

    # Save it.
    with open('data/predicted-frames-' + train_or_test + '.pkl', 'wb') as fout:
        pickle.dump(predictions, fout)

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dump Predictions(probability distribution) for each frame')
    parser.add_argument('--test', action='store_true', help='Use labeled frames')
    args = parser.parse_args()
    main(not args.test)
