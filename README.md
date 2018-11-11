# sign-language-gesture-recognition-from-video-sequences
SIGN LANGUAGE GESTURE RECOGNITION FROM VIDEO SEQUENCES  USING RNN AND CNN

The Paper on this work is published [here](https://link.springer.com/chapter/10.1007/978-981-10-7566-7_63) 

Please do cite it if you find this project useful. :)
## DataSet Used
* [Argentinian Sign Language Gestures](http://facundoq.github.io/unlp/lsa64/). The dataset is made available strictly for academic purposes by the owners. Please read the license terms carefully and cite their paper if you plan to use the dataset.

## Requirements
* Install [opencv](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html).
  Note: **pip install opencv-python** does not have video capabilities. So I recommend to build it from source as described above.
* Install tensorflow:
  ```shell
  pip install tensorflow
  ```
* Install tflearn
  ```shell
  pip install tflearn
  ```

## Training and Testing

### 1. Data Folder
  
Create two folders with any name say **train_videos**  and **test_videos** in the project root directory. It should contain folders corresponding to each cateogry, each folder containing corresponding videos.

For example:

```
train_videos
├── Accept
│   ├── 050_003_001.mp4
│   ├── 050_003_002.mp4
│   ├── 050_003_003.mp4
│   └── 050_003_004.mp4
├── Appear
│   ├── 053_003_001.mp4
│   ├── 053_003_002.mp4
│   ├── 053_003_003.mp4
│   └── 053_003_004.mp4
├── Argentina
│   ├── 024_003_001.mp4
│   ├── 024_003_002.mp4
│   ├── 024_003_003.mp4
│   └── 024_003_004.mp4
└── Away
    ├── 013_003_001.mp4
    ├── 013_003_002.mp4
    ├── 013_003_003.mp4
    └── 013_003_004.mp4
```



### 2. Extracting frames

#### Command

- **usage:**

    ```
    video-to-frame.py [-h] gesture_folder target_folder
    ```

    Extract frames from gesture videos.

- **positional arguments:**
    
    ```
    gesture_folder:  Path to folder containing folders of videos of different
                      gestures.
    target_folder:   Path to folder where extracted frames should be kept.
    ```

- **optional arguments:**

    ```
    -h, --help      show the help message and exit
    ```

The code involves some hand segmentation (based on the data we used) for each frame. (You can remove that code if you are working on some other data set)

#### Extracting frames form training videos

```bash
python3 "video-to-frame.py" train_videos train_frames
```
Extract frames from gestures in `train_videos` to `train_frames`.

#### Extracting frames form test videos

```bash
python3 "video-to-frame.py" test_videos test_frames
```
Extract frames from gestures in `test_videos` to `test_frames`.

### 3. Retrain the Inception v3 model.

- Download retrain.py.
   ```shell
   curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py
   ```
  Note: This link may change in the future. Please refer [Tensorflow retrain tutorial](https://www.tensorflow.org/tutorials/image_retraining#training_on_flowers)
- Run the following command to retrain the inception model.
  
    ```shell
    python3 retrain.py --bottleneck_dir=bottlenecks --summaries_dir=training_summaries/long --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=train_frames
    ```

This will create two file `retrained_labels.txt` and `retrained_graph.pb`

For more information about the above command refer [here](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3).


### 4. Intermediate Representation of Videos

#### Command

- **usage:**

    ```
    predict_spatial.py [-h] [--input_layer INPUT_LAYER]    
                       [--output_layer OUTPUT_LAYER] [--test]    
                       [--batch_size BATCH_SIZE]    
                       graph frames_folder
    ```

- **positional arguments:** 

    ```
    - graph                 graph/model to be executed
    - frames_folder         Path to folder containing folders of frames of
                            different gestures.
    ```

- **optional arguments:**

    ```
      -h, --help            show this help message and exit
      --input_layer INPUT_LAYER
                            name of input layer
      --output_layer OUTPUT_LAYER
                            name of output layer
      --test                passed if frames_folder belongs to test_data
      --batch_size BATCH_SIZE
                            batch Size
    ```

#### Approach 1

- Each Video is represented by a sequence of `n` dimensional vectors (probability distribution or output of softmax) one for each frame. Here `n` is the number of classes.

    **On Training Data**

    ```shell
    python3 predict_spatial.py retrained_graph.pb train_frames --batch=100
    ```

    This will create a file `predicted-frames-final_result-train.pkl` that will be used by RNN.

    **On Test Data**

    ```shell
    python3 predict_spatial.py retrained_graph.pb test_frames --batch=100 --test
    ```

    This will create a file `predicted-frames-final_result-test.pkl` that will be used by RNN. 

#### Approach 2

- Each Video represented by a sequence of 2048 dimensional vectors (output of last Pool Layer) one for each frame

    **On Training Data**

    ```shell
    python3 predict_spatial.py retrained_graph.pb train_frames \
    --output_layer="module_apply_default/InceptionV3/Logits/GlobalPool" \
    --batch=100
    ```

    This will create a file `predicted-frames-GlobalPool-train.pkl` that will be used by RNN.

    **On Test Data**

    ```shell
    python3 predict_spatial.py retrained_graph.pb train_frames \
            --output_layer="module_apply_default/InceptionV3/Logits/GlobalPool" \
            --batch=100 \
            --test
    ```

    This will create a file `predicted-frames-GlobalPool-test.pkl` that will be used by RNN.

### 5. Train the RNN.

#### Command

  - The file `rnn_train.py ` containse the code to train one of the RNN models (described in `rnn_utils.py`).
  - Based on method used in previous step, you may need to comment/uncomment two lines in `rnn_train.py` (Line 40, 41)
  - Create a directory named `checkpoints` in the project root folder and run the followinng commnad.

    ```shell
    python rnn_train.py
    ```

## Testing

1. Move your testing data to **test** folder.

2. Run step 2 of training, but this time dump the labels to **data/labeled-frames-2.pkl**.

3. Run step 4 of training, but make sure `batch` is set equal to 2 if you are using method 1. If you are using method 2 make sure ```bateches=['2']```.

4. Test RNN
  - Based on method used in previous step, you may need to coment/uncomment two lines in `rnn_eval.py` (Line 50, 51).
  - Run the following data for evaluating the test data..

    ```shell
    python rnn_eval.py
    ```

Happy Coding! :)