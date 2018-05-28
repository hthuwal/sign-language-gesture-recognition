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

## Training

**1. Data Folder**
  
  Create a Folder named **test** in the project root directory. It should contain folders corresponding to each cateogry, each folder containing corresponding videos.
  For example:

    ```
    test
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



**2. Extracting frames from all videos.**
   
   - Run the following commnd to extract frames all videos. This wll create a folder named **majorData** and extract frames to that folder.

     ```shell
     python "Video To Frame.py"
     ```
   
   - Note: Before running the script, make sure that the script is dumping data into file `data/labeled-frames-1.pkl` (Probably line 72). You will need to create a folder named **data** in the root directory.
       
   - The code involves some hand segmentation (based on the data we used) for each frame. (You can remove that code if you are working on some other data set)


**3. Retrain the inception v3 model.**

   - Download the [retrain.py](https://raw.githubusercontent.com/tensorflow/hub/r0.1/examples/image_retraining/retrain.py).
      Note: This link may change in the future. Please refer [Tensorflow retrain tutorial](https://www.tensorflow.org/tutorials/image_retraining#training_on_flowers)
   - Run the following command to retrain the inception model.
      
      ```shell
      python retrain.py --bottleneck_dir=bottlenecks --model_dir=inception --summaries_dir=training_summaries/long --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=majorData
      ```

   This will create two file `retrained_labels.txt` and `retrained_graph.pb`

   For more information about the above command refer [here](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3).


**4. Representng each video as sequence of prediction (instead of sequence of frames)**

   - Method 1 : Prediction -> The output of the final layer (i.e a list of probabilities)

      ```shell
      python make_prediction.py
      ```

     This will create a file `data/predicted-frames-1.pkl` that will be used by RNN.
     
     **Note**: Make sure before exeuting above command, variable `batch` in script `make_prediction.py` is set to 1.

   - Method 2 : Prediction -> The output of the pool layer just before the output

      ```shell
      python make_prediction_pool.py
      ```

      This will create a file `data/cnn-features-frames-1.pkl` that will be used by RNN.

      **Note**: Make sure before exeuting above command, variable `batches=['1']` in script `make_prediction_pool.py`.

**5. Train the RNN.**

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