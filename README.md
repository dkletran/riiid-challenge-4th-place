Below you can find an outline of how to reproduce my solution for the "Riiid! Answer Correctness Prediction" competition.

# CODE CONTENTS

- data_preparation/: contains python scripts for data preprocessing:
	1) **data_prepare.py** : first step of data preprocessing (compute additional features, store maps of indices (in json format) which will be used to build the model's embedding layers.
	2) **train_valid_split.py** : train - valid split for local validation (2.5% of the whole training set for validation)
	3) **tfrecords_create.py** : script to create tfrecords files for training on Colab TPU
	4) **data_map_create.py** : script to store the latest activities (in a window of size 512) of each student in the whole training set. The output object ( a map of numpy arrays) will be used in the submission kernel.  

- modeling_training/: codes and data for model configuration and training 
- submission/:
	- **riiid-model-submission-4th-solution.ipynb** : Kaggle Notebook for the final submission		
# HARDWARE
-	Data preparation : Google Colab Pro Notebook with High Memory mode (about 25Gb RAM)
-	Model training : 
	-	Google Colab Pro Notebook with TPU
	-	GCS bucket (public access) to store tfrecords files for training
-	Inference (submission) : Kaggle GPU Notebook

# SOFTWARE 
Python packages are detailed separately in `requirements.txt` in each of the folders data_preparation/ and modeling_training/):
- Data preparation: all requirements are already met  in Google Colab Notebook.
- Model training: 2 packages needed to be installed/downgraded to the right version Google Colab Notebook: 

		pip install -q transformers==3.5.1
		pip install -q tensorflow-addons==0.11.2
		
- Model Inference  (Kaggle Submission Notebook): all requirements are already met.
# Steps to produce the solution
- Data preparation:
	1) Copy raw data to the working directory. All raw files are from the competition website https://www.kaggle.com/c/riiid-test-answer-prediction, except for the train data (*riiid_train.pkl.gzip*) which is in pickle pandas format (to be read faster) and  can be found here https://www.kaggle.com/rohanrao/tutorial-on-reading-large-datasets/data

	2) Execute the scripts
	
			python data_prepare.py
			python train_valid_split.py
			python tfrecords_create.py
			python data_map_create.py
			
	All output files (and intermediate result files) should be found in the working directory.
	Below are expected output files from the preprocessing scripts that need to be stored for model building, training and inference (submission):
	
	-	**encoded_content_id_map.json** :  output of **data_prepare.py**, this is the mapping between the encoded content indices and (content_id, content_id_type) 			on the raw data. Needed in the submission kernel.
	-	**encoded_content_map_v2.json** :  output of **data_prepare.py**. This map stores indices of metadata features of questions and lectures. This object is 			needed to build the content embedding layer of the model.
	-	**tfrecords/\*/\*** :  output of **tfrecords_create.py**,  tfrecords files for training on TPU. Three sets of files are created : for train, for valid (from 			the train/valid split step) and for the whole train data.
	-	**data_map.pickle** : zipped output of **data_map_create.py**, needed in the submission kernel.
		
	3) Copy tfrecords files to a GCS bucket for training:

			gsutil -m cp -r tfrecords/* YOUR_GCS_DATA_PATH
		
	4) Store the following output files for model training and inference : *encoded_content_id_map.json, encoded_content_map_v2.json, data_map.pickle*
- Model training:
	1) Copy all python codes from modeling_training/ to the working directory
	2) Copy the file encoded_content_map_v2.json to the working directory
	3) Configure the GCS path of the train and valid data in the *training_config.py*

			TRAIN_DATA_PATH = 'YOUR_GCS_DATA_PATH/whole-train/*.tfrecords'
			VALID_DATA_PATH = None
		or

			TRAIN_DATA_PATH = 'YOUR_GCS_DATA_PATH/train/*.tfrecords'
			VALID_DATA_PATH = 'YOUR_GCS_DATA_PATH/valid/*.tfrecords'
		to train on splitted training set (with validation local training score).
		
	4) Launch the training script:
		
			python training.py
	Output files (*weights.h5* and *training.log*) should be found in the working directory.
	
	5) Store *weights.h5* for the inference step. 
- Inference (Kaggle Submission Notebook): Code source for inference notebook is submission/**riiid-model-submission-4th-solution.ipynb**.  The following files must be put in the input data of the submission notebook:  *modeling.py, model_config.py, encoded_content_map_v2.json, encoded_content_id_map.json, data_map.pickle*
	
# Links
- Here is a Colab Notebook for data preparation https://colab.research.google.com/drive/1WxoPMzmYywMOfqkM3xiRLqCChDWGL1T5?usp=sharing
- Here is a Colab Notebook for model training https://colab.research.google.com/drive/1VhtEkTULq7SnMcl8WIbAToXGBRLEGXir?usp=sharing
- Kaggle Submission Notebook https://www.kaggle.com/letranduckinh/riiid-model-submission-4th-place-public-version
