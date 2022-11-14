# Self-supervised speech representations

This is a PyTorch implementation for training learned frame-level speech representations. It can be used to train models for acoustic word embeddings as described in the publications listed below. 

For details on the experiments of this implementation, please see

[L. van Staden and H. Kamper,  "A comparison of self-supervised speech representations as input features for unsupervised acoustic word embeddings" in proc. SLT, 2021](https://arxiv.org/abs/2012.07387)
Van Staden, Lisa. “Improving Unsupervised Acoustic Word Embeddings Using Segment- and Frame-Level Information.” Masters thesis, Stellenbosch University, 2021.

## Setup

### Data
The datasets for English and Xitsonga that was used for this implementation has to be downloaded separately. Please go to [ kamperh /
recipe_bucktsong_awe ](https://github.com/kamperh/recipe_bucktsong_awe) for details on where to download the datasets and how to extract the features.

### Docker
You can run this code inside a docker container. Build your image from Dockerfile.gpu or Dockerfile.cpu if you are using a GPU or not, respectively.

`docker build -f docker/&lt;DOCKER FILE NAME> -t &lt;IMAGE NAME>`

You'll have to mount the volumes containing your datasets when running the docker image. Update config/data_paths.json accordingly.

### Requirements

If you're not using docker, update config/data_paths.json to point to the paths of the datasets on your machine and install the following:
- Python 3.6 or higher
- torch
- scikit-learn
- numpy

## Run
The model configuration can be edited in config/mode_config.json.

See the below table on the description of the arguments to train/load/evaluate/extract the features.

| Arguments                                               | Description                                                                                                                                                                                                                                 |
|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| representation_type                                     | Choose from the following: <br/> -apc (autoregressive predictive coding) <br/> -cpc (contrastive predictive coding) <br/> -capc (correspondence autoregressive predictive coding) <br/> -frame_cae (frame-level correspondence autoencoder) |
| action (required)                                       | Choose from the following:<br/> -train (trains the model)<br/> -evaluate (runs evaluation on the model) <br/>-extract (saves representations to a specified path) <br/>-all (runs all actions)                                              |
| config_file (required)                                  | Path to the config file e.g.  _config/model_config.json_                                                                                                                                                                                    |
| checkpoint_path (required)                              | Path to the directory where checkpoints will be saved or loaded.                                                                                                                                                                            |
| --language (default: english_full)                      | Choose the language on which the model should be trained.                                                                                                                                                                                   |
| --load-from-epoch (optional)                            | If specified, loads the weights from the given epoch before the model is trained e.g. --load-from-epoch 100                                                                                                                                 |
| --save_path (required if action is all or extract)      | The path to which the npz containing the learned representations should be saved.                                                                                                                                                           |
| --extract-language (default: choice given for language) | Choose the language dataset that should be used to extract the learned representations from te model. This is only applicable if the chosen value for action is all or extract.                                                             |
| --extract-training (optional)                           | If specified, the training dataset will be used to extract representations. This is only applicable if the chosen value for action is all or extract.                                                                                       |
| --extract-validation (optional)                         | If specified, the validation dataset will be used to extract representations. This is only applicable if the chosen value for action is all or extract.                                                                                     |
| --extract-test (optional)                               | If specified, the test dataset will be used to extract representations. This is only applicable if the chosen value for action is all or extract.                                                                                           |

As an example, here is the command to train, evaluate and extract CPC representations:
`python learn_speech_reps.py cpc all config/config.json &lt;CHECKPOINT SAVE DIRECTORY> &lt;PATH TO SAVE EXTRACTED REPRESENATIONS>`

Note: remember to set the PYTHONPATH to the src folder.

For other run options see the list of arguments below.
