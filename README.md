# Self-supervised speech representations

This is a PyTorch implementation for training learned frame-level speech representations. The implementations for the following types of representations are included:
- APC (Autoregressive Predictive Coding) [1]
- CAPC (Correspondence Autoregressive Predictive Coding) [2]
- CPC (Contrastive Predictive Coding) [3]
- Frame AE (Frame-level Autoencoder representations) [4]
- Frame CAE (Frame-level Correspondence Autoencoder representations) [4]

For details on the implementations of these models and the experiments that were conducted using these representations to train acoustic word embedding models, please see [4] and [5].

## Setup

### Data
The datasets for English and Xitsonga that was used for this implementation has to be downloaded separately. Please go to [ kamperh /
recipe_bucktsong_awe ](https://github.com/kamperh/recipe_bucktsong_awe) for details on where to download the datasets and how to extract the features.

### Docker
You can run this code inside a docker container. Build your image from Dockerfile.gpu or Dockerfile.cpu if you are using a GPU or not, respectively.

`docker build -f docker/<DOCKER FILE NAME> -t <IMAGE NAME>`

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
`python learn_speech_reps.py cpc all config/config.json <CHECKPOINT SAVE DIRECTORY> <PATH TO SAVE EXTRACTED REPRESENATIONS>`

Note: remember to set the PYTHONPATH to the src folder.

## References
[1] Y.-A. Chung, W.-N. Hsu, H. Tang, and J. R. Glass, “An unsupervised autoregressive model for speech representation learning,” in Proc. Interspeech, 2019.

[2] L. van Staden, “Improving Unsupervised Acoustic Word Embeddings Using Segment- and Frame-Level Information.” Masters thesis, Stellenbosch University, 2021.

[3] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[4] H. Kamper, M. Elsner, A. Jansen, and S. Goldwater, “Unsupervised neural network based feature extraction using weak top-down constraints,” in Proc. ICASSP, 2015

[5] [L. van Staden and H. Kamper,  "A comparison of self-supervised speech representations as input features for unsupervised acoustic word embeddings" in proc. SLT, 2021](https://arxiv.org/abs/2012.07387)
