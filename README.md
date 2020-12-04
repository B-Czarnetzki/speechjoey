# &nbsp; ![Speech-Joey](joey-small.png) Speech Joey

## Goal and Purpose
Speech Joey is an extension of [JoeyNMT](https://github.com/joeynmt/joeynmt)
for end-to-end Automoatich Speech Recognition (ASR) and Automatic Speech Translation (AST).
It keeps joeyNMTs functionality while adding the ability to process speech inputs.
It implements an encoder decoder architecture for speech recognition/translation based on this [paper](https://arxiv.org/abs/1802.04200).
See [seq2seq](https://github.com/alex-berard/seq2seq) for an implemantation in tensorflow.


## Features
Speech Joey implements the following features:
- Speech Encoder using linear layers + CNNs + RNN
- Variational Dropout in Encoder RNN [paper](https://arxiv.org/abs/1506.02557)
- Layer Norm in encoder
- bidirectional projection of encoder outputs
- Conditional recurrent decoder
- Attention Types: MLP, Dot, Multi-Head, Bilinear
- Word-, BPE- and character-based input handling
- BLEU, ChrF, WER, CER evaluation
- Beam search with length penalty and greedy decoding
- Customizable initialization
- Attention visualization
- Learning curve plotting

## Coding
In order to keep the code clean and readable, we make use of:
- Style checks: pylint with (mostly) PEP8 conventions, see `.pylintrc`.
- Typing: Every function has documented input types.
- Docstrings: Every function, class and module has docstrings describing their purpose and usage.


## Installation
SpeechJoey is built on [PyTorch](https://pytorch.org/) and [torchtext](https://github.com/pytorch/text) for Python >= 3.5.

1. Clone this repository:
`git clone https://github.com/B-Czarnetzki/speechjoey`
2. Install speechjoey and it's requirements:
`cd speechjoey`
`pip3 install .` (you might want to add `--user` for a local installation).
3. Run the unit tests:
`python3 -m unittest`

**Warning!** When running on *GPU* you need to manually install the suitable PyTorch version for your [CUDA](https://developer.nvidia.com/cuda-zone) version. This is described in the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).


## Usage

Speech Joey has the same basic usage as joeyNMT except for some key differences regarding the input data and configurations.
If you don't have any experience with joeynmt it is highly suggested to look through
it's tutorial first in joeyNMTs [docs](https://joeynmt.readthedocs.io).


### Data Preparation

#### Parallel Data
For training a speech model, you need parallel data.
SpeechJoey exspects two files per dataset.
The source file contains the necessary information for the audio inputs.
E.g train.lst should contain the path to an (.npy) file storing the audio features for each utterance per line.

`user/yourusername/your/path/to/project/features/audio_utterance_01.npy`
`user/yourusername/your/path/to/project/features/audio_utterance_02.npy`

The target file contains the corresponding transcriptions/translations per line. (e.g train.en)

`This is the transcription of the audio_utterance_01`
`This is the transcription of the audio_utterance_02`

#### Audio pre-processing

SpeechJoey does not extract features from audio files itself.
Instead it exspects these features as input saved in .npy files.
You can use your own way/programm of choice to extract your audio features (e.g MFCCs).
Otherwise this [README](https://github.com/B-Czarnetzki/speechjoey/tree/master/scripts/audio_preprocessing) details how it can be done.
It also explains how to create the toy data example that is used by 'configs/speech_small'.


#### Text Pre-processing
Before training a model on it, the transcriptions/translations are most commonly tokenized, true- or lowercased and often punctuation gets removed.
You might also consider normalizing it in some way (52 --> fifty-two, e.g --> for example).

The Moses toolkit provides a set of useful [scripts](https://github.com/moses-smt/mosesdecoder/tree/master/scripts) for this purpose.

Though it is highly recommended to use a character based model, if you want to try a word or sub-word based model,
SpeechJoey supports the byte-pair-encodings (BPE) format by [subword-nmt](https://github.com/rsennrich/subword-nmt).

### Configuration
Experiments are specified in configuration files, in simple [YAML](http://yaml.org/) format. You can find examples in the `configs` directory.
`speech_small.yaml` contains a detailed explanation of configuration options.

Most importantly, the configuration contains the description of the model architecture (e.g. number of hidden units in the encoder RNN),
paths to the training, development and test data, and the training hyperparameters (learning rate, validation frequency etc.).

#### Speech configurations

To properly use the speech architecture of SpeechJoey some configurations must be exact, while others are highly recomomended to use.

##### Essentials

`speech: True`
To put SpeechJoey into speech processing mode (otherwise it has joeyNMT functionality):

The encoder embedding size must correspond to the number of audio features. (e.g 40 MFCCs + 1 Energy = 41 (toy example)).
`encoder:
    embeddings:
        embedding_dim: 41`

Note: The speech architecture doesn't use transformers: `type: transformer` doesn't work.

##### Highly recommended to use
Experiment showed that the use of the following features seems essential but you are able to not use them.

The architecture is supposed to be a character based model:
`data:
    level: "char"`

The use of variational dropout seems necessary for a good model:
`encoder:
    variational_dropout: True`

The speech architecture this implementation is based on uses a conditional recurrent decoder.
`decoder:
    use_conditional_decoder: True`


##### Memory controll
Training with long audio features can take a lot of memory:

You can filter your dataset by audio and sentence length.

`data:
    max_sent_length: 400
    max_audio_length: 1500`
The audio lenght is messured in timesteps (windows).
E.g MFCCs with 10ms hop size --> 1500 windows = 15 secs audio.

U can reduce memory usage at the cost of training time by using the batch_multiplier.
`training:
    batch_multiplier = 4`

Tip: Don't make batch_size to small otherwise you won't benfit from variational_dropout.

### Training

#### Start
For training, run

`python3 -m speechjoey train configs/speech_small.yaml`.

This will train a model on the training data specified in the config (here: `speech_small.yaml`),
validate on validation data,
and store model parameters, vocabularies, validation outputs and a small number of attention plots in the `model_dir` (also specified in config).

Note that pre-processing like tokenization or BPE-ing is not included in training, but has to be done manually before.

Tip: Be careful not to overwrite models, set `overwrite: False` in the model configuration.

#### Validations
The `validations.txt` file in the model directory reports the validation results at every validation point.
Models are saved whenever a new best validation score is reached, in `batch_no.ckpt`, where `batch_no` is the number of batches the model has been trained on so far.
`best.ckpt` links to the checkpoint that has so far achieved the best validation score.


#### Visualization
SpeechJoey uses Tensorboard to visualize training and validation curves and attention matrices during training.
Launch [Tensorboard](https://github.com/tensorflow/tensorboard) with `tensorboard --logdir model_dir/tensorboard` (or `python -m tensorboard.main ...`) and then open the url (default: `localhost:6006`) with a browser.

For a stand-alone plot, run `python3 scripts/plot_validation.py model_dir --plot_values bleu PPL --output_path my_plot.pdf` to plot curves of validation BLEU and PPL.

#### CPU vs. GPU
For training on a GPU, set `use_cuda` in the config file to `True`. This requires the installation of required CUDA libraries.


### Translating/Transcribing

Whatever data you feed the model for translating, make sure it is properly pre-processed, just as you pre-processed the training data, e.g. tokenized and split into subwords (if working with BPEs).

#### 1. Test Set Evaluation
For testing and evaluating on your parallel test/dev set, run

`python3 -m speechjoey test configs/speech_small.yaml --output_path out`.

This will generate translations/transcriptions for validation and test set (as specified in the configuration) in `out.[dev|test]`
with the latest/best model in the `model_dir` (or a specific checkpoint set with `load_model`).
It will also evaluate the outputs with `eval_metric`.
If `--output_path` is not specified, it will not store the translation/transcriptions, and only do the evaluation and print the results.

Note: The translation mode present in joeynmt isn't implemented for SpeechJoey yet.
