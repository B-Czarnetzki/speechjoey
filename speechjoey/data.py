# coding: utf-8
"""
Data module
"""
import sys
import random
import os
import os.path
import torch
import sklearn
import math
import numpy as np
import warnings

from typing import Optional

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from speechjoey.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from speechjoey.vocabulary import build_vocab, Vocabulary


def load_data(data_cfg: dict) -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    def tok_fun(s): return list(s) if level == "char" else s.split()

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    train_data = TranslationDataset(path=train_path,
                                    exts=("." + src_lang, "." + trg_lang),
                                    fields=(src_field, trg_field),
                                    filter_pred=lambda x: len(vars(x)['src'])
                                    <= max_sent_length
                                    and len(vars(x)['trg'])
                                    <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)

    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio],
            random_state=random.getstate())
        train_data = keep

    dev_data = TranslationDataset(path=dev_path,
                                  exts=("." + src_lang, "." + trg_lang),
                                  fields=(src_field, trg_field))
    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = TranslationDataset(
                path=test_path, exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field))
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=src_field)
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab
    return train_data, dev_data, test_data, src_vocab, trg_vocab


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)

    return data_iter


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(data.Example.fromlist(
                    [src_line], fields))

        src_file.close()

        super(MonoDataset, self).__init__(examples, fields, **kwargs)


def load_audio_data(cfg: dict) -> (Dataset, Dataset, Optional[Dataset],
                                   Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on text side and audios up to `max_audio_length`.

    :param cfg: configuration dictionary for data
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: copy of trg_vocab
        - trg_vocab: target vocabulary extracted from training data
    """
    # load data from files
    data_cfg = cfg["data"]
    src_lang = data_cfg.get("src", "lst")
    trg_lang = data_cfg["trg"]
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg.get("max_sent_length", sys.maxsize)
    max_audio_length = data_cfg.get("max_audio_length", sys.maxsize)
    number = cfg["model"]["encoder"]["embeddings"]["embedding_dim"]
    check_ratio = data_cfg.get("input_length_ratio", sys.maxsize)
    scale = data_cfg.get("scale", None)

    # option to transpose pre loaded feature dimensions
    transpose_features = data_cfg.get("transpose_features", None)

    # pylint: disable=unnecessary-lambda
    if level == "char":
        def tok_fun(s): return list(s)
        char = True
    else:  # bpe or word, pre-tokenized
        def tok_fun(s): return s.split()
        char = False

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    train_data = AudioDataset(path=train_path, text_ext="." + trg_lang,
                              audio_feature_ext="." + src_lang, sfield=src_field, tfield=trg_field,
                              char_level=char, train=True,
                              check=check_ratio, scale=scale,
                              transpose_features=transpose_features,
                              filter_pred=lambda x:
                              len(vars(x)['src']) <= max_audio_length
                              and len(vars(x)['trg']) <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    trg_vocab_file = data_cfg.get("trg_vocab", None)
    src_vocab_file = None
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq, max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)
    src_vocab = build_vocab(field="src", min_freq=src_min_freq, max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    # src_vocab = trg_vocab
    dev_data = AudioDataset(path=dev_path, text_ext="." + trg_lang, audio_feature_ext="." + src_lang,
                            sfield=src_field, tfield=trg_field,
                            char_level=char, train=False, check=check_ratio,
                            scale=scale, transpose_features=transpose_features)
    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = AudioDataset(path=test_path, text_ext="." + trg_lang,
                                     audio_feature_ext="." + src_lang, sfield=src_field, tfield=trg_field,
                                     char_level=char, train=False, check=check_ratio,
                                     scale=scale, transpose_features=transpose_features)
        else:
            # no target is given -> create dataset from src only
            test_data = MonoAudioDataset(path=test_path, audio_ext="." + src_lang,
                                         field=src_field, char_level=char,
                                         scale=scale)
    trg_field.vocab = trg_vocab
    src_field.vocab = src_vocab

    return train_data, dev_data, test_data, src_vocab, trg_vocab


class AudioDataset(TranslationDataset):
    """Defines a dataset for speech recognition/translation."""

    def __init__(self, path: str, text_ext: str, audio_feature_ext: str, sfield: Field, tfield: Field,
                 char_level: bool, train: bool, check: int,
                 scale: str, transpose_features: bool,  **kwargs) -> None:
        """Create an AudioDataset given path and fields.

            :param path: Prefix of path to the data files
            :param text_ext: Containing the extension to path for text file
            :param audio_feature_ext: Containing the extension to path for feature file
            :param fields: Containing the fields that will be used for text data
            :param char_level: Containing the indicator for char level
            :param train: Containing the indicator for training set
            :param check: Containing the length ratio as a filter for training set
            :param scale: Containing the indicator for audio features scaling
            :param transpose_features: Containing the indicator for transposing the audio features
            :param kwargs: Passed to the constructor of data.Dataset.
        """
        audio_field = data.RawField()
        all_fields = [('trg', tfield), ('mfcc', audio_field),
                      ('src', sfield), ('conv', sfield)]

        text_path = os.path.expanduser(path + text_ext)
        audio_feature_path = os.path.expanduser(path + audio_feature_ext)
        examples = []
        if train:
            maxi = 1
            mini = 10
            summa = 0
            count = 0
            log_path = os.path.expanduser(path + '_length_statistics')
            length_info = open(log_path, 'a')

        if len(open(text_path).read().splitlines()) != len(open(audio_feature_path).read().splitlines()):
            raise IndexError('The size of the text and audio dataset differs.')

        else:
            with open(text_path) as text_file, open(audio_feature_path) as audio_feature_file:
                for text_line, audio_feature_line in zip(text_file, audio_feature_file):
                    text_line = text_line.strip()
                    audio_feature_file = audio_feature_line.strip()
                    if not os.path.exists(audio_feature_file) or not audio_feature_file.endswith(".npy"):
                        raise FileNotFoundError(
                            "No featurefile (.npy) at {}, you have to create/save audio features before you can load them".format(audio_feature_file))
                    features = np.load(audio_feature_file)

                    featuresT = features.T if transpose_features else features

                    if scale == "norm":
                        # normalize coefficients column-wise for each example normalizes (each column by aggregating over the rows)
                        # the input array is scaled to the norm between -1 and 1
                        featuresNorm = librosa.util.normalize(featuresT)
                    elif scale == "mean":
                        featuresT = sklearn.preprocessing.scale(
                            featuresT, with_std=False)  # center to the mean
                    elif scale == "unit_var":
                        # component-wise scale to unit variance
                        featuresT = sklearn.preprocessing.scale(
                            featuresT, with_mean=False)
                    elif scale == "all":
                        # center to the mean and component-wise scale to unit variance
                        featuresT = sklearn.preprocessing.scale(featuresT)
                    featureS = torch.Tensor(featuresT)
                    if char_level:
                        # print("FeatureT shape: ", featuresT.shape[0])
                        # generate a line with <unk> of given size
                        audio_dummy = "a" * (featuresT.shape[0])

                        conv_dummy = "a" * \
                            int(math.ceil(
                                math.ceil(featuresT.shape[0] / 2) / 2) - 1)
                    else:
                        # generate a line with <unk> of given size
                        audio_dummy = "a " * (featuresT.shape[0])
                        conv_dummy = "a " * \
                            int(math.ceil(
                                math.ceil(featuresT.shape[0] / 2) / 2) - 1)
                    if train:
                        length_ratio = featuresT.shape[0] // (
                            len(text_line) + 1)
                        if length_ratio < check:
                            examples.append(data.Example.fromlist(
                                [text_line, featureS, audio_dummy, conv_dummy], all_fields))
                        if length_ratio > maxi:
                            maxi = length_ratio
                        if length_ratio < mini:
                            mini = length_ratio
                        summa += length_ratio
                        count += 1
                    else:
                        examples.append(data.Example.fromlist(
                            [text_line, featureS, audio_dummy, conv_dummy], all_fields))

        if train:
            length_info.write('mini={0}, maxi={1}, mean={2}, checked by {3} \n'.format(
                mini, maxi, summa / count, check))
            length_info.close()

        super(TranslationDataset, self).__init__(
            examples, all_fields, **kwargs)

    def __len__(self):
        return len(self.examples)

    def gettext(self, index):
        return self.examples[index].trg

    def getaudio(self, index):
        return self.examples[index].audio


class MonoAudioDataset(TranslationDataset):
    """Defines a dataset for speech recognition/translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, audio_feature_ext: str, field: Field, char_level: bool, **kwargs) -> None:
        """
        Create a MonoAudioDataset (=only sources) given path.

            :param path: Prefix of path to the data file
            :param audio_feature_ext: Containing the extension to path for audio feature file
            :param field: Containing the field for dummy audio data
            :param char_level: Containing the indicator for char level
            :param kwargs: Passed to the constructor of data.Dataset.
        """
        audio_field = data.RawField()
        fields = [('mfcc', audio_field), ('src', field), ('conv', field)]
        audio_feature_path = os.path.expanduser(path + audio_feature_ext)
        examples = []

        with open(audio_feature_path) as audio_feature_file:
            for audio_feature_line in audio_feature_file:
                audio_feature_line = audio_feature_line.strip()
                if not os.path.exists(audio_feature_line) or not audio_feature_line.endswith(".npy"):
                    raise FileNotFoundError(
                        "No featurefile (.npy) at {}, you have to create/save audio features before you can load them".format(audio_feature_file))
                features = np.load(audio_feature_line)

                featuresT = features.T if transpose_features else features

                if scale == "norm":
                    # normalize coefficients column-wise for each example normalizes (each column by aggregating over the rows)
                    # the input array is scaled to the norm between -1 and 1
                    featuresNorm = librosa.util.normalize(featuresT)
                elif scale == "mean":
                    featuresT = sklearn.preprocessing.scale(
                        featuresT, with_std=False)  # center to the mean
                elif scale == "unit_var":
                    # component-wise scale to unit variance
                    featuresT = sklearn.preprocessing.scale(
                        featuresT, with_mean=False)
                elif scale == "all":
                    # center to the mean and component-wise scale to unit variance
                    featuresT = sklearn.preprocessing.scale(featuresT)
                featureS = torch.Tensor(featuresT)
                if char_level:
                    # print("FeatureT shape: ", featuresT.shape[0])
                    # generate a line with <unk> of given size
                    audio_dummy = "a" * (featuresT.shape[0])

                    conv_dummy = "a" * \
                        int(math.ceil(
                            math.ceil(featuresT.shape[0] / 2) / 2) - 1)
                else:
                    # generate a line with <unk> of given size
                    audio_dummy = "a " * (featuresT.shape[0])
                    conv_dummy = "a " * \
                        int(math.ceil(
                            math.ceil(featuresT.shape[0] / 2) / 2) - 1)
                    examples.append(data.Example.fromlist(
                        [featureS, audio_dummy, conv_dummy], fields))
        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    def __len__(self):
        return len(self.examples)
