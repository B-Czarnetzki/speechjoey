# coding: utf-8
"""
Data module
"""
import sys
import random
import os
import os.path
import librosa
import torch
import sklearn
import math
import numpy as np
import warnings

from typing import Optional

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary


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
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    if data_cfg["audio"] == "src":
        audio_lang = src_lang
    else:
        audio_lang = trg_lang
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg.get("max_sent_length", sys.maxsize)
    max_audio_length = data_cfg.get("max_audio_length", sys.maxsize)
    number = cfg["model"]["encoder"]["embeddings"]["embedding_dim"]
    assert number <= 80,\
        "The number of used audio features could not be higher than the number of Mel bands. Change the encoder's embedding_dim."
    check_ratio = data_cfg.get("input_length_ratio", sys.maxsize)
    audio_features = data_cfg["audio_features_level"]
    htk = data_cfg["use_htk"]
    scale = data_cfg.get("scale", None)

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

    train_data = AudioDataset(path=train_path, text_ext="." + audio_lang,
                              audio_ext=".txt", sfield=src_field, tfield=trg_field,
                              num=number, char_level=char, train=True,
                              check=check_ratio, audio_level=audio_features, htk=htk,
                              scale=scale, filter_pred=lambda x:
                              len(vars(x)['src']) <= max_audio_length
                              and len(vars(x)['trg']) <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    trg_vocab_file = data_cfg.get(audio_lang + "_vocab", None)
    src_vocab_file = None
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq, max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)
    src_vocab = build_vocab(field="src", min_freq=src_min_freq, max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    # src_vocab = trg_vocab
    dev_data = AudioDataset(path=dev_path, text_ext="." + audio_lang, audio_ext=".txt",
                            sfield=src_field, tfield=trg_field, num=number,
                            char_level=char, train=False, check=check_ratio,
                            audio_level=audio_features, htk=htk, scale=scale)
    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + audio_lang):
            test_data = AudioDataset(path=test_path, text_ext="." + audio_lang,
                                     audio_ext=".txt", sfield=src_field, tfield=trg_field, num=number,
                                     char_level=char, train=False, check=check_ratio,
                                     audio_level=audio_features, htk=htk, scale=scale)
        else:
            # no target is given -> create dataset from src only
            test_data = MonoAudioDataset(path=test_path, audio_ext=".txt",
                                         field=src_field, num=number, char_level=char)
    trg_field.vocab = trg_vocab
    src_field.vocab = src_vocab

    return train_data, dev_data, test_data, src_vocab, trg_vocab


class AudioDataset(TranslationDataset):
    """Defines a dataset for speech recognition/translation."""

    def __init__(self, path: str, text_ext: str, audio_ext: str, sfield: Field, tfield: Field,
                 num: int, char_level: bool, train: bool, check: int, audio_level: str, htk: bool,
                 scale: str, **kwargs) -> None:
        """Create an AudioDataset given path and fields.

            :param path: Prefix of path to the data files
            :param text_ext: Containing the extension to path for text file
            :param audio_ext: Containing the extension to path for audio file
            :param fields: Containing the fields that will be used for text data
            :param num: Containing the number of features to extract (= dimension of source embeddings)
            :param char_level: Containing the indicator for char level
            :param train: Containing the indicator for training set
            :param check: Containing the length ratio as a filter for training set
            :param audio_level: Containing the extraction level of audio features extension
            :param htk: Containing the indicator for mel filters generation
            :param scale: Containing the indicator for audio features scaling
            :param kwargs: Passed to the constructor of data.Dataset.
        """
        audio_field = data.RawField()
        all_fields = [('trg', tfield), ('mfcc', audio_field),
                      ('src', sfield), ('conv', sfield)]

        text_path = os.path.expanduser(path + text_ext)
        audio_path = os.path.expanduser(path + audio_ext)
        examples = []
        if train:
            maxi = 1
            mini = 10
            summa = 0
            count = 0
            log_path = os.path.expanduser(path + '_length_statistics')
            length_info = open(log_path, 'a')

        if len(open(text_path).read().splitlines()) != len(open(audio_path).read().splitlines()):
            raise IndexError('The size of the text and audio dataset differs.')
        else:
            with open(text_path) as text_file, open(audio_path) as audio_file:
                for text_line, audio_line in zip(text_file, audio_file):
                    text_line = text_line.strip()
                    audio_line = audio_line.strip()
                    if text_line != '' and audio_line != '' and os.path.getsize(audio_line) > 44:
                        y, sr = librosa.load(audio_line, sr=None)
                        # overwrite default values for the window width of 25 ms and stride of 10 ms (for sr = 16kHz)
                        # (n_fft : length of the FFT window, hop_length : number of samples between successive frames)
                        # default values: n_fft=2048, hop_length=512, n_mels=128, htk=False
                        # features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num)
                        # check which audio features should be extracted, default are mfccs
                        if audio_level == "mel_fb":
                            features = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=int(
                                sr / 40), hop_length=int(sr / 100), n_mels=num, htk=htk)
                        elif audio_level == "mfcc":
                            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num, n_fft=int(
                                sr / 40), hop_length=int(sr / 100), n_mels=80, htk=htk)
                        elif audio_level == "mfcc_berard_et_al":
                            features_orig = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=math.floor(num / 3), n_fft=int(sr / 25),
                                                                 hop_length=int(sr / 100), n_mels=80, htk=htk)
                            S, phase = librosa.magphase(librosa.stft(
                                y, n_fft=int(sr / 25), hop_length=int(sr / 100)))
                            rms = librosa.feature.rms(S=S)

                            features_delta_1 = librosa.feature.delta(
                                features_orig, order=1)
                            features_delta_2 = librosa.feature.delta(
                                features_orig, order=2)

                            features = np.concatenate(
                                (features_orig, features_delta_1, features_delta_2, rms), axis=0)

                        featuresT = features.T
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
                            # generate a line with <unk> of given size
                            audio_dummy = "a" * (featuresT.shape[0])
                            conv_dummy = "a" * \
                                int(round(
                                    round(featuresT.shape[0] / 2) / 2))
                        else:
                            # generate a line with <unk> of given size
                            audio_dummy = "a " * (featuresT.shape[0])
                            conv_dummy = "a " * \
                                int(round(
                                    round(featuresT.shape[0] / 2) / 2))
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
                    else:
                        warnings.warn(
                            'There is an empty text line or audio file.')
                        print("Check the text line: ", text_line,
                              " or audio file: ", audio_line)
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

    def __init__(self, path: str, audio_ext: str, field: Field, num: int, char_level: bool, **kwargs) -> None:
        """
        Create a MonoAudioDataset (=only sources) given path.

            :param path: Prefix of path to the data file
            :param audio_ext: Containing the extension to path for audio file
            :param field: Containing the field for dummy audio data
            :param num: Containing the number of mfccs to extract (= dimension of source embeddings)
            :param char_level: Containing the indicator for char level
            :param kwargs: Passed to the constructor of data.Dataset.
        """
        audio_field = data.RawField()
        fields = [('mfcc', audio_field), ('src', field), ('conv', field)]
        audio_path = os.path.expanduser(path + audio_ext)
        examples = []

        with open(audio_path) as audio_file:
            for audio_line in audio_file:
                audio_line = audio_line.strip()
                if audio_line != '' and os.path.getsize(audio_line) > 44:
                    y, sr = librosa.load(audio_line, sr=None)
                    if audio_level == "mfcc":
                        features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num, n_fft=int(
                            sr / 40), hop_length=int(sr / 100), n_mels=80)
                    elif audio_level == "mfcc_berard_et_al":
                        features_orig = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=math.floor(num / 3), n_fft=int(
                            sr / 25),  hop_length=int(sr / 100), n_mels=80, htk=htk)
                        S, phase = librosa.magphase(librosa.stft(
                            y, n_fft=int(sr / 25), hop_length=int(sr / 100)))
                        rms = librosa.feature.rms(S=S)

                        features_delta_1 = librosa.feature.delta(
                            features_orig, order=1)
                        features_delta_2 = librosa.feature.delta(
                            features_orig, order=2)

                        features = np.concatenate(
                            (features_orig, features_delta_1, features_delta_2, rms), axis=0)

                    featuresT = features.T
                    # normalize coefficients column-wise for each example
                    featuresNorm = librosa.util.normalize(featuresT) * 0.01
                    featureS = torch.Tensor(featuresNorm)
                    if char_level:
                        # generate a line with <unk> of given size
                        audio_dummy = "a" * (featuresT.shape[0])
                        conv_dummy = "a" * \
                            int(round(round(featuresT.shape[0] / 2) / 2))
                    else:
                        # generate a line with <unk> of given size
                        audio_dummy = "a " * (featuresT.shape[0])
                        conv_dummy = "a " * \
                            int(round(round(featuresT.shape[0] / 2) / 2))
                    examples.append(data.Example.fromlist(
                        [featureS, audio_dummy, conv_dummy], fields))
        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    def __len__(self):
        return len(self.examples)
