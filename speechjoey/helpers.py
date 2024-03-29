# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from logging import Logger
from typing import Callable, Optional, List
import numpy as np

import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from torchtext.data import Dataset
import yaml
from speechjoey.vocabulary import Vocabulary
from speechjoey.plotting import plot_heatmap


class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """


def make_model_dir(model_dir: str, overwrite=False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError(
                "Model directory exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def make_logger(log_file: str = None) -> Logger:
    """
    Create a logger for logging the training/testing process.

    :param log_file: path to file where log is stored as well
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logging.getLogger("").addHandler(sh)
    logger.info("Hello! This is Joey-NMT extended for speech processing =)")
    return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_data_info(train_data: Dataset, valid_data: Dataset, test_data: Dataset,
                  src_vocab: Vocabulary, trg_vocab: Vocabulary,
                  logging_function: Callable[[str], None]) -> None:
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param src_vocab:
    :param trg_vocab:
    :param logging_function:
    """
    logging_function(
        "Data set sizes: \n\ttrain %d,\n\tvalid %d,\n\ttest %d",
        len(train_data), len(valid_data),
        len(test_data) if test_data is not None else 0)

    logging_function("First training example:\n\t[SRC] %s\n\t[TRG] %s",
                     " ".join(vars(train_data[0])['src']),
                     " ".join(vars(train_data[0])['trg']))

    logging_function("First 10 words (src): %s", " ".join(
        '(%d) %s' % (i, t) for i, t in enumerate(src_vocab.itos[:10])))
    logging_function("First 10 words (trg): %s", " ".join(
        '(%d) %s' % (i, t) for i, t in enumerate(trg_vocab.itos[:10])))

    logging_function("Number of Src words (types): %d", len(src_vocab))
    logging_function("Number of Trg words (types): %d", len(trg_vocab))


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def store_attention_plots(attentions: np.array, targets: List[List[str]],
                          sources: List[List[str]],
                          output_prefix: str, indices: List[int],
                          tb_writer: Optional[SummaryWriter] = None,
                          steps: int = 0) -> None:
    """
    Saves attention plots.

    :param attentions: attention scores
    :param targets: list of tokenized targets
    :param sources: list of tokenized sources
    :param output_prefix: prefix for attention plots
    :param indices: indices selected for plotting
    :param tb_writer: Tensorboard summary writer (optional)
    :param steps: current training steps, needed for tb_writer
    :param dpi: resolution for images
    """
    for i in indices:
        if i >= len(sources):
            continue
        plot_file = "{}.{}.pdf".format(output_prefix, i)
        src = sources[i]
        trg = targets[i]
        attention_scores = attentions[i].T
        try:
            fig = plot_heatmap(scores=attention_scores, column_labels=trg,
                               row_labels=src, output_path=plot_file,
                               dpi=100)
            if tb_writer is not None:
                # lower resolution for tensorboard
                fig = plot_heatmap(scores=attention_scores, column_labels=trg,
                                   row_labels=src, output_path=None, dpi=50)
                tb_writer.add_figure("attention/{}.".format(i), fig,
                                     global_step=steps)
        # pylint: disable=bare-except
        except:
            print("Couldn't plot example {}: src len {}, trg len {}, "
                  "attention scores shape {}".format(i, len(src), len(trg),
                                                     attention_scores.shape))
            continue


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


class vdp_LSTM(nn.Module):
    """
    Modified pytorch LSTM to use variational dropout and optional layernormalization
    """

    def __init__(self, embedding_size, hidden_size, num_layers,
                 idrop=0.0, layerdrop=0.0, batch_first=True, bidirectional=True, layer_norm=False):
        """
        Create a new pytorch LSTM that uses variational dropout
        and optionally layernormalization.

        :param embedding_size: size of input embeddings
        :param hidden_size: hidden size of lstm cells
        :param num_layers: number of lstm layers
        :param idrop: dropout probability for lstm input
        :param layerdrop: dropout probability in between lstm layers
        :param batch_first: whether batch size is in the first dimension of the input
        :param bidirectionoal: whether lstm layers are bidirectional
        :param layer_norm: whether to use layernormalization
        """

        super(vdp_LSTM, self).__init__()

        # Modified LockedDropout that support batch first arrangement
        self.lockdrop = LockedDropout(batch_first=batch_first)
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        self.idrop = idrop
        self.layerdrop = layerdrop
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        directions = 2 if bidirectional else 1

        if self.layer_norm:
            self.input_norm = nn.LayerNorm(embedding_size)
            self.inlayer_norm = nn.LayerNorm(
                2 * hidden_size if bidirectional else hidden_size)

        self.rnns = [
            nn.LSTM(embedding_size if l == 0 else hidden_size * directions,
                    hidden_size, num_layers=1, batch_first=batch_first, bidirectional=bidirectional)
            for l in range(num_layers)
        ]

        self.rnns = torch.nn.ModuleList(self.rnns)

    def forward(self, input, conv_length):
        """
        Apply forward step on LSTM input

        :param input: LSTM input
        :param conv_length: length of src inputs after convolutions
            (counting tokens before padding), shape (batch_size)

        :return:
            - raw_output: Packedsequence object containing the raw output data as Packedsequence and the batch sizes
                - raw_output.data: output of shape (max_seq_len, num_directions * hidden_size):
                    tensor containing the output features (h_t) from the last layer of the LSTM, for each t
                - raw_output.batch_sizes: Tensor of integers
                    holding information about the batch size at each sequence step
            - hidden: h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len.
        """
        raw_output = self.lockdrop(input, self.idrop)
        if self.layer_norm:
            raw_output = self.input_norm(raw_output)
        raw_output = pack_padded_sequence(
            raw_output, conv_length, batch_first=True)
        new_hidden = []
        for l, rnn in enumerate(self.rnns):
            raw_output, (n_hidden, n_cell) = rnn(raw_output)
            if l < self.num_layers:
                raw_output, _ = pad_packed_sequence(
                    raw_output, batch_first=True)
                raw_output = self.lockdrop(raw_output, self.layerdrop)
                if self.layer_norm:
                    raw_output = self.inlayer_norm(raw_output)
                raw_output = pack_padded_sequence(
                    raw_output, conv_length, batch_first=True)
            new_hidden.append(n_hidden)
        hidden = torch.cat(new_hidden, 0)
        return raw_output, hidden


class LockedDropout(nn.Module):
    """
    Variational dropout mask
    https://arxiv.org/abs/1506.02557
    tldr: Applies same dropout mask for every example in the batch.
    """

    def __init__(self, batch_first):
        super().__init__()
        self.batch_first = batch_first

    def forward(self, x, dropout=0.0):
        """
        Apply a variational dropout mask to the batch

        :param x: input batch
        :param dropout: dropout probability

        :return: batch with applied variational dropout mask
        """

        if not self.training or not dropout:
            return x
        if self.batch_first:
            m = x.data.new(x.size(0), 1, x.size(2)
                           ).bernoulli_(1 - dropout)
        else:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
