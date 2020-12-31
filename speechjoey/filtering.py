# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""
import os
import sys
from typing import List, Optional
from logging import Logger
import numpy as np

import torch
from torchtext.data import Dataset, Field

from speechjoey.helpers import load_config, make_logger,\
    get_latest_checkpoint, load_checkpoint
from speechjoey.model import build_model, Model
from speechjoey.speech_model import build_speech_model, SpeechModel
from speechjoey.batch import Batch
from speechjoey.data import load_data, load_audio_data, make_data_iter
from speechjoey.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from speechjoey.loss import XentLoss

# pylint: disable=too-many-arguments,too-many-locals,no-member


def generate_perplexities_on_data(model: Model, data: Dataset,
                                  logger: Logger,
                                  use_cuda: bool, max_output_length: int,
                                  loss_function: torch.nn.Module = None,
                                  ) \
        -> List[float]:
    """
    Generate a list of perplexities for every data example
    in given data, by validating on them.

    :param model: model module
    :param logger: logger
    :param data: dataset for validation
    :param use_cuda: if True, use CUDA
    :param max_output_length: maximum length for generated hypotheses
    :param loss_function: loss function that computes a scalar loss
        for given inputs and targets

    :return:
        - ppls_list: List of ppls results on data examples,
    """

    valid_iter = make_data_iter(
        dataset=data, batch_size=1, batch_type="sentence",
        shuffle=False, train=False)
    valid_sources_raw = data.src
    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        ppls_list = []
        logger.info("Starting train data validation")
        for i, valid_batch in enumerate(iter(valid_iter)):
            # run as during training to get validation loss (e.g. xent)

            if i % 1000 == 0:
                logger.info("{} sentences done".format(str(i)))

            batch = Batch(valid_batch, pad_index, use_cuda=use_cuda)
            # sort batch now by src length and keep track of order

            # run as during training with teacher forcing
            if loss_function is not None and batch.trg is not None:
                batch_loss = model.get_loss_for_batch(
                    batch, loss_function=loss_function)
                current_loss = batch_loss
                current_ntokens = batch.ntokens
                current_ppl = torch.exp(current_loss / current_ntokens)
                ppls_list.append(float(current_ppl))

        logger.info("Done with all {} sentences".format(i + 1))

    return ppls_list


# pylint: disable-msg=logging-too-many-args
def filter_noise(cfg_file,
                 ckpt: str,
                 output_path: str = None,
                 logger: Logger = None) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = make_logger()

    cfg = load_config(cfg_file)

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError("No checkpoint found in directory {}."
                                    .format(model_dir))
        try:
            step = ckpt.split(model_dir + "/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"

    use_cuda = cfg["training"].get("use_cuda", False)
    max_output_length = cfg["training"].get("max_output_length", None)

    # load the data
    if cfg.get("speech", True):
        train_data, _, _, src_vocab, trg_vocab = load_audio_data(
            cfg=cfg)
    else:
        train_data, _, _, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"])

    data_to_predict = ("train", train_data)

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    if cfg.get("speech", True):
        model = build_speech_model(
            cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    else:
        model = build_model(
            cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    pad_index = model.pad_index
    label_smoothing = 0.0
    loss_function = XentLoss(pad_index=pad_index,
                             smoothing=label_smoothing)

    data_set_name, data_set = data_to_predict

    #pylint: disable=unused-variable
    ppls_list = generate_perplexities_on_data(
        model, data=data_set, max_output_length=max_output_length,
        use_cuda=use_cuda, loss_function=loss_function,
        logger=logger)
    #pylint: enable=unused-variable

    if output_path is None:
        raise ValueError("Output path must be specified")

    else:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        output_path_set = os.path.join(
            output_path, data_set_name + "_perplexities.txt")
        with open(output_path_set, "w") as outfile:
            first_iteration = True
            for ppls in ppls_list:
                if not first_iteration:
                    outfile.write("\n")
                outfile.write(str(ppls))
                first_iteration = False

        logger.info("Perplexities saved to: %s", output_path_set)
