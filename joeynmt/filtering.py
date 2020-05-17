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
import numpy as np

from joeynmt.helpers import bpe_postprocess, load_config, make_logger,\
    get_latest_checkpoint, load_checkpoint, store_attention_plots
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy, \
    wer, cer
from joeynmt.model import build_model, Model
from joeynmt.speech_model import build_speech_model, SpeechModel
from joeynmt.batch import Batch
from joeynmt.data import load_data, load_audio_data, make_data_iter, \
    MonoDataset
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.loss import XentLoss

# pylint: disable=too-many-arguments,too-many-locals,no-member


def validate_on_data(model: Model, data: Dataset,
                     logger: Logger,
                     batch_size: int,
                     use_cuda: bool, max_output_length: int,
                     level: str, eval_metric: Optional[str],
                     loss_function: torch.nn.Module = None,
                     beam_size: int = 1, beam_alpha: int = -1,
                     batch_type: str = "sentence"
                     ) \
        -> (float, float, float, List[str], List[List[str]], List[str],
            List[str], List[List[str]], List[np.array]):
    """
    Generate translations for the given data.
    If `loss_function` is not None and references are given,
    also compute the loss.

    :param model: model module
    :param logger: logger
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param use_cuda: if True, use CUDA
    :param max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param eval_metric: evaluation metric, e.g. "bleu"
    :param loss_function: loss function that computes a scalar loss
        for given inputs and targets
    :param beam_size: beam size for validation.
        If <2 then greedy decoding (default).
    :param beam_alpha: beam search alpha for length penalty,
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)

    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_sources_raw: raw validation sources (before post-processing),
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    if batch_size > 1000 and batch_type == "sentence":
        logger.warning(
            "WARNING: Are you sure you meant to work on huge batches like "
            "this? 'batch_size' is > 1000 for sentence-batching. "
            "Consider decreasing it or switching to"
            " 'eval_batch_type: token'.")
    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size, batch_type=batch_type,
        shuffle=False, train=False)
    valid_sources_raw = data.src
    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        ppls_dict = {}
        for i, valid_batch in enumerate(iter(valid_iter)):
            # run as during training to get validation loss (e.g. xent)

            if i % 1000 == 0:
                logger.info("{} sentences done".format(str(i)))

            batch = Batch(valid_batch, pad_index, use_cuda=use_cuda)

            # print(loss_function)
            # run as during training with teacher forcing
            if loss_function is not None and batch.trg is not None:
                batch_loss = model.get_loss_for_batch(
                    batch, loss_function=loss_function)
                current_loss = batch_loss
                current_ntokens = batch.ntokens
                current_nseqs = batch.nseqs
                # print(current_loss)
                current_ppl = torch.exp(current_loss / current_ntokens)
                ppls_dict[i] = float(current_ppl)

        # evaluate with metric on full dataset
        join_char = " " if level in ["word", "bpe"] else ""
        #valid_sources = [join_char.join(s) for s in data.src]
        valid_references = [join_char.join(t) for t in data.trg]
        #valid_hypotheses = [join_char.join(t) for t in decoded_valid]

        # post-process
        if level == "bpe":
            #valid_sources = [bpe_postprocess(s) for s in valid_sources]
            valid_references = [bpe_postprocess(v)
                                for v in valid_references]
            # valid_hypotheses = [bpe_postprocess(v) for
            #                   v in valid_hypotheses]
        """
        # if references are given, evaluate against them
        if valid_references:
            assert len(valid_hypotheses) == len(valid_references)
            current_valid_score = 0
            if eval_metric.lower() == 'bleu':
                # this version does not use any tokenization
                current_valid_score = bleu(valid_hypotheses, valid_references)
            elif eval_metric.lower() == 'chrf':
                current_valid_score = chrf(valid_hypotheses, valid_references)
            elif eval_metric.lower() == 'token_accuracy':
                current_valid_score = token_accuracy(
                    valid_hypotheses, valid_references, level=level)
            elif eval_metric.lower() == 'sequence_accuracy':
                current_valid_score = sequence_accuracy(
                    valid_hypotheses, valid_references)
            elif eval_metric.lower() == 'wer':
                current_valid_score = wer(valid_hypotheses, valid_references)
            elif eval_metric.lower() == 'cer':
                current_valid_score = cer(valid_hypotheses, valid_references)
        else:
            current_valid_score = -1

    """
    # print(valid_references)
    # print(all_ppls)
    #ppl_dict = {}
    # for k, v in zip(valid_references, all_ppls):
     #   ppl_dict[k] = v

    return ppls_dict


# pylint: disable-msg=logging-too-many-args
def filter_noise(cfg_file,
                 ckpt: str,
                 output_path: str = None,
                 save_attention: bool = False,
                 logger: Logger = None) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param save_attention: whether to save the computed attention weights
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = make_logger()

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

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

    batch_size = cfg["training"].get(
        "eval_batch_size", cfg["training"]["batch_size"])
    batch_type = cfg["training"].get(
        "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # load the data
    if cfg.get("speech", True):
        train_data, dev_data, test_data, src_vocab, trg_vocab = load_audio_data(
            cfg=cfg)
    else:
        train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"])

    data_to_predict = {"train": train_data}

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

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 1)
        beam_alpha = cfg["testing"].get("alpha", -1)
    else:
        beam_size = 1
        beam_alpha = -1

    pad_index = model.pad_index
    label_smoothing = 0.0
    loss_function = XentLoss(pad_index=pad_index,
                             smoothing=label_smoothing)

    for data_set_name, data_set in data_to_predict.items():

        #pylint: disable=unused-variable
        ppls_dict = validate_on_data(
            model, data=data_set, batch_size=batch_size,
            batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric=eval_metric,
            use_cuda=use_cuda, loss_function=loss_function, beam_size=beam_size,
            beam_alpha=beam_alpha, logger=logger)
        #pylint: enable=unused-variable

        # print(ppl_dict)

        aux = [(ppls_dict[key], key) for key in ppls_dict]
        aux.sort()
        aux.reverse()
        ppls_sorted_list = aux
        #np.save("testing_ppls_list.npy", ppls_sorted_list)

        # print(np.load("testing_ppls_list.npy"))
        if output_path is not None:
            output_path_set = os.path.join(
                output_path, data_set_name + "_perplexities.npy")
            np.save(output_path_set, ppls_sorted_list)
            logger.info("Perplexities saved to: %s", output_path_set)
