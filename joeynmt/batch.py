import numpy as np
import torch
import torch.nn as nn

# coding: utf-8

"""
Implementation of a mini-batch.
"""


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch, pad_index, use_cuda=False):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_lengths = torch_batch.src
        self.src_mask = (self.src != pad_index).unsqueeze(-2)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None
        self.use_cuda = use_cuda

        if hasattr(torch_batch, "conv"):
            self.conv, self.conv_lengths = torch_batch.conv
            self.conv_mask = (self.conv != pad_index).unsqueeze(-2)

        if hasattr(torch_batch, "mfcc"):
            self.mfcc = torch_batch.mfcc
            max_tensor = max(self.mfcc, key=lambda x: x.shape[0])
            max_dim = max_tensor.shape[0]
            print(max_dim)
            padded_mfcc = []
            for x in self.mfcc:
                m = nn.ZeroPad2d((0, 0, 0, max_dim - x.shape[0]))
                current_ten = m(x)
                padded_mfcc.append(current_ten)
                #padded_mfcc.append(current_ten.permute(1, 0, 2))

            for ten in padded_mfcc:
                print(ten.size())
            self.mfcc = torch.stack(padded_mfcc)

        if hasattr(torch_batch, "trg"):
            trg, trg_lengths = torch_batch.trg
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]
            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]
            # we exclude the padded areas from the loss computation
            self.trg_mask = (self.trg != pad_index)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()
            self.trg = self.trg.cuda()
            self.trg_mask = self.trg_mask.cuda()

        if hasattr(self, "mfcc"):
            self.mfcc = self.mfcc.cuda()

        if hasattr(self, "conv"):
            self.conv = self.conv.cuda()
            self.conv_mask = self.conv_mask.cuda()

    def sort_by_src_lengths(self):
        """
        Sort by src length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.src_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src_lengths = self.src_lengths[perm_index]
        sorted_src = self.src[perm_index]
        sorted_src_mask = self.src_mask[perm_index]
        if self.trg_input is not None:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_lengths = self.trg_lengths[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]
        if hasattr(self, "mfcc"):
            sorted_mfcc = self.mfcc[perm_index]
            self.mfcc = sorted_mfcc
        if hasattr(self, "conv"):
            sorted_conv_lengths = self.conv_lengths[perm_index]
            sorted_conv = self.conv[perm_index]
            sorted_conv_mask = self.conv_mask[perm_index]
            self.conv = sorted_conv
            self.conv_lengths = sorted_conv_lengths
            self.conv_mask = sorted_conv_mask

        self.src = sorted_src
        self.src_lengths = sorted_src_lengths
        self.src_mask = sorted_src_mask

        if self.trg_input is not None:
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg_lengths = sorted_trg_lengths
            self.trg = sorted_trg

        if self.use_cuda:
            self._make_cuda()

        return rev_index
