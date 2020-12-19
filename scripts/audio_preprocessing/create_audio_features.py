
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script is adapted from Alexander berard seq2seq feature extract script:
# https://github.com/alex-berard/seq2seq/blob/master/scripts/speech/extract.py

from __future__ import division
import argparse
import numpy as np
import yaafelib
import tarfile
import tempfile
import os
from collections import Counter
import math
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('inpath',
                    help='Path to folder, that contains all the wav files')
parser.add_argument(
    'outpath', help='Output path to folder, to save features to')
parser.add_argument('logpath', help="Path to save log files to")
parser.add_argument('n_cpus', help='Number of cpus for multiprocessing')
parser.add_argument('--derivatives', action='store_true')


args = parser.parse_args()

parameters = dict(
    step_size=160,  # corresponds to 10 ms (at 16 kHz)
    block_size=640,  # corresponds to 40 ms
    mfcc_coeffs=40,
    # more filters? (needs to be at least mfcc_coeffs+1, because first coeff is ignored)
    mfcc_filters=41
)

# TODO: ensure that all input files use this rate
fp = yaafelib.FeaturePlan(sample_rate=16000)

mfcc_features = 'MFCC MelNbFilters={mfcc_filters} CepsNbCoeffs={mfcc_coeffs} ' \
                'blockSize={block_size} stepSize={step_size}'.format(
                    **parameters)
energy_features = 'Energy blockSize={block_size} stepSize={step_size}'.format(
    **parameters)

fp.addFeature('mfcc: {}'.format(mfcc_features))
if args.derivatives:
    fp.addFeature('mfcc_d1: {} > Derivate DOrder=1'.format(mfcc_features))
    fp.addFeature('mfcc_d2: {} > Derivate DOrder=2'.format(mfcc_features))

fp.addFeature('energy: {}'.format(energy_features))
if args.derivatives:
    fp.addFeature('energy_d1: {} > Derivate DOrder=1'.format(energy_features))
    fp.addFeature('energy_d2: {} > Derivate DOrder=2'.format(energy_features))

if args.derivatives:
    keys = ['mfcc', 'mfcc_d1', 'mfcc_d2', 'energy', 'energy_d1', 'energy_d2']
else:
    keys = ['mfcc', 'energy']

df = fp.getDataFlow()
engine = yaafelib.Engine()
engine.load(df)
afp = yaafelib.AudioFileProcessor()

frame_counter = Counter()

inpath = args.inpath
outpath = args.outpath
logpath = args.logpath

errorlogfile = os.path.join(logpath, "errorlog.txt")
erroraudios = os.path.join(logpath, "erroraudios.txt")
progresslog = os.path.join(logpath, "progress.txt")

if not os.path.exists(outpath):
    os.makedirs(outpath)
if not os.path.exists(logpath):
    os.makedirs(logpath)

with open(errorlogfile, "w") as errorfile:
    errorfile.write(
        "This file is a log for all the errors that occured while creating the MFCC features.\n")
    errorfile.write(
        "See erroraudios.txt for a list of the audios, that caused errors.\n\n")

with open(erroraudios, "w") as collectionfile:
    pass
with open(progresslog, "w") as progressfile:
    pass


audios = []
n_cpus = int(args.n_cpus)

for audio in os.listdir(inpath):
    audios.append(audio)

chunk_size = math.floor(len(audios) / n_cpus)
chunk_size_leftover = len(audios) - chunk_size * n_cpus
last_chunk = chunk_size + chunk_size_leftover

print("Chunk size = {}".format(chunk_size))
print("Last chunk size = {}".format(last_chunk))


commands_1_n = audios[:-last_chunk]
command_n = audios[-last_chunk:]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def do_command(filenames, process_id):
    counter = 0
    for it, file_name in enumerate(filenames):
        counter += 1
        if it % 1000 == 0:
            progress_message = "Process {} done {} out of {}".format(
                process_id, it, len(filenames))
            with open(progresslog, "a+") as progressfile:
                print(progress_message)
                progressfile.write(progress_message + "\n")

        try:
            afp.processFile(engine, os.path.join(inpath, file_name))
            feats = engine.readAllOutputs()
            feats = np.concatenate([feats[k] for k in keys], axis=1)
            frames, dim = feats.shape

            feats = feats.astype(np.float32)

            if frames == 0:
                with open(errorlogfile, "a+") as errorfile:
                    errorfile.write(
                        "{} seems to be an empty audio or not an audio file \n\n".format(file_name))

                with open(erroraudios, "a+") as collectionfile:
                    collectionfile.write(
                        file_name + "\n")
            else:
                np.save(os.path.join(
                    outpath, file_name.replace(".wav", ".npy")), feats)
        except Exception as e:
            with open(errorlogfile, "a+") as errorfile:
                errorfile.write(
                    "Error occured at file: {} \n".format(file_name))
                errorfile.write("Error type: {}\n".format(type(e)))
                errorfile.write("Error message: {}\n\n".format(str(e)))
            with open(erroraudios, "a+") as collectionfile:
                collectionfile.write(
                    file_name + "\n")
        if it == len(filenames) - 1:
            print("Process {} is Done".format(process_id))


processes = []
for i, chunk in enumerate(chunks(commands_1_n, chunk_size)):
    p = multiprocessing.Process(target=do_command, args=[chunk, i])
    processes.append(p)

p = multiprocessing.Process(target=do_command, args=[command_n, i + 1])
processes.append(p)
for i, p in enumerate(processes):
    p.start()
    print("Started process {}".format(i + 1))
print("started all {} processes".format(n_cpus))

for process in processes:
    process.join()

print("\n\nDone with everything")
