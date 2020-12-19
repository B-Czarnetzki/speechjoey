# Audio preprocessing

## Create audio features

First you have to install [**yaafe**](https://github.com/Yaafe/Yaafe)

To create the necessary audio features use `create_audio_features.py`
You need a folder containing all your audio files and execute:

`python3 scripts/audio_preprocessing/create_audio_features.py inpath outpath logpath n_cpus`

`Inpath` being the folder containing the audio files.  
`Outpath` being the place you want to save the features to.  
`Logpath` being the place you want to save the logging files of the script to.  
`n_cpus` being the number of cpu cores the script can use for multiprocessing.  

This will by default create MFCC features for every audio in your inpath based on the settings in the script:  
40 MFCCs (created from 41 MFCC filters, first gets ignored) + 1 Energy feature  
Hop size = 10ms, window size = 40ms

If you add the option `--derivatives` it also concatenates the first and second derivative of every MFCC feature.
You can of course change the specifications or even featuretype in the script if you want.

**Warning**: The script exspects audio files with sample rate = 16kHz if you have different sample rates you have to adapt the window and hop size in the script.

The features get saved in .npy files located in your specified Outpath.

The logging files will give you information on the processes progress and also collect all specified audio files that produced some error.
These files might be corrupted, empty or not audios at all.
You can than easily filter them out of your dataset files.

## Create featurepaths file

SpeechJoey exspects a listingfile that contains the path to the features of the audio corresponding to the
parallel transcription/translation.
You either have to create this yourself or if you have a file that contains the paths to the audio files like this:
```
user/yourusername/path/to/your/project/audios/audio_1.wav
user/yourusername/path/to/your/project/audios/audio_2.wav
```
you can use:

`scripts/audio_preprocessing/create_featurpaths.py audio_listfile featurefolder outfilename

`audio_listfile` being your inputfile containing the paths to the audios.  
`featurefolder` being the folder you saved the features to when using `create_audio_features.py`.  
`outfilename` being the desired name of the output file.  

## Toy example
If you want to test if SpeechJoey runs fine for you, you can use `configs/speech_small.yaml`.
But first you'll have to create the proper featurefiles.
Just execute the following commands:

Hint: It is assumed that you are in the SpeechJoey head directory.

```
python3 scripts/audio_preprocessing/create_audio_features.py test/data/toy_audios test/data/toy_features ./toy_logs 1
python3 scripts/audio_preprocessing/create_featurepaths.py test/data/speechtoy/toy_audios.txt test/data/toy_features test/data/speechtoy/toy.lst
python3 -m speechjoey train configs/speech_small.yaml
```

Note: speech_small.yaml uses a toy example.  
      The specified train, dev and test sets are the same.  
      You'll ofcourse need a proper train, dev, test split for your own datasets.
