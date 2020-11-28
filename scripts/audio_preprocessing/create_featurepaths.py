import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('input',
                    help='Inputfile containing the paths to the wav files')
parser.add_argument(
    'featurepath', help='Path to folder, containing the feature files (.npy')


args = parser.parse_args()


inputfile = args.input
featurepath = args.featurepath


first_iteration = True
with open(inputfile, "r") as infile, open("featurepaths.lst", "w") as outfile:
    for line in infile.readlines():
        line = line.strip()
        audio_name = line.split("/")[-1]
        feature_file_name = audio_name.replace(".wav", ".npy")
        if not first_iteration:
            outfile.write("\n")
        if featurepath.startswith("/"):
            outfile.write(str(os.path.join(featurepath, feature_file_name)))
        else:
            current_dir = os.getcwd()
            outfile.write(
                str(os.path.join(current_dir, featurepath, feature_file_name)))
        first_iteration = False

print("Featurespathfile created: featurepaths.lst".format())
