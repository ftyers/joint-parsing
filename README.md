# Joint syntactic and morphological parser

## General information
This is a joint syntactic and morphological parser. It does two things:
- assigns dependency relation and syntactic head to each token in the input file;
- selects the best morphological analysis for each token, provided that there were multiple.

Details on implementation and theoretical background can be found in the paper in this repository.

In addition to parser usage, this README occasionally explains command-line use and package installation. This is to ensure that less experienced users may also install and use the software successfully. Experienced users are welcome to skip those.

## How to use
### Requirements
This parser runs on Python 3. Check if you have Python 3 by typing the following command in the terminal:
```
$ python3 --version
```
If command is not found, install Python 3 from python.org.

The parser depends on a few Python modules. They are listed in requirements.txt for automatic installation, and are the following:
- numpy
- scipy
- scikit-learn

### Installation
To install the parser, make sure that you have installed all the above requirements. Then clone this repository (or download and unzip the zip file) and navigate to the parser folder.
```
$ git clone https://github.com/Sereni/joint-parsing.git
$ cd joint-parsing
```
### Parsing
To parse a corpus with the joint parser, you need the following:
- an input file in CoNLL-U format
- a parsing model, vectorizer, and feature set
- a tagging model, vectorizer, and feature set

Models, vectorizers and feature sets are created during parser training. This distribution comes with a pre-trained model for Kazakh, which can be found in models/. If you would like to train your own model, see section below.

Provided you have all the things above, run the following command to parse your corpus (don't actually, see below):
```
$ python3 jdp.py INPUT OUTPUT -pm parsing_model -pvec parsing_vectorizer -pf parsing_features -tm tagging_model -tvec tagging_vectorizer -tf tagging_features
```
If the above command makes sense to you, please skip the rest of this section.

This command calls Python 3 and tells it to run a script called jdp.py. Unless you are in the parser folder, you need to specify a full path to this file, which looks similar to /home/your_username/stuff/joint-parsing/parser/jdp.py. Other paths in this command should be full (or valid relative) paths as well.

INPUT and OUTPUT are also paths to files. INPUT is where the input corpus is. OUTPUT is where you would like the parsed corpus to go (it doesn't have to exist, the parser will create file). Remember to check the paths to make sure that your system can find the necessary files.

The rest of the command are options. All of them are required, and the parser will raise an error if any are missing. They are all paths as well. Take a look into the *models* folder, which has a set of these files for Kazakh. You can run the following command to see a brief explanation of each option (-h for help):
```
python3 jdp.py -h
```
### Training the parser
This section is coming sometime. If it has been long and/or you need it, please ping me.

## If stuff does not work
Please raise an issue at https://github.com/Sereni/joint-parsing/issues or submit a pull request if you fixed it yourself (thanks!). This includes the cases when you have trouble running the parser following this readme.

## Credits
The parser is based on code by Ilnar Salimzianov, which is hosted here: https://gitlab.com/selimcan/SDP.
Thanks to @ftyers for helping with architecture and code. For questions and comments raise an issue or write to yekaterina.ageeva at gmail.