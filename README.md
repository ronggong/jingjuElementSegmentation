# jingjuElementSegmentation
Segmentation of jingju note into ascending, desending and vibrato parts

## Usage
1. download Sonic Annotator [here](https://code.soundsoftware.ac.uk/projects/sonic-annotator/files) according to your operating system, then put the binary into sonicAnnotator folder, change name to sonic-annotator.
2. set audioFolder directing to the audio file folder
3. In demo.py, change recordingNamesPredict list into [nameOfaudio]. You can add multiple names, if you have multiple mp3 [name0, name1, ...]

## dependencies
* numpy
* scipy
* scikit-learn==0.16.1
* [essentia](https://github.com/MTG/essentia/releases)
