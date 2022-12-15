# Deep Musician

Generating new and unheard musical melodies with a deep neural network that was
trained with existing MIDI files. The network uses a sequence aware Encoder-Decoder
structure that is capable of creating sequences of notes of arbitrary length.
The encoder and decoder each consist of a 2-layer GRU network, whereas the
decoder has an additional classifier.

## Idea

### Creating MIDI Files with a deep neural network

With **MuseNet OpenAI** created a deep neural network that "can generate
4-minute musical compositions with 10 different instruments, and can combine
styles from country to Mozart to the Beatles." [[1]](#1) Behind this project
resides the (philosophical) idea that musical compositions can arise not only from a
particular (abstract) artistic understanding of harmony, rhythm, melody, etc., but also
*solely* from a variety of previous works that are incorporated into the new, unheard
piece as a wealth of experience.

MuseNet is fuelled by "a large-scale **transformer model** trained to predict
the next note(s) in a sequence." [[1]](#1) While this approach using a
transformer model is certainly state of the art for sequential data and produces a truly vibrant and rich musical style, it is very resource intensive.

In contrast, there are also more lightweight approaches that use
a variety of RNN structures (such as LSTM models [[2-6]](#2) or RNN with
Self-Attention [[7]](#7)) and even a CNN [[8]](#8) - which is quite interesting
considering the temporal dimension of music that CNN are not designed to depict.
(For a general overview of the different approaches of generative music models see: [[9]](#9).)

In **this project** I want to implement a sequence aware model, that is capable
of generating musical sequences of arbitrary length. In doing so, I hope to
gain insights into the increasing explanatory power and creative potential of
this type of models.

### Data Structure

#### Representing Music

The basis of music is formed by sequentially played sounds or tones that can
be represented as a complex **waveform**. These individual sounds can be joined
together in any way to form an entire piece of music, which in turn is again a
single waveform that we can play back and listen to in different audio
formats (MP3, FLAC, WAV etc.).

#### MIDI

While this form of representation already depicts a concrete shaping of
the music in the form of an unique audio file, it is also possible to specify
the individual tones of the piece in the form of notes with different
parameters. The advantage here is that the concrete instrumentation is
abstracted from and only the internal **structure of the piece** is considered. The
generally accepted standard for this representation is **MIDI**. By means of MIDI it
is possible to transmit not only the pitch and length of the individual notes,
but also other parameters such as velocity - yet, no concrete waveform is produced.

Thus, due to its abstract nature MIDI offers the possibility to extend the input of the model successively. While initially only monophonic audio tracks with constant dynamics and tone length are used, these parameters are to be successively added to the input to see how the created melodies change.

#### Music genre

Additionally I would also like to discuss the characteristics of different
styles of music. Initially, it is planned to single out only one style of
music, or better said, only one artist: Mozart. As a master of melody Mozart
offers the perfect introduction to the world of beautiful tunes. Subsequently,
I would like to dissolve this restriction and include other artists and epochs
as well.[[10]](#10)

#### Dataset

For my project I will use MIDI data from different sources:

1. [**Meastro**](https://magenta.tensorflow.org/datasets/maestro)(1,291)
2. [**Classical Archives**](https://www.classicalarchives.com>) (4,918)
3. [**Symbolic Music Midi Data V1.1**](https://arxiv.org/pdf/1606.01368.pdf)
4. [**Video Game Music**](https://archive.org/details/video-game-music-90000-midi-files) (92,861)
5. [**Video Game Music Archive**](https://www.vgmusic.com/) (31,581)
6. [**Bitmidi**](https://bitmidi.com/) (5,311)
7. [**The Lakh MIDI Dataset**](https://colinraffel.com/projects/lmd/)(45,129)
8. [**The Magic of MIDI V1**](https://archive.org/details/themagicofmidiv1) (169,454)

The number of total titles are indicated in parenthesis. 1-3 cover pieces from
*classical music*. 4 and 5 contain *video game* music. 6-8 comprise of all
different sorts of music genres. The data acquisition will include a mixture of
simple bulk downloads and web scraping.

### Extensions

This project can be summarised under the type of **bring your own method** as it can be expanded successively on the basis of four axes:

- Monophonic - polyphonic
- Additional midi parameters
- Different music styles
- Complexity of the model: RNN > LSTM > Transformer

The fact that these extensions are largely independent of each other results in
a **modular structure** of the project, in that the individual modules can be
strung together as desired. This allows me to look at the different aspects of
the individual components and evaluate them, but on the other hand does not
give a definite goal of the project or model, rather only a trajectory that
travels along the lines of a generative music model, that tries to enhance its
creative potential. However, this is intentional and is meant to encourage the
project to be pursued and expanded beyond the university levy in order to develop a musical model that can independently generate creative music that is (almost) indistinguishable from human-produced pieces.

## Install and Quick Start

First create a new conda environment with python 3.10 and activate it:

```bash
conda create -n deepmusician python=3.10
conda activate deepmusician
```

Then install this repository as a package, the `-e` flag installs the package in editable mode, so you can make changes to the code and they will be reflected in the package.

```bash
pip install -e .
```

### The directory structure and the architecture of the project

```
📦DeepMusician
 ┣ .circleci
 ┣ 📂data
 ┃ ┣ 📜README.md
 ┃ ┣ 📜dl_bitmidi.py
 ┃ ┗ 📜download.sh
 ┣ deepmusician
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜seq2seq.py
 ┃ ┣ 📜utils_music21.py
 ┃ ┗ 📜utils_pretty_midi.py
 ┣ 📂scripts
 ┃ ┣ 📜evaluate.py
 ┃ ┣ 📜generate.py
 ┃ ┗ 📜train.py
 ┣ 📂tests
 ┃ ┣ 📂test_data
 ┃ ┣ 📜test_postprocess.py 
 ┃ ┣ 📜test_preprocess.py 
 ┃ ┗ 📜test_train.py
 ┣ 📜.gitignore
 ┣ 📜pyproject.toml
 ┣ 📜README.md
 ┗ 📜tox.ini
```

- `data`: This folder contains the data that is used for training.
- `scripts`: This folder contains the scripts that are used to train the model
  and generate new sequences with it. The model is stored in deepmusician/
- `tests`: This folder contains the unit tests for the project.
- `deepmusician`: This folder contains the code of the project. This is a python
  package that is installed in the conda environment. This package is used to import
  the code in my scripts. The `pyproject.toml` file contains
  all the information about the installation of this repository. The structure
  of this folder is the following:
  - `__init__.py`: This file is used to initialize the `deepmusician` package.
  - `seq2seq`: coCtains the Sequence2Sequence model.
  - `utils_music21.py`: Contains all the necessary functions for preprocessing
    with data with the music21 package.
  - `utils_pretty_midi.py`: Contains all the necessary functions for preprocessing
    with data with the pretty midi package.
- `pyproject.py`: This file contains all the information about the installation
  of this repository. Can be used  to install this repository as a
  package in a conda environment.
- `tox.ini`: Contains information about the testing process.
- `.circleci`: Contains information about the CI process

### Running the code

Information about the relevant resources for the project can be found in the
`data/README.md` file.

The `scripts` folder contains the scripts to train the model and generate new
sequences.

#### Training

`train.py`: This script is used to train the model. It expects a dataset
  directory as input and runs the training with a set of predefined parameters,
  that proved useful during the experiments.

```bash
train.py [-h] [-p PATH] [-e N_EPOCHS] [-b BATCH_SIZE] [-s SEQ_LEN] [-l NUM_LAYERS] [-c {bce,focal,focal+}] [-g GAMMA] [-a ALPHA] [-lr LEARNING_RATE] [-t THRESHOLD] [-tf TEACHER_FORCING_RATIO] [-d DECODER_N_LAYERS] [-rz] [-div DIVISION] [-ac {cpu,gpu,tpu}]
```

To train the model for 10 Epochs with Mozart's Sonatas, save checkpoints and logging run:

```bash
python scripts/train.py -p 'data/classical_archives/Classical Archives - The Greats (MIDI)/Mozart/Piano Sonatas/' -e 10
```

#### Generation

- `generate.py`: This script is used to generate new sequences of notes with a
  pretrained model. It expects a checkpoint in the form of .ckpt file as well
  as the sequence length of the generated melody.

```bash
generate.py [-h] -c CHECKPOINT [-l SEQ_LEN] [-d DIVISION] [-i {zero,random,guided}] [-s SAVE]
```

To generate a new sequence with a trained model, run:

```bash
python scripts/generate.py -c data/model.ckpt -l 192
```

Additional information about the parameters can be returned via:

```bash
python scripts/train.py -h
python scripts/generate.py -h
```

## Experiment

### Goal

The goal of the project is to create a model that produces melodies that sound
human, or natural, i.e., not mechanical. Making this goal mathematically
quantifiable is not trivial, since there are a variety of ways to represent
notes and sequences of notes that each require different metrics. In addition
most of these metrics are not able to capture the essence of a creative
sequence of notes. Of course, there are metrics that describe how well an
algorithm predicts a certain sequence, but as we will see below, these have
weighty drawbacks. Therefore, I remain with the approach of measuring generated
melodies according to my human ear.

### Representing MIDI

As discussed in more detail above, musical pieces are initially represented
symbolically as midi files. However, neural networks cannot be trained with
midi files themselves. Therefore, these must be converted into another form in
which the notes can be passed to the network. There are a lot of possibilities
for this, of which I have chosen a classical one: The piano roll. Here a 2
dimensional matrix is spanned, whose x-axis represents the time and whose
y-axis represents the 88 notes of the piano. Each touch of a note is marked
with 1 in the matrix at the corresponding time t - all other cells remain empty
(0). This representation is very clear and intuitive. However, it has a big
problem: since only a small percentage of the available cells are filled, i.e.
most of the time NO note is played, there is a big imbalance in the data.

### Metric and Loss

Classical metrics have difficulty dealing with this problem and return
suggestive and misleading values, while classical losses do not optimise for
the desired goal, that is the generation of a human sounding melody.

I faced this problem during the later stages of my experiment, when the model
easily learned according to the classical BCE-loss, but afterwards during
testing only generated empty melodies. This was due to the fact, that it
actually guessed almost all of the notes correctly as they were not being
played. So the empty sequence resembled the input it was given most of the
time. Or put, differently the model was stuck in a local minimum.

#### Focal loss

I learned that image classification faces a similar problem and solves this by
using a so called focal loss. Focal loss is a loss function that is used in
image classification tasks, particularly those involving object detection. The
main idea behind focal loss is to down-weight the contribution of easy examples
in the training data and focus more on the hard examples, which are typically
the ones that are more challenging to classify correctly. This is achieved by
modifying the standard cross-entropy loss function by introducing a weighting
term that increases the loss for easy examples and decreases the loss for hard
examples. The result is a loss function that is more "focal" on the hard
examples and helps the model to better learn from them and improve its
performance on the task.

With the introduction of focal loss in my model it started generating
meaningful sequences of notes. To keep track of the validity of the generation
of sequences I also introduced a density metric, that measures the average
notes played per time step.

Yet, the two parameters of the focal loss (alpha and gamma) need to be
carefully adjusted to obtain meaningful results.

### Architecture

- EncoderRNN: GRU(input: 88, hidden: 512, num_layers=2, dropout=0.2)
- DecoderRNN: GRU(input: 88, hidden: 512, num_layers=2, dropout=0.2)
- Classifier
  - Linear(in_features=512, out_features=256, bias=True)
  - ReLU
  - Dropout(p=0.5)
  - Linear(in_features=256, out_features=88, bias=True)
  - Sigmoid

### Limits

In addition to the aforementioned problem of empty melodies, I also had to
struggle in particular with the limitations of my hardware, which does NOT
include an Nvidia graphics card. To train an epoch merely with Mozart's sonatas
with my CPU takes me a little more than half an hour. However, in this field of
research, training over several hundred epochs is not uncommon. Because of this
limitation, I have only ever been able to carry out my experiments with even
smaller samples and a few epochs, which of course greatly distorts the results.
Over the Christmas holidays, I plan to refine my experiments using Google Colab
Pro+ to really get a model that generates musical sequences, as the melodies
are still very generic and monotonous. The best set of hyperparameters can be
found in ```train.py``` in the corresponding constants.

## Work-Breakdown Structure

| Task        | estimated   | actual      |
| ----------- | ----------- | ----------- |
| Dataset collection      | 7       | 12 |
| Exploring, analysing and preparing data   | 12        | 45
| Designing and building an appropriate network | 25 | 40
| Training and fine-tuning that network | 15 | 15
| Building an application to present the results | 20 |
| Writing the final report | 8 |
| Preparing the presentation of your work | 5 |
| **Sum** | 92 | 112 |

---
## References

<a id="1">[1]</a>
Payne, Christine. "MuseNet." OpenAI, 25 Apr. 2019, openai.com/blog/musenet

<a id="2">[2]</a>
Nicolas Boulanger-Lewandowski, Yoshua Bengio, and Pascal Vincent. 2012. Modeling temporal dependencies in high-dimensional sequences: application to polyphonic music generation and transcription. In Proceedings of the 29th International Coference on International Conference on Machine Learning (ICML'12). Omnipress, Madison, WI, USA, 1881–1888.

<a id="3">[3]</a>
M. K. Jędrzejewska, A. Zjawiński and B. Stasiak, "Generating Musical Expression
of MIDI Music with LSTM Neural Network," 2018 11th International Conference on
Human System Interaction (HSI), 2018, pp. 132-138, doi:
10.1109/HSI.2018.8431033.

<a id="4">[4]</a>
Nabil Hewahi, Salman AlSaigal & Sulaiman AlJanahi (2019) Generation of music
pieces using machine learning: long short-term memory neural networks approach,
Arab Journal of Basic and Applied Sciences, 26:1, 397-413, DOI:
10.1080/25765299.2019.1649972

<a id="5">[5]</a>
Ycart, A., & Benetos, E. (2017). A Study on LSTM Networks for Polyphonic Music Sequence Modelling. ISMIR.

<a id="6">[6]</a>
Mangal, Sanidhya & Modak, Rahul & Joshi, Poorva. (2019). LSTM Based Music Generation System.

<a id="7">[7]</a>
A. Jagannathan, B. Chandrasekaran, S. Dutta, U. R. Patil and M. Eirinaki, "Original Music Generation using Recurrent Neural Networks with Self-Attention," 2022 IEEE International Conference On Artificial Intelligence Testing (AITest), 2022, pp. 56-63, doi: 10.1109/AITest55621.2022.00017.

<a id="8">[8]</a>
Yang, Li-Chia & Chou, Szu-Yu & Yang, yi-hsuan. (2017). MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation using 1D and 2D Conditions.

<a id="9">[9]</a>
Briot, Jean-Pierre & HADJERES, Gaëtan & Pachet, Francois. (2019). Deep Learning
Techniques for Music Generation - A Survey.

<a id="10">[10]</a>
H. H. Mao, T. Shin and G. Cottrell, "DeepJ: Style-Specific Music Generation,"
2018 IEEE 12th International Conference on Semantic Computing (ICSC), 2018, pp.
377-382, doi: 10.1109/ICSC.2018.00077.
