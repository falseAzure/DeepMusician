# Deep Musician

Generating new, unheard musical melodies with a deep neural network on the
basis of existing MIDI
files and exploring the different advantages and disadvantages of a variety of
models and the changing complexity of the input.

## Creating MIDI Files with a deep neural network

With **MuseNet OpenAI** created a deep neural network that "can generate
4-minute musical compositions with 10 different instruments, and can combine
styles from country to Mozart to the Beatles." [[1]](#1) Behind this project is
the (philosophical) idea that musical compositions can arise not only from a
particular artistic understanding (of harmony, rhythm, melody, etc.), but
*soley* from a variety of previous works that are incorporated into the new
piece as a wealth of experience. MuseNet is fuelled by "a large-scale
**transformer model** trained to predict the next note(s) in a sequence." [[1]](#1)
While this approach using a transformer model is certainly state of the art and
produces a truly vibrant and rich musical style, it is very resource intensive.

In contrast, there are also more lightweight approaches that use
a variety of RNN structures (such as LSTM models [[2-6]](#2) or RNNs with
Self-Attention [[7]](#7)) and even CNN [[8]](#8) - which is quite interesting
taking the temporal dimension of music into account.
(For a general overview see: [[9]](#9).)

In this project, I want to explore the development, facets, and
differences of generative music models and identify the individual advantages
and disadvantages of each model. In doing so, I hope to gain insight into the
increasing explanatory power and creative potential of the models as their
complexity increases, and to weigh these factors against their cost.

## Dataset

The basis of music is formed by sequentially played sounds or tones, which can
be represented and played as a single complex waveform. While this form of
representation already represents a concrete, unique shaping of the music, it
is also possible to specify the individual tones of the piece in the form of
notes with different parameters. The advantage here is that the concrete
instrumentation is abstracted from and only the structure of the piece is
considered. The generally accepted standard for this is MIDI. By means of MIDI
it is possible to transmit not only pitch and length but also other parameters
such as velocity, which offers the advantage of successively expanding my
model.

Thus, due to its abstract nature MIDI offers the possibility to extend the input of the model successively. While initially only monophonic audio tracks with constant dynamics and tone length are used, these parameters are to be successively added to the input to see how the created melodies change.

I would also like to discuss the characteristics of different styles of music.
Initially, it is planned to single out only one style of music, or better said,
only one artist: Mozart. As a master of melody, Mozart offers the perfect
introduction to the world of beautiful tunes. Subsequently, I would like to
dissolve this restriction and include other artists and epochs as well.[[10]](#10)

For my project I will use MIDI data from different sources
like the **MAESTRO** dataset from Magenta as well as pieces from
**ClassicalArchives**, **BitMidi** and others. This will include a mixture of simple
bulk downloads and web scraping.

## Extensions

This project can thus be summarised under the type of **bring your own method** as it can be expanded successively on the basis of four axes:

- Monphonic - polyphonic
- Additional midi parameters
- Different music styles
- Complexity of the model: RNN - LSTM - Transformer

The fact that these extensions are largely independent of each other results in a modular structure of the project, in that the individual modules can be strung together as desired. This allows me to look at the different aspects of the individual components, but on the other hand does not give a definite goal of the project or model, but only a trajectory. However, this is also intentional and is meant to encourage the project to be pursued and expanded beyond the university levy.

## Work-Breakdown Structure

- dataset collection: 7h
- exploring, analysing and preparing data: 12h
- designing and building an appropriate network: 25h
- training and fine-tuning that network: 15h
- building an application to present the results: 20h
- writing the final report: 8h
- preparing the presentation of your work: 5h

Sum: 92h

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
