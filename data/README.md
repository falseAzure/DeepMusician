# Data

The data consists of MIDI-files from several databases, that originate from different sources and
cover a variety of genres:

1. [Meastro](https://magenta.tensorflow.org/datasets/maestro) (maestro/) (1,291)
2. [Classical Archives](https://www.classicalarchives.com>) (classical_archives/) (4,918)
3. [Symbolic Music Midi Data V1.1](https://arxiv.org/pdf/1606.01368.pdf)
4. [Video Game
   Music](https://archive.org/details/video-game-music-90000-midi-files) (vgm/)
   (92,861)
5. [Video Game Music Archive](https://www.vgmusic.com/) (vgma/) (31,581)
6. [Bitmidi](https://bitmidi.com/) (5,311) (bitmidi/)
7. [The Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)(45,129) (lmd_matched/)
8. [The Magic of MIDI V1](https://archive.org/details/themagicofmidiv1) (magic_of_midi/) (169,454)

The number of total titles are indicated in parenthesis. 1-3 cover pieces from
*classical music*. 4 and 5 contain *video game* music. 6-8 comprise of all
different sorts of music genres.

MIDI-files capture the notes and timing of a piece of music with various
additional information.

To download the files into the corresponding folders please run:

```bash
bash data/download.sh
```

Since the databases are quite big, it might be useful to simply download a
smaller sample. To do so please copy the corresponding part in the
```data/download.sh```and execute it in the bash.
