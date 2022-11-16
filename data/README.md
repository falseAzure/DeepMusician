# Data

The data consists of MIDI-files from several databases, that originate from different sources and
cover a variety of genres:

1. [Meastro](https://magenta.tensorflow.org/datasets/maestro) (maestro/) (1,291)
2. [Classical Archives](https://www.classicalarchives.com>) (classical_archives/) (4,918)
3. [Video Game
   Music](https://archive.org/details/video-game-music-90000-midi-files) (vgm/)
   (92,861)
4. [Video Game Music Archive](https://www.vgmusic.com/) (vgma/) (31,581)
5. [Bitmidi](https://bitmidi.com/) (bitmidi/)
6. [The Magic of MIDI V1](https://archive.org/details/themagicofmidiv1) (magic_of_midi/) (169,454)

The location and the number of total
titles are indicated in parentheses. (1) and (2) cover pieces from *classical music*. (3) and (4) contain *video game* music. (5) and (6) comprise
of all
different sorts of music genres.

MIDI-files capture the notes and timing of a piece of music with various
additional information.

To download the files into the corresponding folders please run:

```bash
bash data/download.sh
```
