# Data

# MEASTRO 3.0
wget "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip" -O "data/maestro-v3.0.0_midi.zip"
unzip "data/maestro-v3.0.0_midi.zip" -d "data/"
rm "data/maestro-v3.0.0_midi.zip"

# Classical Archives The Greats
wget "https://archive.org/download/ClassicalArchivesTheGreatsMIDILibrary/Classical%20Archives%20-%20The%20Greats%20%28MIDI%20Library%29.rar" -O "data/ClassicalArchivesTheGreatsMIDILibrary.rar"
unrar e "data/ClassicalArchivesTheGreatsMIDILibrary.rar" "data/"
rm "data/ClassicalArchivesTheGreatsMIDILibrary.rar"

# VG Music Archive
wget "https://download1584.mediafire.com/kawkv7sqsvqg/i7q64yoj9j27xbu/2011-03-12-vgmusic.com.zip" -O "data/2011-03-12-vgmusic.com.zip"
unzip "data/2011-03-12-vgmusic.com.zip" -d "data/vgma/"
unzip data/vgma/\*.zip -d "data/vgma/"
rm "data/2011-03-12-vgmusic.com.zip"
rm data/vgma/*.zip

# Video Game Music
wget "https://archive.org/compress/video-game-music-90000-midi-files/formats=ZIP&file=/video-game-music-90000-midi-files.zip" -O "data/video-game-music-90000-midi-files.zip"
unzip "data/video-game-music-90000-midi-files.zip" -d "data/"
unzip data/vgm/\*.zip -d "data/vgm/"
rm "data/video-game-music-90000-midi-files.zip"
rm data/vgm/*.zip

# The Magic of Midi
wget "https://archive.org/download/themagicofmidiv1/The_Magic_of_MIDI.7z" -O "data/The_Magic_of_MIDI.7z"
7z e "data/The_Magic_of_MIDI.7z" -o "data/"
rm "data/The_Magic_of_MIDI.7z"