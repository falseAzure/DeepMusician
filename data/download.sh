# Data

# MEASTRO 3.0
wget "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip" -O "data/maestro-v3.0.0_midi.zip"
unzip "data/maestro-v3.0.0_midi.zip" -d "data/"
rm "data/maestro-v3.0.0_midi.zip"
mv "data/maestro-v3.0.0" "data/maestro"

# Classical Archives The Greats
wget "https://archive.org/download/ClassicalArchivesTheGreatsMIDILibrary/Classical%20Archives%20-%20The%20Greats%20%28MIDI%20Library%29.rar" -O "data/ClassicalArchivesTheGreatsMIDILibrary.rar"
unrar e "data/ClassicalArchivesTheGreatsMIDILibrary.rar" "data/"
rm "data/ClassicalArchivesTheGreatsMIDILibrary.rar"
mv "data/Classical Archives - The Greats (MIDI Library)" "data/classical_archives"

# VG Music Archive
wget "https://archive.org/download/31581VideogameMusicMIDIFileswReplayGain8mbgmsfx.sf2/31%2C581%20Videogame%20Music%20MIDI%20Files%20%28w%20Replay%20Gain%20-%208mbgmsfx.sf2%29.zip" -O "data/vgmusic.com.zip"
unzip "data/vgmusic.com.zip" -d "data/"
mv "data/31,581 Videogame Music MIDI Files (w Replay Gain - 8mbgmsfx.sf2)" "data/vgma"
rm "data/vgmusic.com.zip"
rm "data/S-YXG50-4MB.dll"
rm "data/8mbgmsfx.sf2"
find "data/vgma" -type d -empty -delete

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
mv "data/The_Magic_of_MIDI" "data/magic_of_midi"