import base64
import io

import numpy as np
from midiutil.MidiFile import MIDIFile


def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode("utf-8")


def pianoroll_to_midi(
    pianoroll, track_name="Sample Track", tempo=120, volume=100, duration=1 / 2
):
    mf = MIDIFile(1)  # only 1 track
    track = 0  # the only track

    time = 0  # start at the beginning
    mf.addTrackName(track, time, track_name)
    mf.addTempo(track, time, tempo)
    # add some notes
    channel = 0

    for i in range(len(pianoroll)):
        n_notes = sum(
            pianoroll[
                i,
            ]
        )
        if n_notes == 1:
            pitch = int(
                np.where(
                    pianoroll[
                        i,
                    ]
                    == 1
                )[0]
            )
            pitch = pitch + 21
            mf.addNote(track, channel, pitch, i, duration, volume)
        if n_notes > 1:
            pitches = np.where(
                pianoroll[
                    i,
                ]
                == 1
            )
            for pitch in pitches[0]:
                pitch = pitch + 21
                mf.addNote(track, channel, pitch, i, duration, volume)
    return mf
