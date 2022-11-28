"""
Provides utils functions for the preprocessing of the midi files with music21.
"""

import os
import warnings

import matplotlib.pyplot as plt
import music21
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def get_notes_np_from_instrument(instrument):
    """Transform the notes given by an instrument track into a 2D (n_notes, 4)
    numpy array, that holds pitch, offset, velocity and duration of the each
    individual note. Chords are separated into the individual notes.
    """
    notes = instrument.recurse().notes
    pitch = []
    offset = []
    velocity = []
    duration = []
    for note in notes:
        if note.duration.quarterLength == 0:
            continue
        if isinstance(note, music21.chord.Chord):
            for pitch_ in note.pitches:
                pitch.append(pitch_.midi)
                offset.append(note.offset)
                velocity.append(note.volume.velocity)
                duration.append(note.duration.quarterLength)
        elif isinstance(note, music21.note.Unpitched) or isinstance(
            note, music21.percussion.PercussionChord
        ):
            pitch.append(0)
            offset.append(note.offset)
            velocity.append(note.volume.velocity)
            duration.append(note.duration.quarterLength)
        else:
            pitch.append(note.pitch.midi)
            offset.append(note.offset)
            velocity.append(note.volume.velocity)
            duration.append(note.duration.quarterLength)

    notes_np = np.array([offset, pitch, velocity, duration], dtype=float).T
    return notes_np


def get_notes_np_from_m21(midi21):
    instruments = music21.instrument.partitionByInstrument(midi21)
    notes_all_np = np.empty((0, 5), dtype=float)
    valid_instruments = [ins for ins in instruments if len(ins.notes) > 0]
    for i, instrument in enumerate(valid_instruments):
        notes_np = get_notes_np_from_instrument(instrument)
        notes_np = np.append(notes_np, np.full((len(notes_np), 1), i), axis=-1)
        notes_all_np = np.append(notes_all_np, notes_np, axis=0)
    return notes_all_np, instruments


def get_notes_np_from_file(midi_file):
    midi21 = music21.converter.parse(midi_file)
    return get_notes_np_from_m21(midi21)


def extract_info_instruments(instruments):
    return {
        i: {
            "instrument_name": ins.getInstrument().partName,
            "instrument_type": ins.partName,
        }
        for i, ins in enumerate(instruments)
    }


# def extract_info_instruments(instruments):
#     info_instruments = {}
#     for i, ins in enumerate(instruments):
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             instrument_name = ins.getInstrument().partName
#             instrument_type = ins.partName
#         info_instruments[i] = {
#             "instrument_name": instrument_name,
#             "instrument_type": instrument_type,
#         }
#     return info_instruments


def get_notes_np_from_files(midi_files=list):
    meta_dict = {}
    notes_all_np = np.empty((0, 6), dtype=float)
    for i, fname in enumerate(midi_files):
        name = os.path.basename(fname)
        print(f"Processing {i+1}/{len(midi_files)}: {name:100}", end="\r", flush=True)
        # music21 issues warning if instrument is unknown
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            midi21 = music21.converter.parse(fname)
        notes_np, instruments = get_notes_np_from_m21(midi21)
        notes_np = np.append(notes_np, np.full((len(notes_np), 1), i), axis=-1)
        notes_all_np = np.append(notes_all_np, notes_np, axis=0)
        meta_dict[i] = {
            "filename": fname,
            "instruments": dict(extract_info_instruments(instruments)),
        }
    return notes_all_np, meta_dict


def notes_np_to_df(notes_np):
    notes_df = pd.DataFrame(
        notes_np,
        columns=[
            "offset",
            "pitch",
            "velocity",
            "duration",
            "instrument_id",
            "midi_id",
        ],
    )
    notes_df["track_id"] = notes_df.groupby(["midi_id", "instrument_id"]).ngroup()
    return notes_df


def meta_dict_to_df(meta_dict):
    midi_id = []
    midi_name = []
    ins_program = []
    ins_name = []
    for idx in meta_dict.keys():
        for ins in meta_dict[idx]["instruments"].keys():
            midi_id.append(idx)
            midi_name.append(meta_dict[idx]["filename"])
            ins_program.append(ins)
            ins_name.append(meta_dict[idx]["instruments"][ins]["instrument_name"])

    meta_df = pd.DataFrame(
        {
            "midi_id": midi_id,
            "midi_name": midi_name,
            "ins_program": ins_program,
            "ins_name": ins_name,
        }
    )
    return meta_df


def df_track_to_pianoroll(notes_df, division=1 / 16):
    offset = np.round(np.array(notes_df.offset / (division * 4), dtype=int))
    pitch = np.array(notes_df.pitch - 21, dtype=int)
    assert len(offset) == len(pitch)

    pianoroll = np.zeros((int(offset.max()) + 1, 88), dtype=int)
    for i, off in enumerate(offset):
        # duplicated notes are ignored, since they are placed on the same
        # offset and pitch > the returned pianoroll might have less notes than the input
        # only piano notes are considered (pitch: 21-108)
        if pitch[i] < 88 and pitch[i] >= 0:
            pianoroll[off, pitch[i]] = 1

    return pianoroll


def df_to_pianorolls(notes_df, division=1 / 16):
    pianorolls = []
    for track in notes_df.track_id.unique():
        pianorolls.append(
            df_track_to_pianoroll(notes_df[notes_df.track_id == track], division)
        )
    return pianorolls


def process_midi_files(midi_files=list, division=1 / 16):
    notes_all_all, meta_dict = get_notes_np_from_files(midi_files)
    notes_df = notes_np_to_df(notes_all_all)
    meta_df = meta_dict_to_df(meta_dict)
    pianorolls = df_to_pianorolls(notes_df, division)
    return pianorolls, notes_df, meta_df


def get_density(pianoroll):
    return sum(sum(pianoroll)) / len(pianoroll)


def get_train_val_test(pianorolls, train=0.8, val=0.1, test=0.1):
    assert train + val + test == 1
    n = len(pianorolls)
    train_idx = int(n * train)
    val_idx = int(n * (train + val))
    return (
        torch.tensor(np.concatenate(pianorolls[:train_idx], axis=0)),
        torch.tensor(np.concatenate(pianorolls[train_idx:val_idx], axis=0)),
        torch.tensor(np.concatenate(pianorolls[val_idx:], axis=0)),
        range(train_idx),
        range(train_idx, val_idx),
        range(val_idx, n),
    )


def plot_pianoroll(pianoroll, length=128):
    min_note = min(np.where(pianoroll[:length] == 1)[1])
    max_note = max(np.where(pianoroll[:length] == 1)[1])
    first_note = min(np.where(pianoroll == 1)[0])

    data = pianoroll[first_note:length, min_note : max_note + 1].T

    x = range(min_note, max_note + 1)
    x_notes = []
    for n in x:
        x_notes.append(str(music21.pitch.Pitch(n + 21)))
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        data,
        cbar=False,
        square=True,
        linecolor="black",
        linewidths=0.1,
        cmap="Blues",
        yticklabels=x_notes,
    )
    ax.invert_yaxis()
    plt.xlabel("1/16 Notes")
    plt.ylabel("Note")
    plt.show()


def pianoroll_to_note(pianoroll, division=1 / 16):
    music21_notes = []
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
            note = music21.note.Note(pitch + 21)
            note.storedInstrument = music21.instrument.Piano()
            note.offset = i * (division * 4)
            music21_notes.append(note)
        if n_notes > 1:
            chord_list = []
            pitches = np.where(
                pianoroll[
                    i,
                ]
                == 1
            )
            for pitch in pitches[0]:
                note = music21.note.Note(pitch + 21)
                chord_list.append(note)
            chord = music21.chord.Chord(chord_list)
            chord.storedInstrument = music21.instrument.Piano()
            chord.offset = i * (division * 4)
            music21_notes.append(chord)
    return music21_notes


def notes_to_midi(music21_notes, save=None):
    midi_stream = music21.stream.Stream(
        music21_notes, timeSignature=music21.meter.TimeSignature("4/4")
    )
    if save is not None:
        print("Saving midi file to {}".format(save))
        midi_stream.write("midi", fp=save)
    return midi_stream


def pianoroll_to_midi(pianoroll, division=1 / 16, save=None):
    music21_notes = pianoroll_to_note(pianoroll, division)
    return notes_to_midi(music21_notes, save)


def pianoroll_to_df(pianoroll):
    division = 1 / 16
    notes_df = pd.DataFrame(pianoroll, columns=["note" + str(i) for i in range(88)])
    notes_df["id"] = notes_df.index
    notes_df = pd.wide_to_long(notes_df, stubnames=["note"], i="id", j="p")
    notes_df = notes_df[notes_df.note == 1]
    notes_df["pitch"] = notes_df.index.get_level_values(1) + 21
    notes_df["offset"] = notes_df.index.get_level_values(0) * (division * 4)
    notes_df["velocity"] = 100
    notes_df["duration"] = 1
    notes_df.sort_index(inplace=True)
    notes_df.reset_index(drop=True, inplace=True)
    notes_df['n_notes'] = notes_df.groupby('offset').transform('count')['note']
    return notes_df
