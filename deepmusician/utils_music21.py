"""
Provides utils functions to preprocess midi files with music21.
"""

import os
import warnings

import matplotlib.pyplot as plt
import music21
import numpy as np
import pandas as pd
import seaborn as sns

DIVISION = 1 / 16


def get_notes_np_from_instrument(instrument):
    """
    Transform the notes given by an music21.instrument object into a 2D numpy
    array (n_notes, 4), that holds pitch, offset, velocity and duration of the
    each individual note. Chords are separated into the individual notes with
    the same offset.

    Args:
        instrument: instrument of a music21 midi object

    Returns:
        numpy.array: contains the notes of the instrument track (n_notes, 4)
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
    """
    Transform the notes given by an music21.midi object into a 2D numpy array
    (n_notes, 5). It does so by iterating over all the individual instrument
    tracks of a midi file and calling get_notes_np_from_instrument. The first 4
    parameters of each note thus represent the same as in
    get_notes_np_from_instrument (pitch, offset, velocity, duration), while the
    last parameter is the instrument number.

    Args:
        midi21: music21 midi object

    Returns:
        numpy.array: contains the notes of the midi file (n_notes, 5)
        instruments: list of instrument tracks of the midi file
    """
    instruments = music21.instrument.partitionByInstrument(midi21)
    notes_all_np = np.empty((0, 5), dtype=float)
    valid_instruments = [ins for ins in instruments if len(ins.notes) > 0]
    for i, instrument in enumerate(valid_instruments):
        notes_np = get_notes_np_from_instrument(instrument)
        notes_np = np.append(notes_np, np.full((len(notes_np), 1), i), axis=-1)
        notes_all_np = np.append(notes_all_np, notes_np, axis=0)
    return notes_all_np, instruments


def get_notes_np_from_file(midi_file):
    """Wrapper function for get_notes_np_from_m21 to directly read a midi file"""
    midi21 = music21.converter.parse(midi_file)
    return get_notes_np_from_m21(midi21)


def extract_info_instruments(instruments):
    """Extracts the instrument name and type from a list of instruments"""
    return {
        i: {
            "instrument_name": ins.getInstrument().partName,
            "instrument_type": ins.partName,
        }
        for i, ins in enumerate(instruments)
    }


def get_notes_np_from_files(midi_files: list):
    """
    Transforms the notes of a list of midi files into a 2D numpy array
    (n_notes, 6). It's a wrapper function for get_notes_np_from_m21 to directly
    read a list of midi files. It iterates over all the midi files and calls
    get_notes_np_from_m21 for each of them. The additional parameter of each
    note represents the file.

    Args:
        midi_files (list): List of midi files

    Returns:
        numpy.array: contains the notes of the midi files (n_notes, 6)
        dictionary: contains the filenames and the instrument information of
        the midi files.
    """
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
    """
    Transforms a 2D numpy array of notes into a pandas DataFrame.

    Args:
        notes_np (numpy.array): 2D numpy array of notes (n_notes, 6). Usually
        obtained by get_notes_np_from_files

    Returns:
        pandas.DataFrame: Dataframe containing the notes of the midi files and
        its parameters (offset, pitch, velocity, duration, instrument_id, file_id)
    """
    notes_df = pd.DataFrame(
        notes_np,
        columns=[
            "offset",
            "pitch",
            "velocity",
            "duration",
            "instrument_id",
            "file_id",
        ],
    )
    notes_df["track_id"] = notes_df.groupby(["file_id", "instrument_id"]).ngroup()
    return notes_df


def meta_dict_to_df(meta_dict):
    """Transforms a dictionary of meta information into a pandas DataFrame."""
    file_id = []
    midi_name = []
    ins_program = []
    ins_name = []
    for idx in meta_dict.keys():
        for ins in meta_dict[idx]["instruments"].keys():
            file_id.append(idx)
            midi_name.append(meta_dict[idx]["filename"])
            ins_program.append(ins)
            ins_name.append(meta_dict[idx]["instruments"][ins]["instrument_name"])

    meta_df = pd.DataFrame(
        {
            "file_id": file_id,
            "midi_name": midi_name,
            "ins_program": ins_program,
            "ins_name": ins_name,
        }
    )
    return meta_df


def df_track_to_pianoroll(notes_df, division=DIVISION):
    """
    Transforms a track (represented by a DataFrame of notes) into a 2D numpy
    (track length, 88), that represents a pianoroll. The first dimension
    captures the time and is defined by the offset of the notes. The second
    dimension captures the pitch of the notes: 88 is the number of piano notes
    (pitch: 21-108).

    Args:
        notes_df (DataFrame): DataFrame containing the notes of the midi files.
        division (float, optional): The granularity/division to subdivide each
        bar. E.g. the smallest possible distance between two notes. Defaults to 1/16.

    Returns:
        numpy.array: 2D numpy array (track length, 88) that represents a pianoroll.
    """
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


def df_to_pianorolls(notes_df, division=DIVISION):
    """
    Transforms a DataFrame of notes into a list of pianorolls. Its a wrapper
    of df_track_to_pianoroll that iterates over all the tracks of the
    DataFrame. Division is the granularity/division to subdivide each bar.
    """
    pianorolls = []
    for track in notes_df.track_id.unique():
        pianorolls.append(
            df_track_to_pianoroll(notes_df[notes_df.track_id == track], division)
        )
    return pianorolls


def process_midi_files(midi_files=list, division=DIVISION):
    """
    Wrapper of the functions to process a list of midi files. It returns a list
    of pianorolls, a DataFrame of notes and a DataFrame of meta information.
    Division is the granularity/division to subdivide each bar.
    """
    notes_all_all, meta_dict = get_notes_np_from_files(midi_files)
    notes_df = notes_np_to_df(notes_all_all)
    meta_df = meta_dict_to_df(meta_dict)
    pianorolls = df_to_pianorolls(notes_df, division)
    return pianorolls, notes_df, meta_df


def get_density(pianoroll):
    return pianoroll.sum() / len(pianoroll)


def plot_pianoroll(pianoroll, length=128, division=DIVISION):
    """
    Plot a pianoroll as a heatmap, that depicts the notes played at each
    time step. Division is the granularity/division to subdivide each bar.
    """
    assert sum(sum(pianoroll)) > 0, "pianoroll is empty"
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
    plt.xlabel(str(division) + "Notes")
    plt.ylabel("Note")
    plt.show()


def pianoroll_to_note(pianoroll, division=DIVISION):
    """
    Backtransform a pianoroll into a list of music21 notes. Division is the
    granularity/division to subdivide each bar.
    """
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


def notes_to_midi(music21_notes):
    """
    Backtransforms a list of music21 notes into a midi file.
    """
    midi_stream = music21.stream.Stream(
        music21_notes, timeSignature=music21.meter.TimeSignature("4/4")
    )
    return midi_stream


def save_midi(midi_stream, path=None):
    if path is not None:
        print("Saving midi file to {}".format(path))
        midi_stream.write("midi", fp=path)


def pianoroll_to_midi(pianoroll, division=DIVISION, save=None):
    """Wrapper of pianoroll_to_note and notes_to_midi."""
    music21_notes = pianoroll_to_note(pianoroll, division)
    return notes_to_midi(music21_notes)


def pianoroll_to_df(pianoroll, division=DIVISION):
    """Backtransform a pianoroll into a DataFrame of notes."""
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
    notes_df["n_notes"] = notes_df.groupby("offset").transform("count")["note"]
    return notes_df
