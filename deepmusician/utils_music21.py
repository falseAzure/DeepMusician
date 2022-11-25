import os

import music21
import numpy as np
import pandas as pd


def get_numpy_notes_from_instrument(instrument):
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
        else:
            pitch.append(note.pitch.midi)
            offset.append(note.offset)
            velocity.append(note.volume.velocity)
            duration.append(note.duration.quarterLength)

    np_notes = np.array([offset, pitch, velocity, duration], dtype=float).T
    return np_notes


def get_numpy_notes_from_m21(m21_midi):
    instruments = music21.instrument.partitionByInstrument(m21_midi)
    np_notes_all = np.empty((0, 5), dtype=float)
    for i, instrument in enumerate(instruments):
        np_notes = get_numpy_notes_from_instrument(instrument)
        np_notes = np.append(np_notes, np.full((len(np_notes), 1), i), axis=-1)
        np_notes_all = np.append(np_notes_all, np_notes, axis=0)
    return np_notes_all


def get_numpy_notes_from_file(fname):
    midi = music21.converter.parse(fname)
    return get_numpy_notes_from_m21(midi)


def extract_info_instruments(instruments):
    return {
        i: {
            "instrument_name": ins.getInstrument().partName,
            "instrument_type": ins.partName,
        }
        for i, ins in enumerate(instruments)
    }


def get_numpy_notes_from_files(midi_files=list):
    meta_dict = {}
    np_notes_all = np.empty((0, 6), dtype=float)
    for i, fname in enumerate(midi_files):
        name = os.path.basename(fname)
        print(f"Processing {i+1}/{len(midi_files)}: {name:100}", end="\r", flush=True)
        midi = music21.converter.parse(fname)
        meta_dict[i] = {
            "filename": fname,
            # "instruments": dict(extract_info_instruments(midi.parts)),
        }
        np_notes = get_numpy_notes_from_m21(midi)
        np_notes = np.append(np_notes, np.full((len(np_notes), 1), i), axis=-1)
        np_notes_all = np.append(np_notes_all, np_notes, axis=0)
    return np_notes_all, meta_dict


def numpy_notes_to_pd(np_notes):
    df_notes = pd.DataFrame(
        np_notes,
        columns=[
            "offset",
            "pitch",
            "velocity",
            "duration",
            "instrument_id",
            "midi_id",
        ],
    )
    df_notes["track_id"] = df_notes.groupby(["midi_id", "instrument_id"]).ngroup()
    return df_notes


def meta_dict_to_df(meta_dict):
    midi_id = []
    midi_name = []
    # ins_program = []
    # ins_name = []
    # for idx in meta_dict.keys():
    #     for ins in meta_dict[idx]["instruments"].keys():
    #         midi_id.append(idx)
    #         midi_name.append(meta_dict[idx]["filename"])
    #         ins_program.append(ins)
    #         ins_name.append(meta_dict[idx]["instruments"][ins]["instrument_name"])
            
    for idx in meta_dict.keys():
        midi_id.append(idx)
        midi_name.append(meta_dict[idx]["filename"])

    df_meta = pd.DataFrame(
        {
            "midi_id": midi_id,
            "midi_name": midi_name,
            # "ins_program": ins_program,
            # "ins_name": ins_name,
        }
    )
    return df_meta


def df_track_to_pianoroll(df_notes, track, division=1 / 16):
    offset = np.round(
        np.array(
            df_notes[df_notes.track_id == track].offset / (division * 4), dtype=int
        )
    )
    pitch = np.array(df_notes[df_notes.track_id == track].pitch - 21, dtype=int)
    assert len(offset) == len(pitch)

    pianoroll = np.zeros((int(offset.max()) + 1, 88), dtype=int)
    for i, o_ in enumerate(offset):
        # only pinao notes
        if pitch[i] < 88 and pitch[i] >= 0:
            pianoroll[o_, pitch[i]] = 1

    return pianoroll


def df_to_pianorolls(df_notes, division=1 / 16):
    pianorolls = []
    for track in df_notes.track_id.unique():
        pianorolls.append(df_track_to_pianoroll(df_notes, track, division))
    return pianorolls


def process_midi_files(midi_files, division=1 / 16):
    np_notes_all, meta_dict = get_numpy_notes_from_files(midi_files)
    df_notes = numpy_notes_to_pd(np_notes_all)
    df_meta = meta_dict_to_df(meta_dict)
    pianorolls = df_to_pianorolls(df_notes, division)
    return pianorolls, df_notes, df_meta
