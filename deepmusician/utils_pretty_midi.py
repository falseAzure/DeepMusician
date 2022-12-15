"""
Provides utils functions to preprocess midi files with pretty_midi.
"""
import os

import IPython
import numpy as np
import pandas as pd
import pretty_midi as pm


def get_notes_np_from_notes(notes, pmidi):
    """
    Transforms a list of pretty_midi.Note objects into a 2D numpy array
    (n_notes, 6) with the following columns: start, end, pitch, velocity,
    start_tick, end_tick.
    """
    notes_np = np.zeros([len(notes), 6])
    for i, note in enumerate(notes):
        notes_np[i, 0] = note.start
        notes_np[i, 1] = note.end
        notes_np[i, 2] = note.pitch
        notes_np[i, 3] = note.velocity
        notes_np[i, 4] = pmidi.time_to_tick(note.start)
        notes_np[i, 5] = pmidi.time_to_tick(note.end)
    return notes_np


def get_notes_np_from_instrument(instrument, pmidi):
    """
    Wrapper for get_notes_np_from_notes that iterates over all instruments
    of a pretty_midi.PrettyMIDI object.
    """
    instrument.remove_invalid_notes()
    notes = instrument.notes
    notes_np = get_notes_np_from_notes(notes, pmidi)
    return notes_np


def get_notes_np_from_instruments(instruments, pmidi):
    """Wrapper for get_notes_np_from_instrument that iterates over all
    instruments. Adds a column with the instrument program number."""
    notes_all_np = np.empty((0, 7))
    for instrument in instruments:
        notes_np = get_notes_np_from_instrument(instrument, pmidi)
        notes_np = np.append(
            notes_np, np.full((len(notes_np), 1), instrument.program), axis=-1
        )
        notes_all_np = np.append(notes_all_np, notes_np, axis=0)
    return notes_all_np


def get_notes_np_from_pm(pmidi):
    """Wrapper for get_notes_np_from_instruments that handles a single
    pretty_midi object."""
    instruments = pmidi.instruments
    notes_all_np = get_notes_np_from_instruments(instruments, pmidi)
    return notes_all_np


def get_notes_np_from_file(midi_file):
    """Wrapper for get_notes_np_from_pm that handles a single midi file."""
    pmidi = pm.PrettyMIDI(midi_file)
    return get_notes_np_from_pm(pmidi)


def extract_info_instruments(instruments):
    """Extract instrument information from a list of pretty_midi.Instrument objects."""
    return {
        ins.program: {"instrument_name": ins.name, "is_drum": ins.is_drum}
        for ins in instruments
    }


def get_notes_np_from_files(midi_files=list):
    """
    Wrapper for get_notes_np_from_pm that handles a list of midi files and
    returns a 2D numpy array (n_notes, 8) with the following columns: start,
    end, pitch, velocity, start_tick, end_tick, instrument_id, file_id. As well
    as a dictionary with metadata for each midi file.
    """
    meta_dict = {}
    notes_all_np = np.empty((0, 8))
    for i, fname in enumerate(midi_files):
        name = os.path.basename(fname)
        print(f"Processing {i+1}/{len(midi_files)}: {name:100}", end="\r", flush=True)
        pmidi = pm.PrettyMIDI(fname)
        meta_dict[i] = {
            "filename": fname,
            "tempo": pmidi.estimate_tempo(),
            "end_time": pmidi.get_end_time(),
            "end_tick": pmidi.time_to_tick(pmidi.get_end_time()),
            "instruments": dict(extract_info_instruments(pmidi.instruments)),
        }
        notes_np = get_notes_np_from_pm(pmidi)
        notes_np = np.append(notes_np, np.full((len(notes_np), 1), i), axis=-1)
        notes_all_np = np.append(notes_all_np, notes_np, axis=0)
    return notes_all_np, meta_dict


def notes_np_to_df(notes_np):
    """Transform the 2D (n_notes, 8) notes numpy array into a pandas
    DataFrame."""
    notes_df = pd.DataFrame(
        notes_np,
        columns=[
            "start_time",
            "end_time",
            "pitch",
            "velocity",
            "start_tick",
            "end_tick",
            "instrument_id",
            "file_id",
        ],
    )
    notes_df["track_id"] = notes_df.groupby(["file_id", "instrument_id"]).ngroup()
    return notes_df


def meta_dict_to_df(meta_dict):
    """Transforms the metadata dictionary into a pandas DataFrame."""
    file_id = []
    midi_name = []
    midi_tempo = []
    midi_end_time = []
    midi_end_tick = []
    ins_program = []
    ins_name = []
    ins_is_drum = []
    for idx in meta_dict.keys():
        for ins in meta_dict[idx]["instruments"].keys():
            file_id.append(idx)
            midi_name.append(meta_dict[idx]["filename"])
            midi_tempo.append(meta_dict[idx]["tempo"])
            midi_end_time.append(meta_dict[idx]["end_time"])
            midi_end_tick.append(meta_dict[idx]["end_tick"])
            ins_program.append(ins)
            ins_name.append(meta_dict[idx]["instruments"][ins]["instrument_name"])
            ins_is_drum.append(meta_dict[idx]["instruments"][ins]["is_drum"])

    meta_df = pd.DataFrame(
        {
            "file_id": file_id,
            "midi_name": midi_name,
            "midi_tempo": midi_tempo,
            "midi_end_time": midi_end_time,
            "midi_end_tick": midi_end_tick,
            "ins_program": ins_program,
            "ins_name": ins_name,
            "ins_is_drum": ins_is_drum,
        }
    )
    return meta_df


def play_midi(midi_file):
    """Plays a midi file in a jupyter notebook."""
    pmidi = pm.PrettyMIDI(midi_file)
    return IPython.display.Audio(pmidi.synthesize(fs=16000), rate=16000)


# TODO write a function, that plays a piano roll
