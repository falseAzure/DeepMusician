import numpy as np
import pandas as pd
import pretty_midi as pm
import os

# utils for pretty_midi
def get_numpy_notes_from_notes(notes, pmidi):
    np_notes = np.zeros([len(notes), 6])
    for i, note in enumerate(notes):
        np_notes[i, 0] = note.start
        np_notes[i, 1] = note.end
        np_notes[i, 2] = note.pitch
        np_notes[i, 3] = note.velocity
        np_notes[i, 4] = pmidi.time_to_tick(note.start)
        np_notes[i, 5] = pmidi.time_to_tick(note.end)
    return np_notes


def get_numpy_notes_from_instrument(instrument, pmidi):
    instrument.remove_invalid_notes()
    notes = instrument.notes
    np_notes = get_numpy_notes_from_notes(notes, pmidi)
    return np_notes


def get_numpy_notes_from_instruments(instruments, pmidi):
    np_notes_all = np.empty((0, 7))
    for instrument in instruments:
        np_notes = get_numpy_notes_from_instrument(instrument, pmidi)
        np_notes = np.append(
            np_notes, np.full((len(np_notes), 1), instrument.program), axis=-1
        )
        np_notes_all = np.append(np_notes_all, np_notes, axis=0)
    return np_notes_all


def get_numpy_notes_from_pm(pmidi):
    instruments = pmidi.instruments
    np_notes_all = get_numpy_notes_from_instruments(instruments, pmidi)
    return np_notes_all


def get_numpy_notes_from_file(midi_file):
    pmidi = pm.PrettyMIDI(midi_file)
    return get_numpy_notes_from_pm(pmidi)


def extract_info_instruments(instruments):
    return {
        ins.program: {"instrument_name": ins.name, "is_drum": ins.is_drum}
        for ins in instruments
    }


def get_numpy_notes_from_files(midi_files=list):
    meta_dict = {}
    np_notes_all = np.empty((0, 8))
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
        np_notes = get_numpy_notes_from_pm(pmidi)
        np_notes = np.append(np_notes, np.full((len(np_notes), 1), i), axis=-1)
        np_notes_all = np.append(np_notes_all, np_notes, axis=0)
    return np_notes_all, meta_dict


def np_notes_to_pd(np_notes):
    df_notes = pd.DataFrame(
        np_notes,
        columns=[
            "start_time",
            "end_time",
            "pitch",
            "velocity",
            "start_tick",
            "end_tick",
            "instrument_id",
            "midi_id",
        ],
    )
    df_notes["track_id"] = df_notes.groupby(["midi_id", "instrument_id"]).ngroup()
    return df_notes


def meta_dict_to_df(meta_dict):
    midi_id = []
    midi_name = []
    midi_tempo = []
    midi_end_time = []
    midi_end_tick = []
    ins_program = []
    ins_name = []
    ins_is_drum = []
    for idx in meta_dict.keys():
        for ins in meta_dict[idx]["instruments"].keys():
            midi_id.append(idx)
            midi_name.append(meta_dict[idx]["filename"])
            midi_tempo.append(meta_dict[idx]["tempo"])
            midi_end_time.append(meta_dict[idx]["end_time"])
            midi_end_tick.append(meta_dict[idx]["end_tick"])
            ins_program.append(ins)
            ins_name.append(meta_dict[idx]["instruments"][ins]["instrument_name"])
            ins_is_drum.append(meta_dict[idx]["instruments"][ins]["is_drum"])

    df_meta = pd.DataFrame(
        {
            "midi_id": midi_id,
            "midi_name": midi_name,
            "midi_tempo": midi_tempo,
            "midi_end_time": midi_end_time,
            "midi_end_tick": midi_end_tick,
            "ins_program": ins_program,
            "ins_name": ins_name,
            "ins_is_drum": ins_is_drum,
        }
    )
    return df_meta
