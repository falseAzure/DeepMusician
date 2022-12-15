"""
Tests for preprocessing pipeline
"""
from pathlib import Path

import numpy as np
import pytest

from deepmusician.utils_music21 import (df_to_pianorolls,
                                        get_notes_np_from_files,
                                        meta_dict_to_df, notes_np_to_df,
                                        process_midi_files)


# Fixtures
@pytest.fixture(scope="session")
def get_midi_files():
    files = list(Path("tests/test_data/midi_files/").glob("*.mid"))
    files.sort()
    return files


@pytest.fixture(scope="session")
def get_notes_np_meta_dict(get_midi_files):
    notes_np, meta_dict = get_notes_np_from_files(get_midi_files)
    return notes_np, meta_dict


@pytest.fixture(scope="session")
def get_notes_df_meta_df(get_notes_np_meta_dict):
    notes_df = notes_np_to_df(get_notes_np_meta_dict[0])
    meta_df = meta_dict_to_df(get_notes_np_meta_dict[1])

    return notes_df, meta_df


@pytest.fixture(scope="session")
def get_notes_np(get_notes_np_meta_dict):
    return get_notes_np_meta_dict[0]


@pytest.fixture(scope="session")
def get_meta_dict(get_notes_np_meta_dict):
    return get_notes_np_meta_dict[1]


@pytest.fixture(scope="session")
def get_notes_df(get_notes_df_meta_df):
    return get_notes_df_meta_df[0]


@pytest.fixture(scope="session")
def get_meta_df(get_notes_df_meta_df):
    return get_notes_df_meta_df[1]


@pytest.fixture(scope="session")
def get_pianorolls(get_notes_df):
    pianorolls = df_to_pianorolls(get_notes_df, division=1 / 16)
    return pianorolls


@pytest.fixture(scope="session")
def get_all_from_entire_preprocess(get_midi_files):
    pianorolls, notes_df, meta_df = process_midi_files(get_midi_files, division=1 / 16)
    return pianorolls, notes_df, meta_df


# Tests
def test_midi_files_length(get_midi_files):
    assert len(get_midi_files) == 7, "Wrong number of midi files"


def test_notes_np(get_notes_np, get_meta_dict, get_midi_files):
    # length
    assert max(get_notes_np[:, 5]) + 1 == len(
        get_midi_files
    ), "Wrong number of midi files"
    assert len(get_meta_dict) == len(get_midi_files), "Wrong number of midi files"
    assert get_notes_np.shape == (56421, 6), "Wrong shape of notes_np"


def test_notes_df(get_notes_df, get_meta_df):
    # length
    assert len(get_notes_df) == 56421, "Wrong number of notes"
    assert len(get_notes_df.track_id.unique()) == 7, "Wrong number of tracks"

    # min and max of pitch
    assert max(get_notes_df.pitch) <= 127, "Wrong max pitch"
    assert min(get_notes_df.pitch) >= 0, "Wrong min pitch"

    # min and max of velocity
    assert max(get_notes_df.velocity) <= 127, "Wrong max velocity"
    assert min(get_notes_df.velocity) >= 0, "Wrong min velocity"

    # number of unique tracks
    assert sum(get_notes_df.file_id.unique()) == sum(
        get_meta_df.file_id.unique()
    ), "Wrong number of midi files"

    # sums
    assert np.isclose(
        sum(get_notes_df.offset), 50476319.749999896
    ), "Wrong sum of offset"
    assert np.isclose(sum(get_notes_df.pitch), 3780784), "Wrong sum of pitch"
    assert np.isclose(sum(get_notes_df.velocity), 3397254), "Wrong sum of velocity"
    assert np.isclose(
        sum(get_notes_df.duration), 28320.083333332597
    ), "Wrong sum of duration"
    assert np.isclose(sum(get_notes_df.track_id), 162352), "Wrong sum of track_id"


def test_pianorolls(get_pianorolls):

    # check list of pianorolls
    assert len(get_pianorolls) == 7, "Wrong number of tracks"
    assert isinstance(get_pianorolls, list), "Wrong type of get_pianorolls"

    # check individual pianorolls
    for pianoroll in get_pianorolls:
        # all Pianorolls have 88 notes
        assert pianoroll.shape[1] == 88, "Wrong number of notes"
        # np.array
        assert isinstance(pianoroll, np.ndarray), "Wrong type of pianoroll"
        # pianorolls have only 0 and 1
        assert (
            np.unique(pianoroll) == np.array([0, 1])
        ).all(), "Wrong values in pianoroll"


def test_entire_preprocess_pipeline(get_all_from_entire_preprocess):
    # check notes_df
    # length
    assert len(get_all_from_entire_preprocess[1]) == 56421, "Wrong number of notes"
    assert (
        len(get_all_from_entire_preprocess[1].track_id.unique()) == 7
    ), "Wrong number of tracks"

    # min and max of pitch
    assert max(get_all_from_entire_preprocess[1].pitch) <= 127, "Wrong max pitch"
    assert min(get_all_from_entire_preprocess[1].pitch) >= 0, "Wrong min pitch"

    # min and max of velocity
    assert max(get_all_from_entire_preprocess[1].velocity) <= 127, "Wrong max velocity"
    assert min(get_all_from_entire_preprocess[1].velocity) >= 0, "Wrong min velocity"

    # number of unique tracks
    assert sum(get_all_from_entire_preprocess[1].file_id.unique()) == sum(
        get_all_from_entire_preprocess[2].file_id.unique()
    ), "Wrong number of tracks"

    # check pianorolls
    # check list of pianorolls
    assert len(get_all_from_entire_preprocess[0]) == 7, "Wrong number of tracks"
    assert isinstance(
        get_all_from_entire_preprocess[0], list
    ), "Wrong type of get_pianorolls"

    # check individual pianorolls
    for pianoroll in get_all_from_entire_preprocess[0]:
        # all Pianorolls have 88 notes
        assert pianoroll.shape[1] == 88, "Wrong number of notes"
        # np.array
        assert isinstance(pianoroll, np.ndarray), "Wrong type of pianoroll"
        # pianorolls have only 0 and 1
        assert (
            np.unique(pianoroll) == np.array([0, 1])
        ).all(), "Wrong values in pianoroll"
