import sys

sys.path.append("deepmusician/")
sys.path.append("../deepmusician/")
from pathlib import Path

import numpy as np
import pytest
import utils_music21


# Fixtures
@pytest.fixture(scope="session")
def get_midi_files():
    files = list(Path("tests/test_data/midi_files/").glob("*.mid"))
    files.sort()
    return files


@pytest.fixture(scope="session")
def get_notes_np_meta_dict(get_midi_files):
    notes_np, meta_dict = utils_music21.get_notes_np_from_files(get_midi_files)
    return notes_np, meta_dict


@pytest.fixture(scope="session")
def get_notes_df_meta_df(get_notes_np_meta_dict):
    notes_df = utils_music21.notes_np_to_df(get_notes_np_meta_dict[0])
    meta_df = utils_music21.meta_dict_to_df(get_notes_np_meta_dict[1])

    return notes_df, meta_df


@pytest.fixture(scope="session")
def get_notes_np(get_notes_np_meta_dict):
    return get_notes_np_meta_dict[0]


@pytest.fixture
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
    pianorolls = utils_music21.df_to_pianorolls(get_notes_df, division=1 / 16)
    return pianorolls


@pytest.fixture(scope="session")
def get_all_from_entire_preprocess(get_midi_files):
    pianorolls, notes_df, meta_df = utils_music21.process_midi_files(
        get_midi_files, division=1 / 16
    )
    return pianorolls, notes_df, meta_df


# Tests
def test_midi_files_length(get_midi_files):
    assert len(get_midi_files) == 7


def test_notes_np(get_notes_np, get_meta_dict, get_midi_files):
    # length
    assert max(get_notes_np[:, 5]) + 1 == len(get_midi_files)
    assert len(get_meta_dict) == len(get_midi_files)
    assert get_notes_np.shape == (56421, 6)


def test_notes_df(get_notes_df, get_meta_df):
    # length
    assert len(get_notes_df) == 56421
    assert len(get_notes_df.track_id.unique()) == 7

    # min and max of pitch
    assert max(get_notes_df.pitch) <= 127
    assert min(get_notes_df.pitch) >= 0

    # min and max of velocity
    assert max(get_notes_df.velocity) <= 127
    assert min(get_notes_df.velocity) >= 0

    # number of unique tracks
    assert sum(get_notes_df.midi_id.unique()) == sum(get_meta_df.midi_id.unique())

    # sums
    assert np.isclose(sum(get_notes_df.offset), 50476319.749999896)
    assert np.isclose(sum(get_notes_df.pitch), 3780784)
    assert np.isclose(sum(get_notes_df.velocity), 3397254)
    assert np.isclose(sum(get_notes_df.duration), 28320.083333332597)
    assert np.isclose(sum(get_notes_df.track_id), 175163)


def test_pianorolls(get_pianorolls):

    # check list of pianorolls
    assert len(get_pianorolls) == 7
    assert isinstance(get_pianorolls, list)

    # check individual pianorolls
    for pianoroll in get_pianorolls:
        # all Pianorolls have 88 notes
        assert pianoroll.shape[1] == 88
        # np.array
        assert isinstance(pianoroll, np.ndarray)
        # pianorolls have only 0 and 1
        assert (np.unique(pianoroll) == np.array([0, 1])).all()


def test_entire_preprocess_pipeline(get_all_from_entire_preprocess):
    # check notes_df
    # length
    assert len(get_all_from_entire_preprocess[1]) == 56421
    assert len(get_all_from_entire_preprocess[1].track_id.unique()) == 7

    # min and max of pitch
    assert max(get_all_from_entire_preprocess[1].pitch) <= 127
    assert min(get_all_from_entire_preprocess[1].pitch) >= 0

    # min and max of velocity
    assert max(get_all_from_entire_preprocess[1].velocity) <= 127
    assert min(get_all_from_entire_preprocess[1].velocity) >= 0

    # number of unique tracks
    assert sum(get_all_from_entire_preprocess[1].midi_id.unique()) == sum(
        get_all_from_entire_preprocess[2].midi_id.unique()
    )

    # check pianorolls
    # check list of pianorolls
    assert len(get_all_from_entire_preprocess[0]) == 7
    assert isinstance(get_all_from_entire_preprocess[0], list)

    # check individual pianorolls
    for pianoroll in get_all_from_entire_preprocess[0]:
        # all Pianorolls have 88 notes
        assert pianoroll.shape[1] == 88
        # np.array
        assert isinstance(pianoroll, np.ndarray)
        # pianorolls have only 0 and 1
        assert (np.unique(pianoroll) == np.array([0, 1])).all()
