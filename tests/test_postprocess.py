"""
Tests for postprocessing pipeline, e.g. generating music from a trained model
"""
from pathlib import Path

import pytest

from deepmusician.seq2seq import MidiDataModule, Seq2Seq
from deepmusician.utils_music21 import pianoroll_to_df, process_midi_files

DIVISION = 1 / 4


@pytest.fixture(scope="session")
def get_midi_files():
    files = list(Path("tests/test_data/midi_files/").glob("*.mid"))
    files.sort()
    return files


@pytest.fixture(scope="session")
def get_pianorolls(get_midi_files):
    files = get_midi_files
    pianorolls, _, _ = process_midi_files(files[:1], division=DIVISION)
    return pianorolls


@pytest.fixture(scope="session")
def get_datamodule(get_pianorolls):
    pianorolls = get_pianorolls
    datamodule = MidiDataModule(
        pianorolls=pianorolls, split=0.8, batch_size=32, seq_len=96, remove_zeros=False
    )
    datamodule.setup()
    return datamodule


@pytest.fixture(scope="session")
def get_model(get_datamodule):
    datamodule = get_datamodule
    s2s = Seq2Seq(
        n_training_steps=len(datamodule.train_dataloader()), info={"div": DIVISION}
    )
    return s2s


def test_generation(get_model):
    model = get_model
    seq = model.generate_sequence(seq_len=192, init_hidden="zero")
    assert seq.shape == (192, 88), "Generated sequence has wrong shape"
    seq = model.generate_sequence(seq_len=192, init_hidden="random")
    assert seq.shape == (192, 88), "Generated sequence has wrong shape"
    seq = model.generate_sequence(seq_len=192, init_hidden="guided")
    assert seq.shape == (192, 88), "Generated sequence has wrong shape"


def test_backtransformation(get_model):
    model = get_model
    seq = model.generate_sequence(seq_len=192, init_hidden="zero")
    df = pianoroll_to_df(seq, division=DIVISION)
    assert df is not None, "Backtransformation failed"
