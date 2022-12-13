# import sys

# sys.path.append("deepmusician/")
# sys.path.append("../deepmusician/")
from pathlib import Path

import pytest
from seq2seq import MidiDataModule, Seq2Seq
from utils_music21 import process_midi_files

DIV = 1 / 4


@pytest.fixture(scope="session")
def get_midi_files():
    files = list(Path("tests/test_data/midi_files/").glob("*.mid"))
    files.sort()
    return files


@pytest.fixture(scope="session")
def get_pianorolls(get_midi_files):
    files = get_midi_files
    pianorolls, _, _ = process_midi_files(files[:1], division=DIV)
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
        n_training_steps=len(datamodule.train_dataloader()), info={"div": DIV}
    )
    return s2s


def test_generation(get_model):
    model = get_model
    seq = model.generate_sequence(seq_len=192, init_hidden="zero")
    assert seq.shape == (192, 88)
    seq = model.generate_sequence(seq_len=192, init_hidden="random")
    assert seq.shape == (192, 88)
    seq = model.generate_sequence(seq_len=192, init_hidden="guided")
    assert seq.shape == (192, 88)
