"""
Tests for training pipeline
"""
from pathlib import Path

import pytest

from deepmusician.seq2seq import MidiDataModule, Seq2Seq, get_trainer
from deepmusician.utils_music21 import process_midi_files

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


@pytest.fixture(scope="session")
def get_trainer_pytest():
    return get_trainer(accelerator="cpu", n_epochs=1, test=True)


def test_model(get_model):
    model = get_model
    assert model is not None, "Model is None"


def test_trainer(get_trainer_pytest):
    trainer = get_trainer_pytest
    assert trainer is not None, "Trainer is None"


def test_training(get_model, get_datamodule, get_trainer_pytest):
    model = get_model
    datamodule = get_datamodule
    trainer = get_trainer_pytest
    trainer.fit(model, datamodule)
    assert trainer is not None, "Training failed"
