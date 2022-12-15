import argparse
import os
import re
from pathlib import Path

from deepmusician.seq2seq import (
    DecoderRNN,
    EncoderRNN,
    MidiDataModule,
    Seq2Seq,
    get_trainer,
)
from deepmusician.utils_music21 import process_midi_files

DIV = 1 / 4
INPUT_DIM = 88
SEQ_LEN = 96
HIDDEN_SIZE = 512
N_LAYERS = 2
BATCH_SIZE = 128
N_EPOCHS = 100
LEARNING_RATE = 0.001
DROP_OUT = 0.2
TEACHER_FORCING_RATIO = 0.5
N_WORKERS = 8
CRITERION = "focal"
GAMMA = 2  # 1
ALPHA = 0.000001  # 1e-6,
THRESHOLD = 0.5
ACCELERATOR = "cpu"
BASE_PATH = "data/classical_archives/Classical Archives - The Greats (MIDI)/"
PATH = os.path.join(
    *[
        BASE_PATH,
        "Mozart",
        "Piano Sonatas",
        "Piano Sonata n01 K279.mid",
    ]
)


def load_data(path, division=DIV):
    """Load midi files from path and preprocess them."""
    # get files
    path = Path(path)
    if path.is_file():
        assert bool(
            re.match(r"\.[mM][iI][dD]", path.suffix)
        ), "Given file is not a midi file"
        files = [path]
    else:
        files = list(path.rglob("*.[mM][iI][dD]"))
    assert len(files) > 0, "No midi files found"
    # preprocess files
    print("Preprocessing Midi Files\n")
    pianorolls, _, _ = process_midi_files(files, division=division)
    print()
    return pianorolls


def train_seq2seq(
    pianorolls,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    num_layers=N_LAYERS,
    criterion=CRITERION,
    gamma=GAMMA,
    alpha=ALPHA,
    learning_rate=LEARNING_RATE,
    threshold=THRESHOLD,
    teacher_forcing_ratio=TEACHER_FORCING_RATIO,
    remove_zeros=False,
    division=DIV,
    accelerator=ACCELERATOR,
):
    """Train a Seq2Seq model with the given parameters."""
    assert criterion in [
        "bce",
        "focal",
        "focal+",
    ], "type must be 'bce', 'focal' or 'bce'"

    # create datamodule
    print("Creating DataModule\n")
    datamodule = MidiDataModule(
        pianorolls=pianorolls,
        split=0.8,
        batch_size=batch_size,
        seq_len=seq_len,
        remove_zeros=remove_zeros,
    )
    datamodule.setup()

    # create model
    print("Creating Model\n")
    n_ = len(datamodule.train_dataloader())
    s2s = Seq2Seq(
        encoder=EncoderRNN(num_layers=num_layers),
        decoder=DecoderRNN(num_layers=num_layers),
        criterion=criterion,
        gamma=gamma,
        alpha=alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        threshold=threshold,
        teacher_forcing_ratio=teacher_forcing_ratio,
        n_training_steps=n_,
        info={"div": division},
    )

    # create trainer
    trainer = get_trainer(accelerator=accelerator, n_epochs=n_epochs)

    # training
    print("\n\nTraining Model for", n_epochs, "epochs\n")
    trainer.fit(s2s, datamodule=datamodule)

    return


def get_args():
    parser = argparse.ArgumentParser(
        description="This script trains a Seq2Seq model with midi files\
            from the classical archives dataset."
    )
    parser.add_argument(
        "-p", "--path", type=str, default=PATH, help="Path to midi files"
    )
    parser.add_argument(
        "-e", "--n_epochs", type=int, default=N_EPOCHS, help="Number of epochs"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "-s", "--seq-len", type=int, default=SEQ_LEN, help="Sequence length"
    )
    parser.add_argument(
        "-l", "--num-layers", type=int, default=N_LAYERS, help="Number of layers"
    )
    parser.add_argument(
        "-c",
        "--criterion",
        type=str,
        default=CRITERION,
        choices=["bce", "focal", "focal+"],
        help="Criterion",
    )
    parser.add_argument("-g", "--gamma", type=float, default=GAMMA, help="Gamma")
    parser.add_argument("-a", "--alpha", type=float, default=ALPHA, help="Alpha")
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=THRESHOLD, help="Threshold"
    )
    parser.add_argument(
        "-tf",
        "--teacher-forcing-ratio",
        type=float,
        default=TEACHER_FORCING_RATIO,
        help="Teacher forcing ratio",
    )
    parser.add_argument(
        "-d", "--decoder-n-layers", type=int, default=N_LAYERS, help="Decoder layers"
    )
    parser.add_argument(
        "-rz", "--remove-zeros", action="store_true", default=False, help="Remove zeros"
    )
    parser.add_argument("-div", "--division", type=int, default=DIV, help="Division")
    parser.add_argument(
        "-ac",
        "--accelerator",
        type=str,
        default=ACCELERATOR,
        choices=["cpu", "gpu", "tpu"],
        help="Accelerator. Either cpu, gpu or tpu.",
    )

    return parser.parse_args()


if "__main__" == __name__:
    args = get_args()

    path = args.path
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_layers = args.num_layers
    criterion = args.criterion
    gamma = args.gamma
    alpha = args.alpha
    learning_rate = args.learning_rate
    threshold = args.threshold
    teacher_forcing_ratio = args.teacher_forcing_ratio
    remove_zeros = args.remove_zeros
    division = args.division
    accelerator = args.accelerator

    try:
        print("Loading train data\n")
        pianorolls = load_data(path, division=division)
    except FileNotFoundError:
        print(f"Training data not found at {path}")
        exit(1)

    train_seq2seq(
        pianorolls,
        n_epochs,
        batch_size,
        seq_len,
        num_layers,
        criterion,
        gamma,
        alpha,
        learning_rate,
        threshold,
        teacher_forcing_ratio,
        remove_zeros,
        division,
        accelerator,
    )
