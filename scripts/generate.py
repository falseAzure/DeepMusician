import argparse

from deepmusician.seq2seq import Seq2Seq
from deepmusician.utils_music21 import pianoroll_to_midi, save_midi

SEQ_LEN = 192


def get_args():
    parser = argparse.ArgumentParser(
        description="This script generates a sequence from a trained model\
            and saves it as a midi file from the classical archives dataset."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "-l", "--seq-len", type=int, default=SEQ_LEN, help="Sequence length"
    )

    parser.add_argument(
        "-d", "--division", type=float, default=1 / 4, help="Division of a beat"
    )

    parser.add_argument(
        "-i",
        "--init_hidden",
        type=str,
        default="guided",
        choices=["zero", "random", "guided"],
        help="Initial hidden state",
    )

    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default="generated_sequence.mid",
        help="Path to save generated sequence",
    )

    return parser.parse_args()


if "__main__" == __name__:
    args = get_args()
    checkpoint_path = args.checkpoint
    seq_len = args.seq_len
    init_hidden = args.init_hidden
    save_patch = args.save
    division = args.division

    print("Loading model from checkpoint\n")
    seq2seq = Seq2Seq()
    seq2seq.load_from_checkpoint(checkpoint_path)
    seq = seq2seq.generate_sequence(init_hidden=init_hidden, seq_len=args.seq_len)
    midi = pianoroll_to_midi(seq, division=division)
    save_midi(midi, save_patch)
