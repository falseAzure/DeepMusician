"""
Modolue for model definition.
"""

import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils import data
from torch.utils.data import DataLoader
from utils_music21 import get_density

# import torch.optim as optim


# wandb_logger = WandbLogger(project="DeepMusician")


# input size = batch size
LAYERS = 2
LEARNING_RATE = 0.001
HIDDEN_SIZE = 512
INPUT_DIM = 88
BATCH_SIZE = 32
NUM_EPOCHS = 100
SEQ_LEN = 96
NUM_WORKERS = 8
DROP_OUT = 0.2
SOS_TOKEN = torch.zeros(1, BATCH_SIZE, 88)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_test(pianorolls, train=0.8):
    assert train < 1 and train > 0
    n = len(pianorolls)
    train_idx = int(n * train) + (n % train > 0)  # round up
    if n == train_idx:  # if there is no test set, due to insufficient data
        print("Returning the same and train and test set, due to insufficient data!")
        return (
            pianorolls[:train_idx],
            pianorolls[:train_idx],
            range(train_idx),
            range(train_idx),
        )
    return (
        pianorolls[:train_idx],
        pianorolls[train_idx:],
        range(train_idx),
        range(train_idx, n),
    )


class MidiDataset(data.Dataset):
    def __init__(self, pianorolls: list, seq_len=96, remove_zeros=False) -> None:
        tracks = []
        for pianoroll in pianorolls:

            # trim leading and trailing zeros
            nz = np.nonzero(pianoroll)[0]
            if remove_zeros:
                nz_unique = np.unique(nz)
                pianoroll = pianoroll[nz_unique]
            else:
                nz_min = nz.min()
                nz_max = nz.max()
                pianoroll = pianoroll[nz_min : nz_max + 1]

            short = seq_len - len(pianoroll) % seq_len
            track = torch.concat(
                [torch.tensor(pianoroll), torch.zeros(short, 88)], dim=0
            )
            tracks.append(track)
        tracks = torch.cat(tracks, dim=0)
        assert len(tracks) % seq_len == 0
        self.notes = tracks
        self.seq_len = seq_len

    def __getitem__(self, index):
        # Select sample
        sample = self.notes[index : (index + self.seq_len * 2)]
        sample = np.asarray(sample).astype(np.float32)
        return sample[: self.seq_len], sample[self.seq_len :]

    def __len__(self):
        return len(self.notes) - self.seq_len * 2


class MidiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        pianorolls=[],
        split=0.8,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        remove_zeros=False,
    ):
        super().__init__()
        self.pianorolls = pianorolls
        self.split = split
        self.seq_length = seq_len
        self.batch_size = batch_size
        self.remove_zeros = remove_zeros

    def setup(self, stage=None):
        train, test, _, _ = get_train_test(self.pianorolls, train=self.split)
        self.train_dataset = MidiDataset(
            train, seq_len=self.seq_length, remove_zeros=self.remove_zeros
        )
        self.test_dataset = MidiDataset(
            test, seq_len=self.seq_length, remove_zeros=self.remove_zeros
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS
        )


class EncoderRNN(pl.LightningModule):
    def __init__(
        self,
        input_size=INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=LAYERS,
        dropout=DROP_OUT,
    ):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout,
        )

    def forward(self, input_seq, hidden=None):
        # input_seq = (batch size, seq len, input dim)
        # Forward pass through GRU
        outputs, hidden = self.gru(input_seq, hidden)
        # Sum bidirectional GRU outputs
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :
        # ,self.hidden_size:]

        return outputs, hidden
        # outputs = (batch size, seq len, hidden size)
        # hidden = (num layers, seq len, hidden size)


class DecoderRNN(pl.LightningModule):
    def __init__(
        self,
        output_size=INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=LAYERS,
        batch_size=BATCH_SIZE,
        dropout=DROP_OUT,
    ):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dropout = dropout

        self.gru = nn.GRU(
            output_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=self.dropout,
        )

    def forward(self, input_step, hidden):
        # input_step = (1, batch size, input dim)
        # Forward pass through RELU
        output = F.relu(input_step)
        # Forward through unidirectional GRU
        output, hidden = self.gru(output, hidden)
        return output, hidden
        # outputs = (1, batch size, hidden size)
        # hidden = (num layers, batch size, hidden size)

    def init_hidden_zeros(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def init_hidden_random(self):
        return torch.randn(self.num_layers, self.batch_size, self.hidden_size)


class Classifier(pl.LightningModule):
    def __init__(self, output_size=INPUT_DIM, hidden_size=HIDDEN_SIZE):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x)


class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        encoder=EncoderRNN(),
        decoder=DecoderRNN(),
        classifier=Classifier(),
        criterion=nn.BCELoss(),
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        n_training_steps=None,
        teacher_forcing_ratio=0.5,
        decoder_n_layers=LAYERS,
        SOS_token=SOS_TOKEN,
        info=[],
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_training_steps = n_training_steps
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.decoder_n_layers = decoder_n_layers
        self.SOS_token = SOS_token
        self.info = info

        self.save_hyperparameters(
            ignore=["encoder", "decoder", "classifier", "criterion", "SOS_token"]
        )

        assert (
            encoder.hidden_size == decoder.hidden_size
        ), "Hidden dimensions of encoder and decoder must be equal!"

    def forward(
        self, input_seq, target_seq, seq_len=SEQ_LEN, teacher_forcing_ratio=None
    ):
        # input_seq = (batch size, seq len, input dim)
        teacher_forcing_ratio = teacher_forcing_ratio or self.teacher_forcing_ratio

        # zero padding the input sequence for the encoder.
        input_zeros = torch.zeros(input_seq.shape[0], 1, input_seq.shape[2])
        input_seq = torch.cat([input_zeros, input_seq], dim=1)

        # zero padding the target sequence for the decoder.
        # target_zeros = torch.zeros(target_seq.shape[0], 1, target_seq.shape[2])
        # target_seq = torch.cat([target_zeros, target_seq], dim=1)

        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq)
        # encoder_hidden = (num layers, seq len, hidden size)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden
        # Initialize decoder input
        decoder_input = self.SOS_token
        # initialize output vector
        output_seq = torch.zeros(seq_len, self.batch_size, self.decoder.output_size)

        for t in range(seq_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # decoder_output = (1, batch size, output dim) = 1 step
            # decoder_hidden = (num layers, seq len, hidden size)

            # run through classifier
            classifier_output = self.classifier(decoder_output)

            # add output of decoder to output_seq
            output_seq[t] = classifier_output

            # teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = (
                target_seq.transpose(0, 1)[t, :, :].unsqueeze(0)
                if teacher_force
                else classifier_output
            )

        return {"output_seq": output_seq.transpose(0, 1), "target_seq": target_seq}
        # output_seq = (seq_len, batch size, input dim)
        # transposed = (batch size, seq_len, input dim)

    def training_step(self, batch, batch_idx):
        train_source = batch[0]
        train_target = batch[1]
        output = self(train_source, train_target)
        output_seq = output["output_seq"]
        loss = self.criterion(output_seq, train_target)
        density = get_density(output_seq.detach().numpy())
        precision = get_precision(output_seq, train_target)
        recall = get_recall(output_seq, train_target)

        # print(
        #     "Epoche: ",
        #     self.current_epoch,
        #     " - Step: ",
        #     batch_idx + 1,
        #     "/",
        #     self.n_training_steps,
        #     " - Loss : ",
        #     round(loss.item(), 4),
        #     " - Density: ",
        #     round(density, 0),
        #     sep="",
        #     end="\r",
        # )
        self.log_dict(
            {
                "train_loss": loss,
                "density": density,
                "precision": precision,
                "recall": recall,
            },
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return {"loss": loss, "output_seq": output_seq, "target_seq": train_target}

    def training_epoch_end(self, outputs):
        print()
        print("Training epoch end")
        predictions = []
        targets = []
        for output in outputs:
            predictions.append(output["output_seq"])
            targets.append(output["target_seq"])
        predictions = torch.stack(predictions, dim=0).reshape(-1, 88)
        targets = torch.stack(targets, dim=0).reshape(-1, 88)
        precision = get_precision(predictions, targets)
        recall = get_recall(predictions, targets)
        print("Precision: ", precision, " - Recall: ", recall)

    def test_step(self, batch, batch_idx):
        train_source = batch[0]
        train_target = batch[1]
        output_seq = self(train_source, train_target)
        output_seq = output_seq.transpose(0, 1)
        loss = self.criterion(output_seq, train_target)
        density = get_density(output_seq)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("density", density, prog_bar=True, logger=True)
        return {"loss": loss, "output_seq": output_seq, "target_seq": train_target}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def generate_sequence(
        self,
        init_hidden="guided",
        init_seq=torch.zeros(SEQ_LEN, INPUT_DIM),  # init_seq = (seq len, input dim)
        seq_len=96,
        threshold=0.5,
    ):
        assert init_hidden in [
            "zero",
            "random",
            "guided",
        ], "init_hidden must be 'zero' 'random' or 'guided'"
        if init_hidden == "zero":
            encoder_hidden = self.decoder.init_hidden_zeros()[:, 0, :]
        elif init_hidden == "random":
            encoder_hidden = self.decoder.init_hidden_random()[:, 0, :]
        else:
            # Forward input through encoder model
            _, encoder_hidden = self.encoder(init_seq)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden
        # Initialize decoder input: one step single batch
        decoder_input = self.SOS_token[:, 0, :]
        # initialize output vector
        output_seq = torch.zeros(seq_len, self.decoder.output_size)

        for t in range(seq_len):
            # run through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # run through classifier
            classifier_output = self.classifier(decoder_output)
            output_seq[t] = classifier_output
            decoder_input = classifier_output

        output_seq = (output_seq > threshold).int()
        return output_seq
        # output_seq = (seq_len, input dim)


def get_trainer(n_epochs=10, accelerator="gpu", log_n_steps=10, test=False):
    if test:
        return pl.Trainer(
            accelerator=accelerator,
            max_epochs=n_epochs,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="train_loss",
        mode="min",
    )
    logger = TensorBoardLogger("lightning_logs", name="midi", default_hp_metric=True)

    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=log_n_steps,
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        max_epochs=n_epochs,
    )

    return trainer


def get_recall(prediction, target, threshold=0.5):
    prediction_int = (prediction > threshold).long()
    return torch.mul(prediction_int.float(), target).float().sum() / target.sum()


def get_precision(prediction, target, threshold=0.5):
    prediction_int = (prediction > threshold).long()
    return (
        torch.mul(prediction_int.float(), target).float().sum()
        / prediction_int.float().sum()
    )
