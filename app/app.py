import base64
import io

from flask import Flask, render_template, request

from deepmusician.seq2seq import Seq2Seq
from deepmusician.utils_app import pianoroll_to_midi

app = Flask(__name__)
model_file = "../playaround/checkpoints/epoch=2-train_loss=0.0000006649.ckpt"
midi_file = "test.mid"
model = Seq2Seq(batch_size=32)
model.load_from_checkpoint(model_file)


@app.route("/")
def home():
    return render_template("index.html")

# route to the generate page
@app.route("/generate", methods=["POST"])
def generate():
    """
    For rendering results on HTML GUI
    """

    # get the sequence length from the form
    int_features = [int(x) for x in request.form.values()]
    seq_len = int_features[0]
    # generate the sequence
    seq = model.generate_sequence(seq_len=seq_len)

    # fig = plot_pianoroll(seq, division=1 / 4, return_plot=True)
    # img = fig_to_base64(fig)

    # convert the sequence to midi
    mf = pianoroll_to_midi(seq)
    # convert the midi to base64 so that it can be rendered in the browser
    midi = io.BytesIO()
    mf.writeFile(midi)
    midi.seek(0)
    midi = base64.b64encode(midi.getvalue()).decode("utf-8")

    # return the generated sequence as a midi file encoded in base64
    return render_template(
        "index.html",
        midi=midi,
        # generation_text=f"Your generated sequence of length {seq_len}:",
        # img=img,
    )


if __name__ == "__main__":
    app.run(debug=True)
