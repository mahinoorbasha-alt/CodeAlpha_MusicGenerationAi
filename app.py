"""
Music Generation with AI
========================
Flask backend for uploading MIDI datasets, preprocessing with Music21,
training an LSTM model (TensorFlow/Keras), and generating new MIDI music.

CodeAlpha Internship Submission.
"""

import os
import glob
import pickle
import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, redirect, url_for, flash
)
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
GENERATED_DIR = os.path.join(BASE_DIR, "generated_music")
NOTES_FILE = os.path.join(MODELS_DIR, "notes.pkl")
MODEL_FILE = os.path.join(MODELS_DIR, "music_lstm.keras")

for d in (DATASET_DIR, MODELS_DIR, GENERATED_DIR):
    os.makedirs(d, exist_ok=True)

ALLOWED_EXT = {".mid", ".midi"}
SEQUENCE_LENGTH = 50

app = Flask(__name__)
app.secret_key = "codealpha-music-ai"
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB upload cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_midi_files():
    """Return all MIDI files currently stored in the dataset folder."""
    files = []
    for ext in ("*.mid", "*.midi"):
        files.extend(glob.glob(os.path.join(DATASET_DIR, ext)))
    return sorted(files)


def list_generated_files():
    """Return all generated MIDI files."""
    return sorted(glob.glob(os.path.join(GENERATED_DIR, "*.mid")))


def get_stats():
    """Build dashboard stats dictionary."""
    notes_count = 0
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "rb") as f:
            notes_count = len(pickle.load(f))
    return {
        "uploaded": len(list_midi_files()),
        "notes": notes_count,
        "model_ready": os.path.exists(MODEL_FILE),
        "generated": len(list_generated_files()),
    }


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", stats=get_stats())


@app.route("/upload")
def upload_page():
    return render_template(
        "upload.html",
        stats=get_stats(),
        files=[os.path.basename(f) for f in list_midi_files()],
    )


@app.route("/train")
def train_page():
    return render_template("train.html", stats=get_stats())


@app.route("/generate")
def generate_page():
    return render_template(
        "generate.html",
        stats=get_stats(),
        files=[os.path.basename(f) for f in list_generated_files()],
    )


@app.route("/about")
def about_page():
    return render_template("about.html", stats=get_stats())


# ---------------------------------------------------------------------------
# Routes — API: Upload
# ---------------------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accept multiple MIDI files and store them in the dataset folder."""
    saved = []
    skipped = []
    for f in request.files.getlist("files"):
        name = secure_filename(f.filename or "")
        ext = os.path.splitext(name)[1].lower()
        if not name or ext not in ALLOWED_EXT:
            skipped.append(f.filename)
            continue
        f.save(os.path.join(DATASET_DIR, name))
        saved.append(name)
    return jsonify({"saved": saved, "skipped": skipped, "stats": get_stats()})


# ---------------------------------------------------------------------------
# Routes — API: Preprocess
# ---------------------------------------------------------------------------
@app.route("/api/preprocess", methods=["POST"])
def api_preprocess():
    """Extract notes & chords from all uploaded MIDI files using Music21."""
    from music21 import converter, instrument, note, chord

    midi_files = list_midi_files()
    if not midi_files:
        return jsonify({"error": "No MIDI files uploaded yet."}), 400

    notes = []
    processed = 0
    for path in midi_files:
        try:
            midi = converter.parse(path)
            parts = instrument.partitionByInstrument(midi)
            elements = parts.parts[0].recurse() if parts else midi.flat.notes
            for el in elements:
                if isinstance(el, note.Note):
                    notes.append(str(el.pitch))
                elif isinstance(el, chord.Chord):
                    notes.append(".".join(str(n) for n in el.normalOrder))
            processed += 1
        except Exception as e:
            print(f"[preprocess] Skipped {path}: {e}")

    with open(NOTES_FILE, "wb") as f:
        pickle.dump(notes, f)

    return jsonify({
        "processed_files": processed,
        "total_notes": len(notes),
        "unique_notes": len(set(notes)),
        "stats": get_stats(),
    })


# ---------------------------------------------------------------------------
# Routes — API: Train
# ---------------------------------------------------------------------------
@app.route("/api/train", methods=["POST"])
def api_train():
    """Build & train an LSTM model on the extracted notes."""
    if not os.path.exists(NOTES_FILE):
        return jsonify({"error": "Run preprocessing first."}), 400

    epochs = int(request.json.get("epochs", 10)) if request.is_json else 10
    batch_size = int(request.json.get("batch_size", 64)) if request.is_json else 64

    with open(NOTES_FILE, "rb") as f:
        notes = pickle.load(f)

    if len(notes) < SEQUENCE_LENGTH + 1:
        return jsonify({"error": "Not enough notes to train. Upload more MIDI files."}), 400

    # Lazy import — keeps server fast to boot
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
    from tensorflow.keras.utils import to_categorical

    pitchnames = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitchnames)}
    n_vocab = len(pitchnames)

    network_input, network_output = [], []
    for i in range(len(notes) - SEQUENCE_LENGTH):
        seq_in = notes[i:i + SEQUENCE_LENGTH]
        seq_out = notes[i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[c] for c in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    X = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1)) / float(n_vocab)
    y = to_categorical(network_output, num_classes=n_vocab)

    model = Sequential([
        LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(128),
        Dropout(0.3),
        Dense(n_vocab),
        Activation("softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)

    model.save(MODEL_FILE)
    with open(os.path.join(MODELS_DIR, "pitchnames.pkl"), "wb") as f:
        pickle.dump(pitchnames, f)

    return jsonify({
        "epochs": epochs,
        "loss": [float(x) for x in history.history["loss"]],
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "vocab": n_vocab,
        "patterns": n_patterns,
        "stats": get_stats(),
    })


# ---------------------------------------------------------------------------
# Routes — API: Generate
# ---------------------------------------------------------------------------
@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Use the trained model to generate a new MIDI sequence."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(NOTES_FILE):
        return jsonify({"error": "Train the model first."}), 400

    length = int(request.json.get("length", 200)) if request.is_json else 200
    from tensorflow.keras.models import load_model
    from music21 import instrument, note, chord, stream

    with open(NOTES_FILE, "rb") as f:
        notes = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "pitchnames.pkl"), "rb") as f:
        pitchnames = pickle.load(f)

    n_vocab = len(pitchnames)
    int_to_note = {i: n for i, n in enumerate(pitchnames)}
    note_to_int = {n: i for i, n in enumerate(pitchnames)}

    # seed a sequence
    start_idx = np.random.randint(0, max(1, len(notes) - SEQUENCE_LENGTH - 1))
    pattern = [note_to_int[n] for n in notes[start_idx:start_idx + SEQUENCE_LENGTH]]

    model = load_model(MODEL_FILE)
    prediction_output = []
    for _ in range(length):
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        pred = model.predict(x, verbose=0)
        index = int(np.argmax(pred))
        prediction_output.append(int_to_note[index])
        pattern.append(index)
        pattern = pattern[1:]

    # Convert to MIDI
    offset = 0
    output_notes = []
    for pat in prediction_output:
        if ("." in pat) or pat.isdigit():
            notes_in_chord = pat.split(".")
            chord_notes = []
            for cn in notes_in_chord:
                n = note.Note(int(cn))
                n.storedInstrument = instrument.Piano()
                chord_notes.append(n)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pat)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    import time
    filename = f"generated_{int(time.time())}.mid"
    out_path = os.path.join(GENERATED_DIR, filename)
    midi_stream.write("midi", fp=out_path)

    return jsonify({"file": filename, "notes": len(prediction_output), "stats": get_stats()})


@app.route("/download/<path:filename>")
def download_generated(filename):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
