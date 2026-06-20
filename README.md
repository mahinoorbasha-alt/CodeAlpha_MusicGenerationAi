# Music Generation with AI

CodeAlpha Internship Project — a Flask web app that learns from MIDI music
using an LSTM neural network (TensorFlow/Keras + Music21) and generates new
musical sequences you can download as `.mid` files.

## Features
- Upload multiple MIDI files (`.mid` / `.midi`)
- Preprocess with Music21 (notes + chords extraction)
- Train an LSTM model with adjustable epochs / batch size
- Generate new music and download the resulting MIDI
- Live dashboard: files uploaded, notes extracted, model status, songs generated
- Modern glassmorphism UI with gradient backgrounds and smooth animations

## Folder Structure
```
music_ai/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── train.html
│   ├── generate.html
│   └── about.html
├── static/
│   ├── css/style.css
│   └── js/main.js
├── dataset/           # uploaded MIDI files
├── models/            # trained model + notes.pkl
└── generated_music/   # generated MIDI files
```

## Setup
```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

## Workflow
1. **Upload Dataset** — upload `.mid` files (classical piano works well).
2. **Train Model** — click *Preprocess*, then *Start Training* (10 epochs is a good start; more = better).
3. **Generate Music** — pick a length and click *Generate*, then download the `.mid`.

## Tech Stack
Flask · TensorFlow / Keras · Music21 · NumPy · HTML5 · CSS3 (glassmorphism) · Vanilla JS

## Notes
- First training run downloads TensorFlow weights; allow a few minutes.
- For better quality, upload 20+ MIDI files in the same style and train for 50+ epochs.
- To play `.mid` files, use VLC, GarageBand, or any DAW.

---

