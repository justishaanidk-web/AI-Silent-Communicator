import os
import json
import zipfile
import tempfile
import time
from collections import deque, Counter

import requests
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from gtts import gTTS

# =========================================================
# MODEL DOWNLOAD + EXTRACTION (models stay in HF Space)
# =========================================================

os.makedirs("alphabet_files", exist_ok=True)
os.makedirs("landmark_files", exist_ok=True)

def download_model(url, filename):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)

def extract_zip(zip_path, destination):
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)

download_model(
    "https://huggingface.co/spaces/justishaanidk/AI-Silent-Communicator/resolve/main/AI_Silent_Communicator_Website_Model.zip",
    "AI_Silent_Communicator_Website_Model.zip"
)
download_model(
    "https://huggingface.co/spaces/justishaanidk/AI-Silent-Communicator/resolve/main/ISL_Hand_Augmented_Backup.zip",
    "ISL_Hand_Augmented_Backup.zip"
)

def find_file(folder, filename):
    for root, _, files in os.walk(folder):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found inside {folder}")

def has_file(folder, filename):
    return any(filename in files for _, _, files in os.walk(folder))

if not has_file("alphabet_files", "isl_model.keras"):
    extract_zip("AI_Silent_Communicator_Website_Model.zip", "alphabet_files")
if not has_file("landmark_files", "isl_hand_augmented_best.keras"):
    extract_zip("ISL_Hand_Augmented_Backup.zip", "landmark_files")

print("Loading alphabet model...")
alphabet_model = tf.keras.models.load_model(
    find_file("alphabet_files", "isl_model.keras")
)
with open(find_file("alphabet_files", "class_names.json"), "r") as f:
    alphabet_labels = json.load(f)
print("Alphabet model loaded:", alphabet_labels)

print("Loading word model...")
word_model = tf.keras.models.load_model(
    find_file("landmark_files", "isl_hand_augmented_best.keras")
)
with open(find_file("landmark_files", "isl_hand_augmented_labels.json"), "r") as f:
    word_labels = json.load(f)
if isinstance(word_labels, dict):
    word_labels = [word_labels[str(i)] for i in range(len(word_labels))]
print("Word model loaded:", word_labels)

if tuple(word_model.input_shape[1:]) != (30, 2, 21, 3):
    raise ValueError(f"Unexpected word model input shape: {word_model.input_shape}")
if word_model.output_shape[-1] != len(word_labels):
    raise ValueError("Word model output count does not match word labels.")

# =========================================================
# SHARED HELPERS
# =========================================================

def make_audio(text):
    if not text.strip():
        return None
    path = os.path.join(tempfile.gettempdir(), "isl_sentence.mp3")
    try:
        gTTS(text=text, lang="en").save(path)
        return path
    except Exception as exc:
        print("TTS error:", exc)
        return None

def draw_text(frame, text, color=(20, 20, 20)):
    if frame is None:
        return None
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 58), color, -1)
    cv2.putText(
        image, text[:55], (12, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
        cv2.LINE_AA
    )
    return image

# =========================================================
# LIVE ALPHABET WEBCAM
# =========================================================

def new_alphabet_state():
    return {"history": [], "last_letter": "", "last_commit": 0.0}

def live_alphabet(frame, state):
    state = state or new_alphabet_state()
    if frame is None:
        return None, "Start the webcam.", "—", state

    # Gradio webcam frames are RGB; the model was trained on RGB images.
    resized = cv2.resize(frame, (128, 128))
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    probabilities = alphabet_model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    letter = str(alphabet_labels[index])
    confidence = float(probabilities[index])

    state["history"].append((letter, confidence))
    state["history"] = state["history"][-7:]

    votes = [label for label, score in state["history"] if score >= 0.60]
    stable_letter = Counter(votes).most_common(1)[0][0] if votes else ""
    stable_count = votes.count(stable_letter) if stable_letter else 0

    if confidence < 0.60 or stable_count < 5:
        display = "Hold your hand sign steady..."
        conf_text = f"{confidence:.1%} (not stable yet)"
    else:
        display = stable_letter
        conf_text = f"{confidence:.1%}"

    annotated = draw_text(frame, f"Alphabet: {display}")
    return annotated, display, conf_text, state

# =========================================================
# LIVE WORD WEBCAM — 30 x 2 x 21 x 3 LANDMARK SEQUENCE
# =========================================================

hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)

def new_word_state():
    return {
        "frames": [],
        "prediction_history": [],
        "sentence": "",
        "last_committed": "",
        "last_commit_time": 0.0,
        "frame_count": 0
    }

def frame_to_landmarks(frame):
    # Gradio supplies webcam frames as RGB, which MediaPipe expects.
    result = hands.process(frame)
    landmarks = np.zeros((2, 21, 3), dtype=np.float32)
    if result.multi_hand_landmarks:
        for hand_index, hand in enumerate(result.multi_hand_landmarks[:2]):
            for point_index, point in enumerate(hand.landmark):
                landmarks[hand_index, point_index] = [point.x, point.y, point.z]
    return landmarks

def live_word(frame, state):
    state = state or new_word_state()
    if frame is None:
        return None, "Start the webcam.", "—", state["sentence"], None, "Camera waiting.", state

    landmarks = frame_to_landmarks(frame)
    state["frames"].append(landmarks)
    state["frames"] = state["frames"][-30:]
    state["frame_count"] += 1

    if not np.any(landmarks):
        state["prediction_history"] = []
        annotated = draw_text(frame, "Show one clear ISL sign")
        return annotated, "No hand detected", "—", state["sentence"], None, "Keep your hand fully visible.", state

    # Predict every 5 streamed frames. Pad early frames to the model's 30-frame input.
    if len(state["frames"]) < 10 or state["frame_count"] % 5 != 0:
        annotated = draw_text(frame, "Word: collecting sign...")
        return annotated, "Hold sign...", "—", state["sentence"], None, "Keep the sign steady.", state

    sequence = np.asarray(state["frames"], dtype=np.float32)
    if len(sequence) < 30:
        pad = np.repeat(sequence[-1][None, ...], 30 - len(sequence), axis=0)
        sequence = np.concatenate([sequence, pad], axis=0)
    else:
        indices = np.linspace(0, len(sequence) - 1, 30).astype(int)
        sequence = sequence[indices]

    probabilities = word_model.predict(sequence[None, ...], verbose=0)[0]
    index = int(np.argmax(probabilities))
    word = str(word_labels[index])
    confidence = float(probabilities[index])

    state["prediction_history"].append((word, confidence))
    state["prediction_history"] = state["prediction_history"][-3:]

    stable_words = [
        label for label, score in state["prediction_history"] if score >= 0.75
    ]
    stable_word = Counter(stable_words).most_common(1)[0][0] if stable_words else ""
    stable_count = stable_words.count(stable_word) if stable_word else 0

    audio = None
    status = f"Model confidence: {confidence:.1%}"
    display_word = word if confidence >= 0.75 else "Not confident — hold sign"

    # Commit only when the same high-confidence word appears repeatedly.
    now = time.time()
    if confidence >= 0.75 and stable_count >= 3:
        if stable_word != state["last_committed"] and now - state["last_commit_time"] >= 2.0:
            state["sentence"] = (state["sentence"] + " " + stable_word).strip()
            state["last_committed"] = stable_word
            state["last_commit_time"] = now
            audio = make_audio(state["sentence"])
            status = f"Added '{stable_word}' to sentence."
        else:
            status = "Stable sign detected. Change to another sign to add the next word."
    elif confidence < 0.75:
        state["last_committed"] = ""
        status = "Low confidence. Keep the hand centered and sign clearly."

    annotated = draw_text(frame, f"Word: {display_word}  ({confidence:.0%})")
    return annotated, display_word, f"{confidence:.1%}", state["sentence"], audio, status, state

def clear_sentence():
    return "", None, "Sentence cleared.", new_word_state()

# =========================================================
# WEBSITE — WEBCAM ONLY (NO IMAGE UPLOAD / NO VIDEO RECORDING)
# =========================================================

CSS = """
.gradio-container {max-width: 1200px !important; margin: auto !important;}
"""

with gr.Blocks(title="AI Silent Communicator", css=CSS) as demo:
    gr.Markdown("# 🤟 AI Silent Communicator")
    gr.Markdown("Indian Sign Language recognition — live webcam, text and speech.")

    with gr.Tabs():
        with gr.Tab("Alphabet Recognition"):
            gr.Markdown("Show an ISL alphabet sign to your webcam. Prediction updates live.")
            with gr.Row():
                alphabet_camera = gr.Image(
                    sources=["webcam"], type="numpy", streaming=True,
                    label="Live Alphabet Camera", height=420
                )
                with gr.Column():
                    alphabet_preview = gr.Image(label="Live Prediction View", interactive=False, height=360)
                    alphabet_result = gr.Textbox(label="Recognized Letter", interactive=False)
                    alphabet_confidence = gr.Textbox(label="Confidence", interactive=False)
            alphabet_state = gr.State(new_alphabet_state())
            alphabet_camera.stream(
                live_alphabet,
                inputs=[alphabet_camera, alphabet_state],
                outputs=[alphabet_preview, alphabet_result, alphabet_confidence, alphabet_state],
                stream_every=0.3,
                concurrency_limit=1,
                show_progress="hidden"
            )

        with gr.Tab("Word Recognition"):
            gr.Markdown(
                "Show one trained word sign to the webcam. Keep it steady; "
                "a word is added only after repeated high-confidence predictions."
            )
            with gr.Row():
                word_camera = gr.Image(
                    sources=["webcam"], type="numpy", streaming=True,
                    label="Live Word Camera", height=420
                )
                with gr.Column():
                    word_preview = gr.Image(label="Live Prediction View", interactive=False, height=360)
                    word_result = gr.Textbox(label="Recognized Word", interactive=False)
                    word_confidence = gr.Textbox(label="Confidence", interactive=False)
                    sentence_display = gr.Textbox(
                        label="Automatically Built Sentence", lines=3, interactive=False
                    )
                    speech_output = gr.Audio(
                        label="Automatic Sentence Speech", type="filepath", autoplay=True
                    )
                    status_output = gr.Textbox(label="Status", interactive=False)
                    clear_button = gr.Button("Clear Sentence")
            word_state = gr.State(new_word_state())
            word_camera.stream(
                live_word,
                inputs=[word_camera, word_state],
                outputs=[
                    word_preview, word_result, word_confidence, sentence_display,
                    speech_output, status_output, word_state
                ],
                stream_every=0.3,
                concurrency_limit=1,
                show_progress="hidden"
            )
            clear_button.click(
                clear_sentence,
                inputs=[],
                outputs=[sentence_display, speech_output, status_output, word_state]
            )

    gr.Markdown("### AI Silent Communicator | ISL Project")

PORT = int(os.environ.get("PORT", 10000))
demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=PORT)
