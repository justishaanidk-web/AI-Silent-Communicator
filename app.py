
import os
import json
import zipfile
import tempfile

import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp

from PIL import Image
from gtts import gTTS

print("Loading AI Silent Communicator...")

# ==========================================
# 1. EXTRACT MODEL PACKAGES
# ==========================================

os.makedirs("alphabet_files", exist_ok=True)
os.makedirs("word_files", exist_ok=True)
os.makedirs("landmark_files", exist_ok=True)

def extract_zip(zip_path, destination):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing ZIP: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(destination)

# Download model ZIPs from Hugging Face
import requests

def download_model(url, filename):
    response = requests.get(url, timeout=300)
    response.raise_for_status()

    with open(filename, "wb") as f:
        f.write(response.content)

download_model(
    "https://huggingface.co/spaces/justishaanidk/AI-Silent-Communicator/resolve/main/AI_Silent_Communicator_Website_Model.zip",
    "AI_Silent_Communicator_Website_Model.zip"
)

download_model(
    "https://huggingface.co/spaces/justishaanidk/AI-Silent-Communicator/resolve/main/ISL_Hand_Augmented_Backup.zip",
    "ISL_Hand_Augmented_Backup.zip"
)

extract_zip(
    "AI_Silent_Communicator_Website_Model.zip",
    "alphabet_files"
)

extract_zip(
    "ISL_Hand_Augmented_Backup.zip",
    "landmark_files"
)
# ==========================================
# 2. LOAD ALPHABET MODEL
# ==========================================

def find_file(folder, filename):
    for root, dirs, files in os.walk(folder):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(filename + " not found.")

alphabet_path = find_file(
    "alphabet_files", "isl_model.keras"
)

alphabet_labels_path = find_file(
    "alphabet_files", "class_names.json"
)

alphabet_model = tf.keras.models.load_model(alphabet_path)

with open(alphabet_labels_path) as f:
    alphabet_labels = json.load(f)

print("Alphabet model loaded!")

# ==========================================
# 3. LOAD NEW LANDMARK WORD MODEL
# ==========================================

word_path = find_file(
    "landmark_files",
    "isl_hand_augmented_best.keras"
)

word_labels_path = find_file(
    "landmark_files",
    "isl_hand_augmented_labels.json"
)

word_model = tf.keras.models.load_model(word_path)

with open(word_labels_path) as f:
    word_labels = json.load(f)

if tuple(word_model.input_shape[1:]) != (30, 2, 21, 3):
    raise ValueError(
        f"Unexpected word model input shape: {word_model.input_shape}"
    )

if word_model.output_shape[-1] != len(word_labels):
    raise ValueError("Word model and label count do not match.")

print("Landmark word model loaded!")
print("Supported words:", word_labels)

# ==========================================
# 4. ALPHABET RECOGNITION
# ==========================================

def predict_alphabet(image):

    if image is None:
        return "Please upload an image.", ""

    image = image.convert("RGB").resize((128, 128))

    array = np.array(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    predictions = alphabet_model.predict(
        array, verbose=0
    )[0]

    index = int(np.argmax(predictions))

    letter = alphabet_labels[index]
    confidence = float(predictions[index]) * 100

    return letter, f"{confidence:.2f}%"

# ==========================================
# 5. MEDIAPIPE LANDMARK EXTRACTION
# ==========================================

def extract_landmarks(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < 2:
        cap.release()
        raise ValueError("Video is too short.")

    indices = np.linspace(
        0, total - 1, 30
    ).astype(int)

    sequence = []

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:

        for index in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

            success, frame = cap.read()

            if not success:
                cap.release()
                raise ValueError("Could not read video frame.")

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            result = hands.process(rgb)

            frame_landmarks = np.zeros(
                (2, 21, 3), dtype=np.float32
            )

            if result.multi_hand_landmarks:

                for hand_index, hand in enumerate(
                    result.multi_hand_landmarks[:2]
                ):

                    for point_index, point in enumerate(
                        hand.landmark
                    ):

                        frame_landmarks[
                            hand_index, point_index
                        ] = [
                            point.x,
                            point.y,
                            point.z
                        ]

            sequence.append(frame_landmarks)

    cap.release()

    return np.asarray(sequence, dtype=np.float32)

# ==========================================
# 6. TEXT TO SPEECH
# ==========================================

def create_speech(text):

    if not text.strip():
        return None

    output_path = os.path.join(
        tempfile.gettempdir(),
        "isl_sentence.mp3"
    )

    try:
        gTTS(text=text, lang="en").save(output_path)
        return output_path
    except Exception as error:
        print("Speech error:", error)
        return None

# ==========================================
# 7. AUTOMATIC WORD + SENTENCE RECOGNITION
# ==========================================

def predict_word(video, current_sentence):

    if video is None:
        return (
            "Please record a video.",
            "",
            current_sentence,
            None,
            "No video received."
        )

    try:

        sequence = extract_landmarks(video)
        sequence = np.expand_dims(sequence, axis=0)

        predictions = word_model.predict(
            sequence, verbose=0
        )[0]

        index = int(np.argmax(predictions))

        word = word_labels[index]
        confidence = float(predictions[index]) * 100

        # Automatically append recognized word
        updated_sentence = (
            current_sentence.strip() + " " + word
        ).strip()

        # Automatically generate speech
        audio = create_speech(updated_sentence)

        status = (
            "Word added automatically."
            if audio else
            "Word added. Speech generation unavailable."
        )

        return (
            word,
            f"{confidence:.2f}%",
            updated_sentence,
            audio,
            status
        )

    except Exception as error:

        return (
            "Recognition error",
            "",
            current_sentence,
            None,
            str(error)
        )

# ==========================================
# 8. CLEAR SENTENCE
# ==========================================

def clear_sentence():
    return "", None, "Sentence cleared."

# ==========================================
# 9. WEBSITE INTERFACE
# ==========================================

with gr.Blocks(
    title="AI Silent Communicator"
) as demo:

    gr.Markdown("# 🤟 AI Silent Communicator")

    gr.Markdown(
        "Indian Sign Language Recognition and Communication"
    )

    sentence_state = gr.State("")

    with gr.Tab("Alphabet Recognition"):

        gr.Markdown("### Recognize ISL alphabet signs A–Z")

        image_input = gr.Image(
            type="pil",
            label="Upload alphabet sign"
        )

        alphabet_button = gr.Button(
            "Recognize Alphabet"
        )

        alphabet_result = gr.Textbox(
            label="Predicted Letter"
        )

        alphabet_confidence = gr.Textbox(
            label="Confidence"
        )

        alphabet_button.click(
            predict_alphabet,
            inputs=image_input,
            outputs=[
                alphabet_result,
                alphabet_confidence
            ]
        )

    with gr.Tab("Word Recognition"):

        gr.Markdown("### Recognize ISL words")

        gr.Markdown(
            "Supported words: Friend, Hello, Market, School"
        )

        video_input = gr.Video(
            sources=["webcam", "upload"],
            label="Record or upload ISL sign video"
        )

        word_button = gr.Button(
            "Recognize Word"
        )

        word_result = gr.Textbox(
            label="Predicted Word"
        )

        word_confidence = gr.Textbox(
            label="Confidence"
        )

        sentence_display = gr.Textbox(
            label="Automatically Built Sentence",
            lines=3
        )

        speech_output = gr.Audio(
            label="Automatic Speech Output",
            type="filepath",
            autoplay=True
        )

        recognition_status = gr.Textbox(
            label="Status"
        )

        word_button.click(
            predict_word,
            inputs=[
                video_input,
                sentence_state
            ],
            outputs=[
                word_result,
                word_confidence,
                sentence_state,
                speech_output,
                recognition_status
            ]
        ).then(
            lambda s: s,
            inputs=sentence_state,
            outputs=sentence_display
        )

        clear_button = gr.Button("Clear Sentence")

        clear_button.click(
            clear_sentence,
            outputs=[
                sentence_state,
                speech_output,
                recognition_status
            ]
        ).then(
            lambda: "",
            outputs=sentence_display
        )

    with gr.Tab("Sentence Builder"):

        gr.Markdown(
            "### Your automatically generated sentence"
        )

        gr.Markdown(
            "Recognized words are added automatically "
            "in the Word Recognition tab."
        )

        sentence_view = gr.Textbox(
            label="Your Sentence",
            lines=4
        )

        demo.load(
            lambda s: s,
            inputs=sentence_state,
            outputs=sentence_view
        )

    gr.Markdown(
        "### AI Silent Communicator | ISL Project"
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)

