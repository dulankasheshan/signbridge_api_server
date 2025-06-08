import asyncio
import websockets
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import os
from datetime import datetime
from collections import deque
from typing import Dict, List

# Configuration
CONFIDENCE_THRESHOLD = 0.8
SEQUENCE_LENGTH = 30
LANDMARK_SIZE = 1467
MIN_CONSECUTIVE_FRAMES = 3
SMOOTHING_WINDOW = 5

# Global dictionary to store loaded gestures
gesture_db: Dict[str, List[dict]] = {}

class SignLanguageServer:
    def __init__(self):
        self.model = None
        self.label_map = {}
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.current_gesture = None
        self.consecutive_count = 0
        self.load_resources()
        self.load_gestures_from_folder()

    def load_resources(self):
        """Load model and label map with validation"""
        try:
            self.model = tf.keras.models.load_model("sign_language_model.h5")
            if os.path.exists("label_map.npy"):
                self.label_map = np.load("label_map.npy", allow_pickle=True).item()
            print(f"Model loaded. Available gestures: {list(self.label_map.values())}")
        except Exception as e:
            print(f"Loading error: {e}")
            exit(1)

    def load_gestures_from_folder(self, folder_path: str = "gestures_animations") -> None:
        """Load all JSON gesture files from specified folder"""
        global gesture_db
        gesture_db = {}

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created gestures directory: {folder_path}")
            return

        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                try:
                    gesture_name = os.path.splitext(filename)[0]
                    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                        gesture_db[gesture_name] = json.load(f)
                    print(f"Loaded gesture: {gesture_name}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        print(f"Total gestures loaded: {len(gesture_db)}")

    def extract_landmarks(self, results):
        """Extract landmarks from MediaPipe results"""
        default_landmark = (0.0, 0.0, 0.0)

        # Face landmarks (468)
        face_landmarks = [default_landmark] * 468
        if results.face_landmarks:
            face_landmarks = [(lm.x, lm.y, lm.z) for lm in results.face_landmarks.landmark]

        # Left hand landmarks (21)
        left_hand_landmarks = [default_landmark] * 21
        if results.left_hand_landmarks:
            left_hand_landmarks = [(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark]

        # Right hand landmarks (21)
        right_hand_landmarks = [default_landmark] * 21
        if results.right_hand_landmarks:
            right_hand_landmarks = [(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark]

        # Pose landmarks (33)
        pose_landmarks = [default_landmark] * 33
        if results.pose_landmarks:
            pose_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

        # Combine and flatten
        landmarks = face_landmarks + left_hand_landmarks + right_hand_landmarks + pose_landmarks
        return np.array(landmarks).flatten()[:LANDMARK_SIZE]

    def normalize_landmarks(self, landmarks):
        """Normalize landmarks to 0-1 range"""
        landmarks = np.array(landmarks)
        landmarks = landmarks.reshape(-1, 3)
        landmarks = (landmarks - np.min(landmarks, axis=0)) / (np.max(landmarks, axis=0) - np.min(landmarks, axis=0))
        return landmarks.flatten()

    async def handle_sign_to_text(self, websocket):
        """Handle sign-to-text conversion"""
        print("Sign-to-text client connected")
        landmark_sequence = []
        sentence = []
        last_predicted_word = None

        try:
            async for message in websocket:
                try:
                    # Decode image
                    frame = cv2.imdecode(np.frombuffer(message, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        await websocket.send(json.dumps({"error": "Invalid frame"}))
                        continue

                    # Process frame with MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.holistic.process(rgb_frame)

                    # Extract and normalize landmarks
                    landmarks = self.extract_landmarks(results)
                    if landmarks.size > 0 and np.max(landmarks) > 0:
                        landmarks = self.normalize_landmarks(landmarks)

                        # Update sequence
                        landmark_sequence.append(landmarks)
                        if len(landmark_sequence) > SEQUENCE_LENGTH:
                            landmark_sequence.pop(0)

                        # Predict when we have enough frames
                        if len(landmark_sequence) == SEQUENCE_LENGTH:
                            sequence_array = np.array(landmark_sequence).reshape(1, SEQUENCE_LENGTH, LANDMARK_SIZE)
                            prediction = self.model.predict(sequence_array, verbose=0)
                            predicted_idx = np.argmax(prediction)
                            confidence = np.max(prediction)

                            # Update prediction history for smoothing
                            self.prediction_history.append(predicted_idx)
                            smoothed_idx = max(set(self.prediction_history), key=self.prediction_history.count)
                            smoothed_confidence = confidence * (
                                    self.prediction_history.count(smoothed_idx) / len(self.prediction_history))

                            if smoothed_confidence >= CONFIDENCE_THRESHOLD:
                                predicted_label = self.label_map.get(smoothed_idx, str(smoothed_idx))

                                # Check for gesture change (with consecutive confirmation)
                                if predicted_label != self.current_gesture:
                                    self.consecutive_count += 1
                                    if self.consecutive_count >= MIN_CONSECUTIVE_FRAMES:
                                        self.current_gesture = predicted_label
                                        self.consecutive_count = 0
                                        print(f"New sign: {self.current_gesture} ({smoothed_confidence:.2f})")

                                        # Update sentence if different
                                        if predicted_label != last_predicted_word:
                                            last_predicted_word = predicted_label
                                            sentence.append(predicted_label)
                                else:
                                    self.consecutive_count = 0

                                response = {
                                    "gesture": self.current_gesture,
                                    "confidence": float(smoothed_confidence),
                                    "sentence": " ".join(sentence)
                                }
                            else:
                                response = {"gesture": "Low Confidence"}
                        else:
                            response = {"status": f"Collecting frames ({len(landmark_sequence)}/{SEQUENCE_LENGTH})"}
                    else:
                        response = {"error": "No valid landmarks"}

                    await websocket.send(json.dumps(response))

                except Exception as e:
                    print(f"Processing error: {e}")
                    continue

        except websockets.exceptions.ConnectionClosed:
            print("Sign-to-text client disconnected")
        finally:
            # Reset for next connection
            self.current_gesture = None
            self.consecutive_count = 0
            self.prediction_history.clear()

    async def handle_text_to_sign(self, websocket):
        """Handle text-to-sign conversion"""
        print("Text-to-sign client connected")
        try:
            # Receive text to animate
            text = await websocket.recv()
            print(f"Received request to animate: {text}")

            # Split into words and collect matching gestures
            words = text.split()
            animation_frames: List[dict] = []
            word_boundaries = []  # Track which frames belong to which word
            total_frames = 0  # Calculate total expected frames upfront

            # First calculate total frames we'll be sending
            for word in words:
                if word in gesture_db:
                    total_frames += len(gesture_db[word])

            for word in words:
                if word in gesture_db:
                    start_frame = len(animation_frames)
                    animation_frames.extend(gesture_db[word])
                    end_frame = len(animation_frames) - 1
                    word_boundaries.append({
                        'word': word,
                        'start': start_frame,
                        'end': end_frame
                    })
                else:
                    print(f"No gesture found for word: {word}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"No gesture found for: {word}"
                    }))
                    continue

            if not animation_frames:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "No matching gestures found for the input text"
                }))
                return

            # Send metadata first
            await websocket.send(json.dumps({
                "type": "metadata",
                "total_frames": total_frames,
                "fps": 30,
                "word_boundaries": word_boundaries
            }))

            # Stream frames with rate control
            for frame_idx, frame_data in enumerate(animation_frames):
                try:
                    # Find current word
                    current_word = None
                    for boundary in word_boundaries:
                        if boundary['start'] <= frame_idx <= boundary['end']:
                            current_word = boundary['word']
                            break

                    await websocket.send(json.dumps({
                        "type": "frame",
                        "data": frame_data,
                        "index": frame_idx,
                        "current_word": current_word
                    }))
                    await asyncio.sleep(1 / 30)  # Maintain ~30fps

                except websockets.exceptions.ConnectionClosed:
                    print("Client disconnected during streaming")
                    return
                except Exception as e:
                    print(f"Error sending frame {frame_idx}: {e}")
                    break

            await websocket.send(json.dumps({"type": "end"}))

        except websockets.exceptions.ConnectionClosed:
            print("Text-to-sign client disconnected")
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            await websocket.close()

    async def handle_connection(self, websocket, path):
        """Route connections to appropriate handler based on path"""
        if path == '/sign-to-text':
            await self.handle_sign_to_text(websocket)
        elif path == '/text-to-sign':
            await self.handle_text_to_sign(websocket)
        else:
            print(f"Unknown path: {path}")
            await websocket.close()

async def start_server():
    server = SignLanguageServer()
    # Use environment variables for host and port to support Railway
    host = os.getenv("HOST", "0.0.0.0")  # Listen on all interfaces
    port = int(os.getenv("PORT", 8765))   # Use Railway's assigned port or default
    async with websockets.serve(
            server.handle_connection,
            host,
            port,
            ping_interval=30,
            max_size=10 * 1024 * 1024  # 10MB
    ):
        print(f"Server running on ws://{host}:{port}")
        print("Available endpoints:")
        print(" - /sign-to-text (for sign language recognition)")
        print(" - /text-to-sign (for gesture animation)")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    # Performance optimizations
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Start server
    asyncio.run(start_server())