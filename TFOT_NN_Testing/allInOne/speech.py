#!/usr/bin/env python3
"""
Speech Activation Phrase Detection System
Generates TTS samples, background noise, trains a TFLite model for wake word detection
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import soundfile as sf
import random
import json
from typing import List, Tuple, Dict
import argparse
from pathlib import Path

# For TTS - Multiple engine support
import pyttsx3

try:
    import edge_tts
    import asyncio

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import gTTS

    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# VITS implementation using espnet2 or direct PyTorch
try:
    import torch
    import numpy as np
    from scipy.io.wavfile import write
    import requests
    import json

    VITS_AVAILABLE = True
except ImportError:
    VITS_AVAILABLE = False

# Alternative VITS using transformers
try:
    from transformers import VitsModel, VitsTokenizer

    VITS_TRANSFORMERS_AVAILABLE = True
except ImportError:
    VITS_TRANSFORMERS_AVAILABLE = False

print(
    f"Available TTS engines: Edge-TTS={EDGE_TTS_AVAILABLE}, gTTS={GTTS_AVAILABLE}, VITS={VITS_AVAILABLE}, VITS-Transformers={VITS_TRANSFORMERS_AVAILABLE}"
)


class TTSGenerator:
    """Text-to-Speech generator with multiple engine support including VITS"""

    def __init__(self, engine="auto", sample_rate=16000):
        self.sample_rate = sample_rate
        self.engine_type = engine

        # Auto-select best available engine
        if engine == "auto":
            if VITS_TRANSFORMERS_AVAILABLE:
                self.engine_type = "vits_transformers"
            elif VITS_AVAILABLE:
                self.engine_type = "vits"
            elif EDGE_TTS_AVAILABLE:
                self.engine_type = "edge_tts"
            elif GTTS_AVAILABLE:
                self.engine_type = "gtts"
            else:
                self.engine_type = "pyttsx3"

        print(f"Using TTS engine: {self.engine_type}")

        # Initialize selected engine
        self._init_engine()

    def _init_engine(self):
        """Initialize the selected TTS engine"""
        if self.engine_type == "vits_transformers":
            self._init_vits_transformers()
        elif self.engine_type == "vits":
            self._init_vits_direct()
        elif self.engine_type == "pyttsx3":
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)
            self.engine.setProperty("volume", 0.8)
            voices = self.engine.getProperty("voices")
            if voices and len(voices) > 1:
                self.engine.setProperty("voice", voices[min(1, len(voices) - 1)].id)

        # Edge-TTS voices (high quality, natural sounding)
        self.edge_voices = [
            "en-US-JennyNeural",  # Female, natural
            "en-US-GuyNeural",  # Male, natural
            "en-US-AriaNeural",  # Female, expressive
            "en-US-DavisNeural",  # Male, expressive
            "en-US-AmberNeural",  # Female, warm
            "en-US-AnaNeural",  # Female, cheerful
        ]

    def _init_vits_transformers(self):
        """Initialize VITS using Hugging Face transformers"""
        try:
            # Load pre-trained VITS model from Hugging Face
            model_name = "facebook/mms-tts-eng"  # Multilingual model with English
            self.vits_model = VitsModel.from_pretrained(model_name)
            self.vits_tokenizer = VitsTokenizer.from_pretrained(model_name)

            # Set to evaluation mode
            self.vits_model.eval()
            print("VITS (Transformers) model loaded successfully")

        except Exception as e:
            print(f"Failed to load VITS Transformers model: {e}")
            self.engine_type = "edge_tts" if EDGE_TTS_AVAILABLE else "pyttsx3"
            self._init_engine()

    def _init_vits_direct(self):
        """Initialize VITS using direct PyTorch implementation"""
        try:
            # You can load your own VITS model here or download pre-trained ones
            # This is a placeholder for direct VITS implementation
            self.vits_model = None
            self.vits_speakers = [
                "p225",
                "p226",
                "p227",
                "p228",
                "p229",
                "p230",
            ]  # VCTK speaker IDs
            print(
                "VITS (Direct) initialization - you may need to provide model checkpoints"
            )

        except Exception as e:
            print(f"Failed to initialize direct VITS: {e}")
            self.engine_type = "edge_tts" if EDGE_TTS_AVAILABLE else "pyttsx3"
            self._init_engine()

    def synthesize(self, text: str, output_path: str) -> str:
        """Synthesize text to speech and save to file"""
        try:
            if self.engine_type == "vits_transformers":
                return self._synthesize_vits_transformers(text, output_path)
            elif self.engine_type == "vits":
                return self._synthesize_vits_direct(text, output_path)
            elif self.engine_type == "edge_tts":
                return self._synthesize_edge_tts(text, output_path)
            elif self.engine_type == "gtts":
                return self._synthesize_gtts(text, output_path)
            else:
                return self._synthesize_pyttsx3(text, output_path)

        except Exception as e:
            print(f"TTS synthesis failed for '{text}' with {self.engine_type}: {e}")
            # Fallback to pyttsx3
            if self.engine_type != "pyttsx3":
                return self._synthesize_pyttsx3(text, output_path)
            return None

    def _synthesize_vits_transformers(
        self,
        text: str,
        output_path: str,
        noise_scale: float = 0.667,
        noise_scale_duration: float = 0.8,
    ) -> str:
        """Synthesize using VITS via Hugging Face transformers with variance control"""
        try:
            # Tokenize input text
            inputs = self.vits_tokenizer(text, return_tensors="pt")

            # Generate speech with adjustable noise parameters
            with torch.no_grad():
                # The forward pass of the VitsModel can accept these noise parameters
                outputs = self.vits_model(
                    **inputs,
                    noise_scale=noise_scale,
                    noise_scale_duration=noise_scale_duration,
                )
                waveform = outputs.waveform.squeeze().cpu().numpy()

            # Normalize audio
            waveform = waveform / np.max(np.abs(waveform))

            # Resample if necessary
            if hasattr(self.vits_model.config, "sampling_rate"):
                model_sr = self.vits_model.config.sampling_rate
            else:
                model_sr = 22050  # Default VITS sample rate

            if model_sr != self.sample_rate:
                waveform = librosa.resample(
                    waveform, orig_sr=model_sr, target_sr=self.sample_rate
                )

            # Save audio
            sf.write(output_path, waveform, self.sample_rate)
            return output_path

        except Exception as e:
            print(f"VITS Transformers synthesis failed: {e}")
            return self._synthesize_edge_tts(text, output_path)

    def _synthesize_vits_direct(self, text: str, output_path: str) -> str:
        """Synthesize using direct VITS implementation"""
        try:
            # This is where you'd implement direct VITS model inference
            # For now, we'll create a simple interface for loading custom VITS models

            if self.vits_model is None:
                # Try to load a VITS model from a common location or download one
                model_path = self._download_or_find_vits_model()
                if model_path:
                    self.vits_model = self._load_vits_model(model_path)

            if self.vits_model:
                # Perform VITS inference
                # This would be your custom VITS inference code
                speaker_id = (
                    random.choice(self.vits_speakers) if self.vits_speakers else None
                )
                waveform = self._vits_inference(text, speaker_id)

                # Save audio
                sf.write(output_path, waveform, self.sample_rate)
                return output_path
            else:
                print("No VITS model available, falling back to Edge-TTS")
                return self._synthesize_edge_tts(text, output_path)

        except Exception as e:
            print(f"Direct VITS synthesis failed: {e}")
            return self._synthesize_edge_tts(text, output_path)

    def _download_or_find_vits_model(self):
        """Download or find a VITS model checkpoint"""
        # Check for local VITS models
        possible_paths = [
            "./models/vits_model.pth",
            "./checkpoints/G_*.pth",
            "~/vits/checkpoints/G_*.pth",
        ]

        import glob

        for pattern in possible_paths:
            matches = glob.glob(os.path.expanduser(pattern))
            if matches:
                print(f"Found VITS model: {matches[0]}")
                return matches[0]

        # Try to download a pre-trained model
        try:
            model_url = (
                "https://github.com/jaywalnut310/vits/releases/download/models/G_0.pth"
            )
            model_path = "./vits_model.pth"

            if not os.path.exists(model_path):
                print("Downloading VITS model...")
                response = requests.get(model_url, stream=True)
                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print("VITS model downloaded successfully")
                    return model_path
        except Exception as e:
            print(f"Failed to download VITS model: {e}")

        return None

    def _load_vits_model(self, model_path):
        """Load VITS model from checkpoint"""
        try:
            # This would load your VITS model architecture and weights
            # You'd need to define your VITS model architecture here
            checkpoint = torch.load(model_path, map_location="cpu")
            print(f"Loaded VITS checkpoint from {model_path}")

            # Return the loaded model
            # This is a placeholder - you'd implement actual VITS model loading
            return checkpoint

        except Exception as e:
            print(f"Failed to load VITS model: {e}")
            return None

    def _vits_inference(self, text, speaker_id=None):
        """Perform VITS model inference"""
        # This is where you'd implement the actual VITS inference
        # For now, return a simple sine wave as placeholder
        duration = len(text) * 0.1  # Rough estimate
        samples = int(duration * self.sample_rate)

        # Generate a simple waveform (placeholder)
        t = np.linspace(0, duration, samples, False)
        waveform = 0.3 * np.sin(2 * np.pi * 220 * t)  # 220 Hz sine wave

        return waveform.astype(np.float32)

    def _synthesize_edge_tts(self, text: str, output_path: str) -> str:
        """Synthesize using Edge-TTS (Microsoft's high-quality TTS)"""
        import tempfile

        async def _generate():
            # Randomly select a voice for variety
            voice = random.choice(self.edge_voices)
            communicate = edge_tts.Communicate(text, voice)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            await communicate.save(tmp_path)
            return tmp_path

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tmp_path = loop.run_until_complete(_generate())
        loop.close()

        # Convert to target sample rate
        audio, sr = librosa.load(tmp_path, sr=self.sample_rate)
        sf.write(output_path, audio, self.sample_rate)

        # Clean up temp file
        os.remove(tmp_path)
        return output_path

    def _synthesize_gtts(self, text: str, output_path: str) -> str:
        """Synthesize using Google Text-to-Speech"""
        import tempfile

        # Create gTTS object with various accents for diversity
        accents = ["us", "co.uk", "ca", "com.au"]  # US, UK, Canada, Australia
        tld = random.choice(accents)

        tts = gTTS.gTTS(text=text, lang="en", tld=tld, slow=False)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        tts.save(tmp_path)

        # Convert MP3 to WAV at target sample rate
        audio, sr = librosa.load(tmp_path, sr=self.sample_rate)
        sf.write(output_path, audio, self.sample_rate)

        # Clean up
        os.remove(tmp_path)
        return output_path

    def _synthesize_pyttsx3(self, text: str, output_path: str) -> str:
        """Synthesize using pyttsx3 (local system TTS)"""
        temp_path = output_path.replace(".wav", "_temp.wav")

        self.engine.save_to_file(text, temp_path)
        self.engine.runAndWait()

        # Convert to desired sample rate using librosa
        if os.path.exists(temp_path):
            audio, sr = librosa.load(temp_path, sr=self.sample_rate)
            sf.write(output_path, audio, self.sample_rate)
            os.remove(temp_path)
            return output_path

        return None


class BackgroundNoiseGenerator:
    """Generate various types of background noise"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def generate_white_noise(self, duration: float, amplitude=0.1) -> np.ndarray:
        """Generate white noise"""
        samples = int(duration * self.sample_rate)
        return np.random.normal(0, amplitude, samples).astype(np.float32)

    def generate_pink_noise(self, duration: float, amplitude=0.1) -> np.ndarray:
        """Generate pink noise (1/f noise)"""
        samples = int(duration * self.sample_rate)
        white = np.random.normal(0, 1, samples)

        # Apply pink noise filter
        fft = np.fft.fft(white)
        freqs = np.fft.fftfreq(samples, 1 / self.sample_rate)
        freqs[0] = 1  # Avoid division by zero
        pink_filter = 1 / np.sqrt(np.abs(freqs))
        pink_fft = fft * pink_filter
        pink = np.real(np.fft.ifft(pink_fft))

        # Normalize
        pink = pink / np.max(np.abs(pink)) * amplitude
        return pink.astype(np.float32)

    def generate_brown_noise(self, duration: float, amplitude=0.1) -> np.ndarray:
        """Generate brown noise (Brownian noise)"""
        samples = int(duration * self.sample_rate)
        white = np.random.normal(0, 1, samples)
        brown = np.cumsum(white)
        brown = brown / np.max(np.abs(brown)) * amplitude
        return brown.astype(np.float32)

    def generate_room_tone(self, duration: float) -> np.ndarray:
        """Generate realistic room tone/ambient noise"""
        samples = int(duration * self.sample_rate)

        # Combine multiple frequencies for realistic room tone
        base_freq = np.random.uniform(50, 200)  # Base hum
        harmonics = [base_freq * i for i in range(1, 4)]

        room_tone = np.zeros(samples)
        for freq in harmonics:
            amplitude = 0.05 / (freq / base_freq)  # Decreasing amplitude for harmonics
            t = np.linspace(0, duration, samples, False)
            sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
            room_tone += sine_wave

        # Add some pink noise for realism
        room_tone += self.generate_pink_noise(duration, 0.02)

        return room_tone.astype(np.float32)

    def save_noise_samples(self, output_dir: str, duration=3.0, count=10):
        """Generate and save various noise samples"""
        os.makedirs(output_dir, exist_ok=True)

        noise_types = {
            "white": self.generate_white_noise,
            "pink": self.generate_pink_noise,
            "brown": self.generate_brown_noise,
            "room_tone": self.generate_room_tone,
        }

        for noise_type, generator in noise_types.items():
            for i in range(count):
                if noise_type == "room_tone":
                    noise = generator(duration)
                else:
                    amplitude = np.random.uniform(0.05, 0.2)
                    noise = generator(duration, amplitude)

                filename = f"{noise_type}_{i:03d}.wav"
                filepath = os.path.join(output_dir, filename)
                sf.write(filepath, noise, self.sample_rate)

        print(f"Generated {len(noise_types) * count} noise samples in {output_dir}")


class AudioPreprocessor:
    """Preprocess audio for model training"""

    def __init__(self, sample_rate=16000, window_size=1.0, hop_size=0.5):
        self.sample_rate = sample_rate
        self.window_samples = int(window_size * sample_rate)
        self.hop_samples = int(hop_size * sample_rate)

    def extract_mfcc_features(self, audio: np.ndarray, n_mfcc=13) -> np.ndarray:
        """Extract MFCC features from audio"""
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=n_mfcc, n_fft=512, hop_length=256
        )
        return mfcc.T  # Transpose to (time, features)

    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral features (mel spectrogram)"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=40, n_fft=512, hop_length=256
        )
        log_mel = librosa.power_to_db(mel_spec)
        return log_mel.T  # Transpose to (time, features)

    def create_windows(self, audio: np.ndarray) -> List[np.ndarray]:
        """Create overlapping windows from audio"""
        windows = []
        start = 0
        while start + self.window_samples <= len(audio):
            window = audio[start : start + self.window_samples]
            windows.append(window)
            start += self.hop_samples
        return windows


class ActivationPhraseModel:
    """TensorFlow Lite compatible model for activation phrase detection"""

    def __init__(
        self, input_shape=(63, 40), num_classes=2
    ):  # 63 time steps, 40 mel features
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        """Build TFLite compatible model with specified operators"""

        # Input layer
        inputs = keras.Input(shape=self.input_shape, name="audio_input")

        # Reshape for Conv2D (add channel dimension)
        x = keras.layers.Reshape((self.input_shape[0], self.input_shape[1], 1))(inputs)

        # Conv2D layer
        x = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=None,  # We'll add ReLU separately
        )(x)
        x = keras.layers.ReLU()(x)

        # DepthwiseConv2D layer
        x = keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3), strides=(1, 1), padding="same", activation=None
        )(x)
        x = keras.layers.ReLU()(x)

        # AveragePooling2D
        x = keras.layers.AveragePooling2D(
            pool_size=(2, 2), strides=(2, 2), padding="valid"
        )(x)

        # Another Conv2D layer
        x = keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=None,
        )(x)
        x = keras.layers.ReLU()(x)

        # Global Average Pooling to reduce dimensions
        x = keras.layers.GlobalAveragePooling2D()(x)

        # Reshape for FullyConnected
        x = keras.layers.Reshape((-1,))(x)

        # FullyConnected (Dense) layers
        x = keras.layers.Dense(64, activation=None)(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Dense(32, activation=None)(x)
        x = keras.layers.ReLU()(x)

        # Output layer with Softmax
        outputs = keras.layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = keras.Model(
            inputs=inputs, outputs=outputs, name="activation_phrase_detector"
        )

        # Compile model
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return self.model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model()

        # Add callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ]

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def convert_to_tflite(self, output_path: str, quantize=True):
        """Convert model to TensorFlow Lite format"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if quantize:
            # Enable quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # You can add representative dataset for better quantization

        # Set supported ops to ensure compatibility
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]

        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        print(f"TFLite model saved to {output_path}")

        # Verify the model
        interpreter = tf.lite.Interpreter(model_path=output_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")


class SpeechDetectionSystem:
    """Main system orchestrating all components"""

    def __init__(self, output_dir="speech_detection_data", sample_rate=16000):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate

        # Create directories
        self.audio_dir = self.output_dir / "audio"
        self.noise_dir = self.output_dir / "noise"
        self.models_dir = self.output_dir / "models"

        for dir_path in [self.audio_dir, self.noise_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.tts = TTSGenerator(engine="auto", sample_rate=sample_rate)
        self.noise_gen = BackgroundNoiseGenerator(sample_rate=sample_rate)
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.model = ActivationPhraseModel()

    def generate_activation_phrases(
        self, phrases: List[str], variations_per_phrase=1000
    ):
        """Generate TTS samples for activation phrases"""
        print("Generating activation phrase audio samples...")

        activation_dir = self.audio_dir / "activation"
        activation_dir.mkdir(exist_ok=True)

        phrase_files = []

        for phrase in phrases:
            for i in range(variations_per_phrase):
                filename = f"{phrase.replace(' ', '_').lower()}_{i:03d}.wav"
                output_path = activation_dir / filename

                result = self.tts.synthesize(phrase, str(output_path))
                if result:
                    phrase_files.append(str(output_path))

        print(f"Generated {len(phrase_files)} activation phrase samples")
        return phrase_files

    def generate_negative_samples(
        self, non_activation_words: List[str], count_per_word=3
    ):
        """Generate negative samples (non-activation phrases)"""
        print("Generating negative samples...")

        negative_dir = self.audio_dir / "negative"
        negative_dir.mkdir(exist_ok=True)

        negative_files = []

        for word in non_activation_words:
            for i in range(count_per_word):
                filename = f"{word.replace(' ', '_').lower()}_{i:03d}.wav"
                output_path = negative_dir / filename

                result = self.tts.synthesize(word, str(output_path))
                if result:
                    negative_files.append(str(output_path))

        print(f"Generated {len(negative_files)} negative samples")
        return negative_files

    def prepare_training_data(
        self, activation_files: List[str], negative_files: List[str]
    ):
        """Prepare training data from audio files"""
        print("Preparing training data...")

        X_data = []
        y_data = []

        # Process activation phrases (label = 1)
        for file_path in activation_files:
            try:
                audio, _ = librosa.load(file_path, sr=self.sample_rate)
                features = self.preprocessor.extract_spectral_features(audio)

                # Ensure consistent shape
                if features.shape[0] >= 63:
                    features = features[:63, :]  # Take first 63 time steps
                else:
                    # Pad if too short
                    pad_width = ((0, 63 - features.shape[0]), (0, 0))
                    features = np.pad(features, pad_width, mode="constant")

                X_data.append(features)
                y_data.append(1)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Process negative samples (label = 0)
        for file_path in negative_files:
            try:
                audio, _ = librosa.load(file_path, sr=self.sample_rate)
                features = self.preprocessor.extract_spectral_features(audio)

                # Ensure consistent shape
                if features.shape[0] >= 63:
                    features = features[:63, :]
                else:
                    pad_width = ((0, 63 - features.shape[0]), (0, 0))
                    features = np.pad(features, pad_width, mode="constant")

                X_data.append(features)
                y_data.append(0)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        X_data = np.array(X_data)
        y_data = np.array(y_data)

        print(f"Prepared {len(X_data)} training samples")
        print(f"Data shape: {X_data.shape}")
        print(
            f"Positive samples: {np.sum(y_data)}, Negative samples: {len(y_data) - np.sum(y_data)}"
        )

        return X_data, y_data

    def train_detection_model(self, X_data, y_data, test_split=0.2):
        """Train the activation phrase detection model"""
        print("Training activation phrase detection model...")

        # Split data
        split_idx = int(len(X_data) * (1 - test_split))
        indices = np.random.permutation(len(X_data))

        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_val = X_data[train_idx], X_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]

        # Train model
        history = self.model.train_model(X_train, y_train, X_val, y_val, epochs=30)

        # Save model
        model_path = self.models_dir / "activation_detector.h5"
        self.model.model.save(str(model_path))
        print(f"Model saved to {model_path}")

        # Convert to TFLite
        tflite_path = self.models_dir / "activation_detector.tflite"
        self.model.convert_to_tflite(str(tflite_path), quantize=True)

        return history

    def run_complete_pipeline(
        self, activation_phrases: List[str], negative_words: List[str]
    ):
        """Run the complete speech detection system pipeline"""
        print("Starting complete speech detection system pipeline...")

        # Step 1: Generate background noise
        print("\n=== Step 1: Generating background noise ===")
        self.noise_gen.save_noise_samples(str(self.noise_dir))

        # Step 2: Generate activation phrase samples
        print("\n=== Step 2: Generating activation phrase samples ===")
        activation_files = self.generate_activation_phrases(activation_phrases)

        # Step 3: Generate negative samples
        print("\n=== Step 3: Generating negative samples ===")
        negative_files = self.generate_negative_samples(negative_words)

        # Step 4: Prepare training data
        print("\n=== Step 4: Preparing training data ===")
        X_data, y_data = self.prepare_training_data(activation_files, negative_files)

        # Step 5: Train model
        print("\n=== Step 5: Training detection model ===")
        history = self.train_detection_model(X_data, y_data)

        print("\n=== Pipeline Complete! ===")
        print(f"All outputs saved to: {self.output_dir}")
        print(f"TFLite model: {self.models_dir}/activation_detector.tflite")

        return history


def main():
    parser = argparse.ArgumentParser(description="Speech Activation Detection System")
    parser.add_argument(
        "--output-dir",
        default="speech_detection_data",
        help="Output directory for all generated files",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Audio sample rate"
    )
    parser.add_argument(
        "--tts-engine",
        choices=["auto", "vits_transformers", "vits", "edge_tts", "gtts", "pyttsx3"],
        default="auto",
        help="TTS engine to use",
    )
    parser.add_argument(
        "--activation-phrases",
        nargs="+",
        default=["hey assistant", "wake up", "listen up", "start listening"],
        help="List of activation phrases to detect",
    )
    parser.add_argument(
        "--negative-words",
        nargs="+",
        default=[
            "hello",
            "goodbye",
            "thank you",
            "please",
            "sorry",
            "yes",
            "no",
            "maybe",
            "computer",
            "phone",
            "music",
            "stop",
        ],
        help="List of non-activation words for negative samples",
    )

    args = parser.parse_args()

    # Initialize system
    system = SpeechDetectionSystem(
        output_dir=args.output_dir, sample_rate=args.sample_rate
    )

    # Override TTS engine if specified
    if args.tts_engine != "auto":
        system.tts = TTSGenerator(engine=args.tts_engine, sample_rate=args.sample_rate)

    # Run complete pipeline
    history = system.run_complete_pipeline(
        activation_phrases=args.activation_phrases, negative_words=args.negative_words
    )

    # Save training history
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {
            key: [float(val) for val in values]
            for key, values in history.history.items()
        }
        json.dump(history_dict, f, indent=2)

    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
