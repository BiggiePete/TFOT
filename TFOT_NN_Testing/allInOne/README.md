I'll help you build a complete speech activation phrase detection system with TTS synthesis, background noise generation, and a TensorFlow Lite model. This is quite comprehensive, so I'll create a script that handles all components.I've created a comprehensive speech activation detection system that addresses all your requirements. Here's what the script provides:

## Key Features

### 1. **TTS Engine with Human-like Speech**

- **Primary**: VITS integration for natural, human-like speech synthesis
- **Fallback**: pyttsx3 for basic TTS if VITS isn't available
- Configurable voice parameters and sample rates

### 2. **Background Noise Generation**

- **White noise**: Pure random noise
- **Pink noise**: 1/f noise (more natural)
- **Brown noise**: Brownian motion noise
- **Room tone**: Realistic ambient room noise with harmonics

### 3. **TensorFlow Lite Compatible Model**

The model uses exactly the operators you specified:

- **Conv2D**: 2D convolution with quantized tensors
- **DepthwiseConv2D**: Efficient depthwise separable convolution
- **AveragePool2D**: Average pooling for dimension reduction
- **FullyConnected (Dense)**: Fully connected layers
- **Reshape**: Tensor reshaping operations
- **Activation Functions**: ReLU, ReLU6, and Softmax

### 4. **Complete Pipeline**

- Generates TTS samples for activation phrases
- Creates negative samples from regular words
- Extracts mel-spectrogram features (40 mel bands, 63 time steps)
- Trains the model with early stopping and learning rate scheduling
- Converts to quantized TFLite format

## Usage

### Basic Usage

```bash
python speech_detection.py
```

### Custom Configuration

```bash
python speech_detection.py \
    --activation-phrases "hey jarvis" "computer wake up" "start listening" \
    --negative-words "hello" "music" "weather" "time" \
    --output-dir my_speech_data \
    --sample-rate 16000
```

## Installation Requirements

```bash
pip install tensorflow librosa soundfile pyttsx3 numpy scipy TTS torch
```

## Output Structure

```
speech_detection_data/
├── audio/
│   ├── activation/     # TTS samples of activation phrases
│   └── negative/       # TTS samples of non-activation words
├── noise/             # Background noise samples
└── models/
    ├── activation_detector.h5      # Full Keras model
    ├── activation_detector.tflite  # Quantized TFLite model
    └── training_history.json       # Training metrics
```

## Model Architecture

The model is specifically designed to be TFLite compatible with quantization support:

1. **Input**: (63, 40) mel-spectrogram features
2. **Conv2D**: 32 filters, 3x3 kernel + ReLU
3. **DepthwiseConv2D**: 3x3 kernel + ReLU6
4. **AveragePool2D**: 2x2 pooling
5. **Conv2D**: 64 filters, 3x3 kernel + ReLU
6. **GlobalAveragePooling**: Dimension reduction
7. **Dense**: 64 units + ReLU
8. **Dense**: 32 units + ReLU
9. **Output**: 2 classes (activation/non-activation) + Softmax

The script handles the complete workflow from audio synthesis to a deployable TFLite model. The model is small, efficient, and ready for edge deployment with quantization support.
