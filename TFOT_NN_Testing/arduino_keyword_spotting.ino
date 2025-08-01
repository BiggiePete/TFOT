/*
  Keyword Spotting with TensorFlow Lite for Microcontrollers
  Generated from Google Colab training notebook.

  This example shows how to set up the TFLite Micro interpreter.
  You need to provide the audio capture and feature extraction pipeline.

  - Target: ESP32 or similar 32-bit microcontroller
  - Dependencies: TensorFlowLite_ESP32 library or the official TFLM library
  - Classes: fish, talking
*/

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --- IMPORTANT: CHOOSE YOUR MODEL ---
// Comment out the one you are not using.
#include "model_quantized.h" // For the INT8 quantized model
// #include "model_float.h"   // For the float32 model

// --- Model Configuration ---
const int kNumClasses = 2;
const char *kClassLabels[2] = {"fish", "talking"};

// --- TFLM Globals ---
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = µ_error_reporter;

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *model_input = nullptr;
TfLiteTensor *model_output = nullptr;

// The Tensor Arena is a buffer for model's input, output, and intermediate tensors.
// Its size is critical and depends on the model. You may need to tune this.
// For this model, 60KB should be safe for all sizes.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// --- Setup ---
void setup()
{
  Serial.begin(115200);
  while (!Serial)
    ; // Wait for serial connection

  error_reporter->Report("Starting Keyword Spotting setup.");

  // Choose the model data to load
#ifdef g_model_quant_data
  model = tflite::GetModel(g_model_quant_data);
  error_reporter->Report("Using INT8 Quantized Model.");
#else
  model = tflite::GetModel(g_model_float_data);
  error_reporter->Report("Using Float32 Model.");
#endif

  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    error_reporter->Report("Model schema version mismatch!");
    return;
  }

  // Use AllOpsResolver to link all TFLM ops
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's tensors
  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    error_reporter->Report("AllocateTensors() failed.");
    return;
  }

  // Get pointers to the input and output tensors
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  error_reporter->Report("Setup complete. Model is ready for inference.");
}

// --- Loop ---
void loop()
{
  // 1. CAPTURE AUDIO: Use a microphone (e.g., I2S) to get a 1-second audio clip.
  //
  // 2. EXTRACT FEATURES: Process the audio into a mel spectrogram. This is the most
  //    challenging part on an MCU. You'll need a DSP library (e.g., KissFFT, CMSIS-DSP).
  //    The spectrogram must have the shape: 40x63
  //
  // 3. RUN INFERENCE: Call the RunInference() function with your features.
  //
  // 4. PROCESS RESULTS: Check the output tensor for the predicted keyword.

  // create mel spectrogram features
  float feature_buffer[40 * 63]; // Example buffer, replace with actual feature extraction

  error_reporter->Report("Loop: Waiting for audio processing implementation...");
  delay(5000);
}

// --- Run Inference ---
// This function takes a pointer to the feature buffer and runs the model.
void RunInference(float *feature_buffer)
{
  // Copy features into the input tensor
  // Handle quantization if using the INT8 model
  if (model_input->type == kTfLiteInt8)
  {
    float input_scale = model_input->params.scale;
    int8_t input_zero_point = model_input->params.zero_point;
    for (int i = 0; i < model_input->bytes; i++)
    {
      model_input->data.int8[i] = (int8_t)(feature_buffer[i] / input_scale + input_zero_point);
    }
  }
  else
  {
    for (int i = 0; i < model_input->bytes / sizeof(float); i++)
    {
      model_input->data.f[i] = feature_buffer[i];
    }
  }

  // Run the model
  if (interpreter->Invoke() != kTfLiteOk)
  {
    error_reporter->Report("Invoke failed.");
    return;
  }

  // Find the class with the highest score
  int8_t max_score = -128;
  int predicted_class = -1;
  for (int i = 0; i < kNumClasses; i++)
  {
    int8_t current_score = model_output->data.int8[i];
    if (current_score > max_score)
    {
      max_score = current_score;
      predicted_class = i;
    }
  }

  // De-quantize the score for display
  float confidence = (max_score - model_output->params.zero_point) * model_output->params.scale;

  error_reporter->Report("Prediction: %s (Confidence: %.2f)", kClassLabels[predicted_class], confidence);
}
