#include <Arduino.h>
// #include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "mel.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include "WiFi.h"
#include "config.h"
// #include "secrets.h"

// --- IMPORTANT: CHOOSE YOUR MODEL ---
// Comment out the one you are not using.
#include "model_quantized.h" // For the INT8 quantized model
// #include "model_float.h"   // For the float32 model

// Prototypes
void RunInference(float *feature_buffer);

// --- Model Configuration ---
const int kNumClasses = 2;
const char *kClassLabels[2] = {"fish", "talking"};
// --- TFLM Globals ---
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *model_input = nullptr;
TfLiteTensor *model_output = nullptr;

// The Tensor Arena is a buffer for model's input, output, and intermediate tensors.
// Its size is critical and depends on the model. You may need to tune this.
// For this model, 60KB should be safe for all sizes.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize] = {0};

void setup()
{
  Serial.begin(115200);
  while (!Serial)
    ; // Wait for Serial to be ready
  error_reporter->Report("Starting TFOT Firmware...");

  model = tflite::GetModel(g_model_quant_data);
  error_reporter->Report("Using INT8 Quantized Model.");

  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    error_reporter->Report("Model schema version mismatch!");
    return;
  }

  // Use AllOpsResolver to link all TFLM ops
  const tflite::MicroMutableOpResolver<10> resolver;
  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
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

  // begin setting up MEMS microphone using I2S

  // begin connecting to wireless network
  WiFi.begin(WIFI_NAME, WIFI_PASSWORD);

  // make callback to make sure to re-connect on disconnect
  WiFi.onEvent([](WiFiEvent_t event, WiFiEventInfo_t info)
               {
    if (event == WiFiEvent_t::ARDUINO_EVENT_WIFI_AP_STADISCONNECTED) {
      WiFi.begin(WIFI_NAME, WIFI_PASSWORD);
    } });
}

void loop()
{
  // put your main code here, to run repeatedly:

  // we will listen through the microphone and run inference on the audio data
  // once we have a keyword detected, we will record for 7s, or we find some way to stream audio to google gemini, then ask for the structured output
}

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