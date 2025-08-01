#ifndef E08A62AE_8E46_40A7_9559_FB6E64C3CA84
#define E08A62AE_8E46_40A7_9559_FB6E64C3CA84
#include "esp_dsp.h"
#include "Arduino.h"

// --- Class Definition: MelSpectrogram ---
// A high-performance tool to create MEL spectrograms on the ESP32.

class MelSpectrogram
{
public:
  MelSpectrogram() : is_initialized(false),
                     fft_size(0),
                     hop_length(0),
                     num_mel_bins(0),
                     sample_rate(0),
                     num_frames(0),
                     frame(nullptr),
                     fft_input(nullptr),
                     window(nullptr),
                     mel_filterbank(nullptr),
                     mel_energies(nullptr) {}

  ~MelSpectrogram()
  {
    if (is_initialized)
    {
      // Free all allocated memory
      free(frame);
      free(fft_input);
      free(window);
      free(mel_filterbank);
      free(mel_energies);
      dsps_fft2r_deinit_fc32();
    }
  }

  /**
   * @brief Initializes the MelSpectrogram calculator.
   *        This must be called before compute().
   *
   * @param a_fft_size The size of the FFT window. Must be a power of 2.
   * @param a_hop_length The number of samples to hop between frames.
   * @param a_num_mel_bins The number of MEL bands to generate (height of spectrogram).
   * @param a_sample_rate The sample rate of the input audio.
   * @param a_num_frames The number of time frames to generate (width of spectrogram).
   * @return true on success, false on failure (e.g., memory allocation failed).
   */
  bool init(int a_fft_size, int a_hop_length, int a_num_mel_bins, int a_sample_rate, int a_num_frames)
  {
    if (is_initialized)
    {
      // Already initialized, maybe de-init first if you want to re-init
      return true;
    }

    fft_size = a_fft_size;
    hop_length = a_hop_length;
    num_mel_bins = a_num_mel_bins;
    sample_rate = a_sample_rate;
    num_frames = a_num_frames;

    // --- Initialize ESP-DSP FFT ---
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
    if (ret != ESP_OK)
    {
      Serial.println("dsps_fft2r_init_fc32 failed");
      return false;
    }

    // --- Allocate Memory Buffers ---
    // These are allocated once to avoid malloc/free in the compute loop
    frame = (float *)malloc(fft_size * sizeof(float));
    fft_input = (float *)malloc(fft_size * 2 * sizeof(float)); // For complex FFT input
    window = (float *)malloc(fft_size * sizeof(float));
    mel_filterbank = (float *)malloc(num_mel_bins * (fft_size / 2 + 1) * sizeof(float));
    mel_energies = (float *)malloc(num_mel_bins * sizeof(float));

    if (!frame || !fft_input || !window || !mel_filterbank || !mel_energies)
    {
      Serial.println("Failed to allocate memory");
      return false;
    }

    // --- Pre-calculate Hann Window ---
    dsps_wind_hann_f32(window, fft_size);

    // --- Pre-calculate MEL Filterbank ---
    create_mel_filterbank();

    is_initialized = true;
    return true;
  }

  /**
   * @brief Computes the MEL spectrogram from a uint8_t audio buffer.
   *
   * @param audio_buffer A pointer to the uint8_t audio buffer (assumed to be 8-bit offset-binary).
   * @param buffer_len The length of the audio buffer.
   * @param spectrogram_output A pointer to a pre-allocated float buffer to store the result.
   *                           Size must be num_mel_bins * num_frames.
   * @return true on success, false on failure.
   */
  bool compute(const uint8_t *audio_buffer, int buffer_len, float *spectrogram_output)
  {
    if (!is_initialized)
    {
      Serial.println("Not initialized!");
      return false;
    }

    for (int i = 0; i < num_frames; i++)
    {
      int start_index = i * hop_length;

      // 1. Framing and Windowing
      for (int j = 0; j < fft_size; j++)
      {
        int buffer_index = start_index + j;
        if (buffer_index < buffer_len)
        {
          // Convert uint8_t (0-255, center 128) to float (-1.0 to 1.0)
          frame[j] = ((float)audio_buffer[buffer_index] - 128.0f) / 128.0f;
        }
        else
        {
          frame[j] = 0.0f; // Zero padding
        }
        // Apply window function
        frame[j] *= window[j];
      }

      // 2. Perform FFT
      // Copy frame to complex FFT input buffer (real part, imaginary part is 0)
      for (int j = 0; j < fft_size; j++)
      {
        fft_input[j * 2] = frame[j];
        fft_input[j * 2 + 1] = 0.0f;
      }
      dsps_fft2r_fc32(fft_input, fft_size);
      dsps_bit_rev_fc32(fft_input, fft_size);

      // 3. Calculate Power Spectrum (magnitude squared)
      float power_spectrum[fft_size / 2 + 1];
      for (int j = 0; j <= fft_size / 2; j++)
      {
        float real = fft_input[j * 2];
        float imag = fft_input[j * 2 + 1];
        power_spectrum[j] = real * real + imag * imag;
      }

      // 4. Apply MEL Filterbank
      for (int bin = 0; bin < num_mel_bins; bin++)
      {
        float mel_energy = 0.0f;
        for (int spec_idx = 0; spec_idx <= fft_size / 2; spec_idx++)
        {
          mel_energy += power_spectrum[spec_idx] * mel_filterbank[bin * (fft_size / 2 + 1) + spec_idx];
        }
        mel_energies[bin] = mel_energy;
      }

      // 5. Take the logarithm of MEL energies
      for (int bin = 0; bin < num_mel_bins; bin++)
      {
        // Add a small epsilon to avoid log(0)
        spectrogram_output[i * num_mel_bins + bin] = log10f(mel_energies[bin] + 1e-6);
      }
    }
    return true;
  }

private:
  bool is_initialized;
  int fft_size;
  int hop_length;
  int num_mel_bins;
  int sample_rate;
  int num_frames;

  // Pre-allocated buffers
  float *frame;
  float *fft_input;
  float *window;
  float *mel_filterbank;
  float *mel_energies;

  // --- Helper functions for MEL scale conversion ---
  float freq_to_mel(float freq)
  {
    return 2595.0f * log10f(1.0f + freq / 700.0f);
  }

  float mel_to_freq(float mel)
  {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
  }

  void create_mel_filterbank()
  {
    float min_mel = freq_to_mel(0);
    float max_mel = freq_to_mel(sample_rate / 2.0f);

    float mel_points[num_mel_bins + 2];
    for (int i = 0; i < num_mel_bins + 2; i++)
    {
      mel_points[i] = min_mel + (float)i * (max_mel - min_mel) / (num_mel_bins + 1);
    }

    float freq_points[num_mel_bins + 2];
    int fft_bin_indices[num_mel_bins + 2];
    for (int i = 0; i < num_mel_bins + 2; i++)
    {
      freq_points[i] = mel_to_freq(mel_points[i]);
      fft_bin_indices[i] = floor((fft_size + 1) * freq_points[i] / sample_rate);
    }

    // Zero out the filterbank
    for (int i = 0; i < num_mel_bins * (fft_size / 2 + 1); i++)
    {
      mel_filterbank[i] = 0.0f;
    }

    for (int bin = 0; bin < num_mel_bins; bin++)
    {
      int start_idx = fft_bin_indices[bin];
      int mid_idx = fft_bin_indices[bin + 1];
      int end_idx = fft_bin_indices[bin + 2];

      for (int i = start_idx; i < mid_idx; i++)
      {
        mel_filterbank[bin * (fft_size / 2 + 1) + i] = (float)(i - start_idx) / (mid_idx - start_idx);
      }
      for (int i = mid_idx; i < end_idx; i++)
      {
        mel_filterbank[bin * (fft_size / 2 + 1) + i] = (float)(end_idx - i) / (end_idx - mid_idx);
      }
    }
  }
};
#endif /* E08A62AE_8E46_40A7_9559_FB6E64C3CA84 */
