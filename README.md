# Forest Fire Detection Project

Welcome to our project! This repository contains various scripts and data related to the detection of forest fires using audio analysis and machine learning techniques. Below is an overview of the project's structure and content. If you have any questions, please feel free to contact me.

## Project Structure

### Folder: `IA`

This folder contains the Python scripts for AI training, testing, and database creation.

#### AI Training Scripts

- **AI_for_a_large_data.py**: Trains the AI using log mel spectrogram and advanced audio characteristics.
- **AI_for_changed_data.py**: Trains the AI using altered data (sound of bad quality). This approach has been found to be less effective, but you are welcome to improve it.
- **AI_old_version.py**: Trains the AI using the normal spectrogram, frequency, and time.

#### AI Testing Scripts

- **AI_test_old_version.py**: Tests the AI trained using the `AI_old_version.py` script.
- **AI_test.py**: Tests the AI models trained using `AI_for_a_large_data.py` and `AI_for_changed_data.py`.

> **Important Note**: Ensure you are in the correct directory (`IA/spectr_data`) when running these scripts.

#### Database Creation Scripts

- **create_database_old_version.py**: Generates CSV files for the `AI_old_version`.
- **create_database.py**: Generates CSV files for the `AI_for_a_large_data`.

#### Other Scripts

- **log_mel_spectro.py**: Calculates the log mel spectrogram.
- **other_features.py**: Calculates advanced features such as MFCCs, spectral contrast, etc.

### Folder: `recording`

This folder contains scripts related to audio data collection and enhancement.

- **config.py**: Receives audio files from the Raspberry Pi.
- **enhance_audio_quality.py**: Enhances the quality of the audio using various methods, primarily equalization.
- **fixing_audio_all_data_base.py**: Enhances the quality of the entire database using one method from the `enhance_audio_quality.py` script with specific values (lowbin, highbin, and the multiplication factor).
- **sound_deter.py**: Alters the sound quality.

## Data Folders

- **spectr_data**: Contains spectrogram data used by the AI models.
- **sound_data**: Contains raw audio data used for training and testing.

## Getting Started

1. **Setting Up**: Ensure you have the necessary dependencies installed. Check the libraries used in the project.

2. **Training the AI**: Navigate to the IA/spectr_data directory and run the desired training script. For example:
```bash
   cd IA/spectr_data
    python AI_for_a_large_data.py
    ```

3. **Testing the AI: ** Similarly, navigate to the IA/spectr_data directory and run the testing script:
```bash
   cd IA/spectr_data
   python AI_test.py
    ```
4. **Creating the Database:** To generate the database, run the appropriate script from the IA directory:
```bash
   cd IA
    python create_database.py
    ```
then say yes to the instrcution