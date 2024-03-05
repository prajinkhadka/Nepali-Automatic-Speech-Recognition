# Nepali Automatic Speech Recognition

This project is focused on developing an automatic speech recognition (ASR) system for the Nepali language. The project uses deep learning techniques, particularly models based on the Wav2Vec 2.0 architecture, to transcribe Nepali speech audio into text. This README provides an overview of the project structure, setup instructions, and guidance on how to train the model and use the pre-trained model for inference.

## Project Structure

- `data_format_scrips/Openslr/`: Contains scripts for formatting and preprocessing the dataset obtained from OpenSLR for Nepali speech recognition.
- `all_files/`: Directory containing miscellaneous files related to the project.
- `external/`: Contains external resources or scripts used in the project.
- `src/`: The source code directory where the main codebase for the model and utility functions are located.
  - `configuration_wav2vec2.py`: Configuration class for the Wav2Vec 2.0 model.
  - `feature_extraction_wav2vec2.py`: Feature extraction utilities for processing audio files.
  - `modeling_flax_wav2vec2.py`: Wav2Vec 2.0 model implementation in Flax.
  - `modeling_wav2vec2.py`: Wav2Vec 2.0 model implementation in PyTorch.
  - `processing_wav2vec2.py`: Processing utilities for Wav2Vec 2.0 model input/output.
  - `tokenization_wav2vec2.py`: Tokenizer for Wav2Vec 2.0.
  - `trainer.py`: Training script for the Wav2Vec 2.0 model.
- `final_preprocessor/`: Contains preprocessing scripts and configuration files.
  - `preprocessor_config.json`: Preprocessor configuration.
  - `special_tokens_map.json`: Special tokens map.
  - `tokenizer_config.json`: Tokenizer configuration.
  - `vocab.json`: Vocabulary file.
- `Deployment/`: Files related to deploying the model using Docker.
  - `Dockerfile`: Dockerfile for building the container.
  - `docker-compose.yml`: Docker Compose file for easy deployment.
  - `gradio_app.py`: A Gradio app for easy demonstration of the model's capabilities.
- `CreatingTrainData_1.ipynb`: Jupyter notebook for the initial steps of training data creation.
- `CreatingTrainData_2-preprocessing.ipynb`: Jupyter notebook for preprocessing steps in training data creation.
- `.gitignore`: Gitignore file.
- `README.md`: This README file.

## Setup

### Prerequisites

- Docker and Docker Compose (for deployment)
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab (for running notebooks)

### Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd Nepali-Automatic-Speech-Recognition
