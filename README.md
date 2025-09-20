Multimodal Sentiment Analysis with Cross-Modal Attention
This project implements a deep learning pipeline for multimodal sentiment analysis using the CMU-MOSI dataset. The model uses an attention-based fusion mechanism to integrate textual, visual, and acoustic features for sentiment prediction.

The entire pipeline is containerized using Docker for full reproducibility.

Project Structure
/configs: Contains YAML files for hyperparameters (config.yaml) and logging (logging_config.yaml).

/docker: Includes the Dockerfile and requirements.txt for building the environment.

/scripts: Contains the main Python scripts for running the pipeline steps.

/src: Holds the core source code, including the model definition and data utilities.

/data: (Created automatically) Stores raw and processed datasets.

/outputs: (Created automatically) Stores all experiment artifacts like models, logs, and metrics.
