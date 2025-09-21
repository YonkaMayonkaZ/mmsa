# Multimodal Sentiment Analysis with Attention Fusion

This project implements a deep learning model for multimodal sentiment analysis on the CMU-MOSI dataset. It fuses features from text, video, and audio modalities to predict sentiment.

The model achieves **59.42% accuracy** on the test set.

### Key Features
* **Word-Level Alignment**: Pre-computes BERT embeddings for each word, ensuring perfect temporal alignment with audio and visual features.
* **Cross-Modal Attention**: Employs bidirectional attention to learn the complex interactions between language, visuals, and acoustics.
* **Tensor Fusion**: Uses a powerful fusion mechanism with `Bilinear` layers to create a rich, unified representation of the three modalities.


