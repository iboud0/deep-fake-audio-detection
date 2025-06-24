# Deepfake Audio Detection Project

## Overview
This project develops a deepfake audio detection system using the ASVspoof 2019 Logical Access dataset. It integrates acoustic and text features to classify audio as real or fake, achieving a peak F1 score of 0.9576 and an Equal Error Rate (EER) of 0.2444.

## Methodology Highlights
- Uses a subsampled ASVspoof 2019 LA dataset (2000 training, 500 development samples) with a 9:1 fake-to-real ratio.
- Extracts text features via Whisper (small) for transcription and DistilBERT for tokenization; extracts 35 acoustic features (MFCCs, spectral contrast, pitch, zero crossing rate) with normalization.
- Employs a custom DistilBERT model combining text (768D) and acoustic (32D) features, with most DistilBERT layers frozen to prevent overfitting.
- Addresses class imbalance using weighted cross-entropy loss and WeightedRandomSampler for balanced training.
- Trains for 10 epochs with AdamW (lr=5e-5) and a linear LR scheduler, selecting optimal thresholds via precision-recall curves.
- Evaluates with accuracy, F1, EER, ROC AUC, confusion matrix, and calibration plots.

## Critical Analysis
The multimodal approach effectively combines acoustic and text features, leveraging complementary signals to achieve a high F1 score (0.9576) despite a 9:1 class imbalance, with class weights and sampling ensuring robust performance. The use of frozen DistilBERT layers and the small Whisper model optimizes computation for resource-constrained environments, while comprehensive metrics (EER, ROC, calibration) provide a thorough performance assessment.

However, the model shows signs of overfitting, with a low training loss (0.0242) compared to a higher validation loss (0.7513) and a drop in accuracy (0.8580). Misclassification of real samples as fake indicates incomplete imbalance correction, and the wide threshold range (0.0032–0.0805) suggests instability for unseen data. The reliance on 35 acoustic features may miss subtle deepfake artifacts, and transcriptions struggle with noisy audio. The subsampled dataset limits generalization, contributing to a moderate EER (0.2444) compared to state-of-the-art systems (EER < 0.1).

## Future Work
- Incorporate advanced audio features (e.g., CQCC, Wav2Vec2) and explore end-to-end models like WavLM for improved detection.
- Expand the dataset with full ASVspoof splits or additional datasets (e.g., FakeAVCeleb) to enhance generalization.
- Experiment with focal loss or generative methods (e.g., GANs) to better address class imbalance.
- Implement cross-validation for stable threshold selection and evaluate on a separate test set.

## Contact and Further Resources
For questions, detailed reports, or inquiries about similar work on image and video deepfake detection, please contact me at [ilyas.boudhaine@um6p.ma](mailto:ilyas.boudhaine@um6p.ma).

## Contributions
Contributions are welcome! Please open an issue to discuss proposed changes or enhancements, and we can collaborate to improve the project.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.