# healthy-beans

## Description:
This project explores deep learning techniques for binary image classification on a low-resource dataset. Specifically, we utilized the Beans dataset, consisting of images of healthy and unhealthy bean plant leaves, as a proxy for screening rare genetic disorders. The insights gained from this task can inform the development of more accessible and cost-effective diagnostic tools.

## Components:

### Convolutional Neural Network (CNN):

Model Architecture: Implemented a custom version of AlexNet due to its simplicity and proven effectiveness.
Hyperparameters:
Learning Rate: 0.01
Batch Size: 16
Epochs: 50
Optimizer: Stochastic Gradient Descent
Loss: Cross Entropy Loss
Performance: Trained on the Beans dataset, achieving notable results on the validation set. Detailed performance is presented in the confusion matrix.
### Autoencoder:
Model Architecture: A CNN-based autoencoder with distinct encoder and decoder components.
Encoder: 4 convolutional layers with ReLU activations and batch normalization, followed by 2 linear layers.
Decoder: 4 convolutional layers with ReLU activations and batch normalization, followed by 2 linear layers.
Classifier: Uses the encoder layer, followed by 3 linear layers for classification.
Hyperparameters:
Encoder/Decoder Learning Rate: 0.01
Classifier Learning Rate: 0.001
Batch Size: 4 (Autoencoder), 16 (Classifier)
Epochs: 7 (Autoencoder), 30 (Classifier)
Optimizer: Adam (Autoencoder), SGD (Classifier)
Loss: Mean Squared Error (Autoencoder), Cross Entropy Loss (Classifier)
Performance: Evaluated on the Beans dataset with results detailed in a confusion matrix.
### Other Methods (Discussion):
Discussed various approaches for effective classification with limited labeled samples, including data augmentation, transfer learning, shallow networks, active learning, and ensemble methods.

## Results:

- CNN Accuracy: Achieved promising results, with learning curves and a confusion matrix for the validation set.
- Autoencoder Performance: Demonstrated effective image reconstruction and classification capabilities.

## Files Included:

- Code: Python scripts for CNN and Autoencoder implementations.
- Models: Saved model parameters and checkpoints.
- Results: Confusion matrices, learning curves, and performance metrics.
- Blind Test Predictions: Results for the blind test dataset in CSV format.
This project showcases the application of deep learning models to low-resource datasets, highlighting the potential for developing scalable and cost-effective diagnostic tools in various fields.
