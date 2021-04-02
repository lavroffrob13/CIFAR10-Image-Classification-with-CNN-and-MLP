# CIFAR10-Image-Classification-with-CNN-and-MLP

To run the convolutional neural network, use "python cnn.py" and to run it without ReLu activation, use "python cnn_noAF.py"

To run the fully connected neural network, use "python fc.py" and to run it without ReLu activation, use "python fc_noAF.py"

For all of these, you can adjust the number of epochs using the "epochs" hyperparameter, the learning rate using the "eta" hyperparameter, and the momentum using the "momentum" hyperparameter. The code tries to run on GPU if possible, so if your computer's driver isn't recent enough to use CUDA, there may be a benign error shown.
