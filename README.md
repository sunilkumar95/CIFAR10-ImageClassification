# Image Classification with CIFAR10 with ~80% accuracy

Requirements:
* Python 2.7.1
* TensorFlow 1.4.1
* CIFAR-10 dataset downloaded from https://www.cs.toronto.edu/~kriz/cifar.html

In order to train the model, navigate to the root directory and run the desired .sh file located in the scripts folder (i.e scripts/conv_net.sh). Both bash files contain the following customizable options:

* data_path (String): The path to the folder containing the CIFAR-10 data from the root directory
* output_dir (String): The path to wherever you want the outputs stored
* log_every (int): The number of episodes between calculating validation score
* num_epochs (int): The number of epochs to train
* train_batch_size (int): The size of the training batch at each step
* eval_batch_size (int): The size of the evaluation batch when calculating validation accuracy
* l2_reg (float): The rate of L2 regularization
* learning_rate (float): The learning rate

Allowing the convolutional model to fully train without changing the parameters should result in a final accuracy of aroun 80% on the CIFAR-10 dataset.
