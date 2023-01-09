__Course: Introduction to Deep Learning__
# __Day 1: Introduction to Deep Learning__
_Tuesday, January 3rd, 2023_

- Distinguish the differences between traditional ML and Deep Learning (capabilities, objective, problem, why, models, ...)

| ML     | DL |
| :---      | :---    |
| need human interaction with data to ensure model fits well, the feature extraction      | need less human involment or no need at all
| can only return acceptable results because it works with human-engineering features| can extract complex and rich feature pattern, hence working better ML models |
| not too good for unstructured data | can perform well also on unstructured data (texts, voices, ...)  |
| only works on moderate size dataset, performance decrease when dataset size increase       | work better in massive dataset, but cannot adapt small scale data 
| required CPU only | required high-computing resources (GPU, TPU)|
| Algorithms that learn from structured data to predict outputs and discover patterns in that data. | Algorithms based on highly complex neural networks that mimic the way a human brain works to detect patterns in large unstructured data sets|

### __what is deep learning?__
> Deep Learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks.

## __What should be learnt in Machine Learning?__
- Understanding ML models
    - Architecture, Structure, or form of the models
    - Computation, inference/reasoning
- Methods/Algorithms for learning models’ parameters
- Model evaluation
- Data Processing: data collection; feature selection; dimensional reduction, noisy filtering,…
- Other issues: overfitting, 

## __How to develope a DL system__
- Understanding DL Models
- Implementation
- Tuning pre-trained models
- Runing Environment
- Data Collection, Data Preprocessing, Data Labeling

### __Implementation__ 
> MNIST dataset classification problem
- a 6-steps piece of python code using `keras` and `tensorflow` API to train a model to classify handwritten number pictures
``` python
from tensorflow import keras
import tensorflow

# data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# modeling
model = Sequential([
    keras.layer.Flatten(input_shape= (28,28)),
    keras.layer.Dense(128,"relu"),
    keras.layer.Dense(10,"relu")
])

# compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
    metrics=["acc"]
)

# train 
model.fit(train_images, train_labels, epochs = 5)

# evaluate
test_loss, test_acc= model.evaluate(test_images, test_labels)
print(f"test acc {test_acc}")

# prediction
predictions = model.predict(test_images)
```

### __Tuning a pretrained model__ 

A pre-trained model is a saved network that was previously trained on a large dataset, typically on a large-scale image-classification task. You either use the pretrained model as is or use transfer learning to customize this model to a given task.

The intuition behind transfer learning for image classification is that if a model is trained on a large and general enough dataset, this model will effectively serve as a generic model of the visual world. You can then take advantage of these learned feature maps without having to start from scratch by training a large model on a large dataset.

_**Two ways to customize a pretrained model:**_

1. _Feature Extraction_: Use the representations learned by a previous network to extract meaningful features from new samples. You simply add a new classifier, which will be trained from scratch, on top of the pretrained model so that you can repurpose the feature maps learned previously for the dataset.

    You do not need to (re)train the entire model. The base convolutional network already contains features that are generically useful for classifying pictures. However, the final, classification part of the pretrained model is specific to the original classification task, and subsequently specific to the set of classes on which the model was trained.

2. _Fine-Tuning_: Unfreeze a few of the top layers of a frozen model base and jointly train both the newly-added classifier layers and the last layers of the base model. This allows us to "fine-tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task.

### __What do you need for setting up a deep learning studying evironment?__
- Annaconda
- Jupyter Notebook 
- Tensorflow
- Keras
- or Nvidia GeForce Experience if you have a green team GPU

## __Topics of this course__
- Basic Neural Networks
- Deep Belief Network
- Convolutional Neural Networks
- Recurrent Neural Networks and Long Short Term Memory
- Transformer Models
- Generative Adversarial Networks
- Reinforcement Learning and Deep RL

## __Course assessment__
- Middle exam (20%)
    - Project and presentation
- Final exam (50%)
    - Project and presentation
- Progress exercises (30%)

## __Summary__
- What is machine learning?
    - objective and characteristics
    - types of machine learning
- Distinguish between conventional machine learning and deep learning?
- What is the difference between developing a traditional ML system
and a deep learning system?


# Lecture 2: Basic of Neural Networks
## __Review__
- MLP 
- Logistic Regression
- Gradient Descent Algorithm
- Mean Squared Error
- How to define a loss function 

### __Logistic Regression__ 
- __Loss function__:
The loss function for linear regression is squared loss. The loss function for logistic regression is Log Loss, which is defined as follows:
$$ \text{Log Loss} = \sum_{(x,y)\in D} -y\log(y') - (1 - y)\log(1 - y') $$ 

where:

 - $(x, y) \in D $ is the data set containing many labeled examples, which are $(x, y)$ pairs.
 - $y$ is the label in a labeled example. Since this is logistic regression, every value of $y$ must either be 0 or 1.
 - $y'$ is the predicted value (somewhere between 0 and 1), given the set of features in $x$.

> __cross-entropy loss function__

* __*Kullback–Leibler divergence*__

    In mathematical statistics, the Kullback–Leibler divergence (also called relative entropy and I-divergence), $ {\displaystyle D_{\text{KL}}(P\parallel Q)} $, is a type of statistical distance: a measure of how one probability distribution P is different from a second, reference probability distribution Q. A simple interpretation of the KL divergence of P from Q is the expected excess surprise from using Q as a model when the actual distribution is P
    
## __Homework__
1. GD for Linear Regression
2. GD for clsfication w/ Logistic Reg

Present about Learning Stage for Multilayer Perceptron for Regression and classification Problems:
- Backpropagation algorithms with Mean Square Error (MSE) Cost function (for Regression) and Cross Entropy Cost Function (for classification).
- Appropriate Codes
- Applying for house pricing and adult datasets 

Put your answer on Github and submit the link
 
> + MLP regression - MSE 
> + MLP classification - cross entropy 
> + dataset - adult + housing
> => 4 models