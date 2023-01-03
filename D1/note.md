__Course: Introduction to Deep Learning__
# __Day 1: Introduction to Deep Learning__

- Distinguish the differences between traditional ML and Deep Learning (capabilities, objective, problem, why, models, ...)

| ML     | DL |
| :---      | :---    |
| need human interaction with data to ensure model fits well        | need less human involment or no need at all
| only works on moderate size dataset, performance decrease when dataset size increase       | work better in massive dataset, but cannot adapt small scale data
| Algorithms that learn from structured data to predict outputs and discover patterns in that data. | Algorithms based on highly complex neural networks that mimic the way a human brain works to detect patterns in large unstructured data sets|

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

### __Fine-tuning model__ 

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



## Course assessment
- Middle exam (20%)
    - Project and presentation
- Final exam (50%)
    - Project and presentation
- Progress exercises (30%)

## Summary
- What is machine learning?
    - objective and characteristics
    - types of machine learning
- Distinguish between conventional machine learning and deep learning?
- What is the difference between developing a traditional ML system
and a deep learning system?