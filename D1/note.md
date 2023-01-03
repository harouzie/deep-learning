__Course: Introduction to Deep Learning__
# __Day 1: Introduction to Deep Learning__

- Distinguish the differences between traditional ML and Deep Learning (capabilities, objective, problem, why, models, ...)

| ML     | DL |
| :---      | :---    |
| need human interaction with data to ensure model fits well        | need less human involment or no need at all
| only works on moderate size dataset, performance decrease when dataset size increase       | work better in massive dataset, but cannot adapt small scale data
| Algorithms that learn from structured data to predict outputs and discover patterns in that data. | Algorithms based on highly complex neural networks that mimic the way a human brain works to detect patterns in large unstructured data sets|

## What should be learnt in Machine Learning?
- Understanding ML models
    - Architecture, Structure, or form of the models
    - Computation, inference/reasoning
- Methods/Algorithms for learning models’ parameters
- Model evaluation
- Data Processing: data collection; feature selection; dimensional reduction, noisy filtering,…
- Other issues: overfitting, 

## How to develope a DL system
- Understanding DL Models
- Implementation
- Tuning pre-trained models
- Runing Environment
- Data Collection, Data Preprocessing, Data Labeling

### Implementation 
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



