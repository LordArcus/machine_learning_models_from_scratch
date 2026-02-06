# Machine Learning Models from Scratch

Simple implementations of machine learning algorithms using NumPy and Python.

## 🎯 Models Implemented

### **Regression Models**

#### SimpleLinearRegression_OLS.py
- **Algorithm**: Ordinary Least Squares (OLS)
- **Use Case**: Single feature regression
- **Key Methods**:
  - `fit(X_train, y_train)`: Fits the model to training data
  - `predict(X_test)`: Makes predictions on test data
  - `coefficients()`: Returns slope and intercept

#### MultipleLinearRegression_0LS.py
- **Algorithm**: Ordinary Least Squares (OLS)
- **Use Case**: Multiple feature regression
- **Key Methods**:
  - `fit(X_train, y_train)`: Fits the model with multiple features
  - `predict(X_test)`: Makes predictions

#### MultipleLinearRegression_GD.py
- **Algorithm**: Gradient Descent Optimization
- **Use Case**: Multiple feature regression with iterative learning
- **Parameters**:
  - `learning_rate`: Step size for weight updates (default: 0.01)
  - `epochs`: Number of training iterations (default: 100)
- **Key Methods**:
  - `fit(X_train, y_train)`: Trains using gradient descent
  - `predict(X_test)`: Makes predictions

#### MultipleLinearRegression_SGD.py
- **Algorithm**: Stochastic Gradient Descent (SGD)
- **Use Case**: Large-scale regression with sample-wise updates
- **Parameters**:
  - `learning_rate`: Step size for weight updates (default: 0.01)
  - `epochs`: Number of training iterations (default: 100)
- **Key Methods**:
  - `fit(X_train, y_train)`: Trains using SGD
  - `predict(X_test)`: Makes predictions

---

### **Classification Models**

#### LogisticRegression.py
- **Algorithm**: Logistic Regression with Gradient Descent
- **Use Case**: Binary classification problems
- **Loss Function**: Binary Cross-Entropy
- **Activation Function**: Sigmoid
- **Key Methods**:
  - `train(alpha, epochs)`: Trains the model with specified learning rate and epochs
  - `predict(X)`: Makes binary predictions
  - `loss()`: Computes binary cross-entropy loss

#### SupportVectorMachine_Linear.py
- **Algorithm**: Linear SVM with Hinge Loss
- **Use Case**: Binary classification with maximum margin separation
- **Loss Function**: Hinge Loss + L2 Regularization
- **Optimization**: Stochastic Gradient Descent
- **Parameters**:
  - `C`: Regularization parameter (default: 1.0)
  - `learning_rate`: Learning rate (default: 0.01)
  - `epochs`: Number of training iterations (default: 1000)
- **Key Methods**:
  - `fit(X, y)`: Trains the SVM model
  - `predict(X)`: Makes predictions

---

### **Neural Networks**

#### MultiNeurons.py
- **Architecture**: Multi-layer Feedforward Neural Network
- **Features**:
  - Support for multiple hidden layers
  - Configurable activation functions (sigmoid, ReLU, softmax)
  - Support for different loss functions (MSE, cross-entropy)
- **Parameters**:
  - `layer_sizes`: List specifying number of neurons per layer
  - `activations`: Activation functions for each layer
  - `error`: Loss function type
- **Key Methods**:
  - `forward_pass(X)`: Computes forward pass
  - `backward_pass(X, y)`: Computes backpropagation
  - `fit(X_train, y_train, epochs)`: Trains the network
  - `predict(X_test)`: Makes predictions

#### RNN.py
- **Architecture**: Recurrent Neural Network with single hidden layer
- **Use Case**: Sequential data and time-series prediction
- **Features**:
  - Backpropagation Through Time (BPTT)
  - Gradient clipping to prevent exploding gradients
  - Configurable hidden layer size
- **Parameters**:
  - `input_size`: Size of input features
  - `hidden_size`: Size of hidden state
  - `output_size`: Size of output layer
  - `learning_rate`: Learning rate (default: 0.001)
  - `epochs`: Number of training epochs (default: 3)
  - `batch_size`: Batch size for training (default: 32)
  - `clip`: Gradient clipping threshold (default: 5.0)
- **Key Methods**:
  - `forward(X)`: Forward pass through RNN
  - `backward(X, y_true, y_pred, hs)`: Backpropagation through time
  - `train(X_train, y_train)`: Trains the model
  - `predict(X_test)`: Makes predictions
  - `evaluate(X_test, y_test)`: Evaluates model performance

#### CNN.py
- **Architecture**: Convolutional Neural Network for 2D inputs
- **Use Case**: Image processing and pattern recognition
- **Features**:
  - 2D Convolution with configurable stride and padding
  - Max Pooling operation
  - Flatten utility for vectorization
- **Key Methods**:
  - `convolution2d(input_matrix, kernel, stride, padding, bias)`: Performs 2D convolution
  - `max_pooling2d(input_matrix, pool_size, stride)`: Performs max pooling
  - `flatten(input_matrix)`: Flattens multi-dimensional input

---

### **Association Rules Mining**

#### association_mining.py
- **Algorithm**: Apriori Algorithm
- **Use Case**: Market basket analysis and frequent itemset mining
- **Metrics Calculated**:
  - **Support**: P(X and Y) / Total Transactions
  - **Confidence**: P(Y|X) = Support(X and Y) / Support(X)
  - **Lift**: Confidence(X → Y) / Support(Y)
- **Key Functions**:
  - `get_support(item, transactions)`: Calculates support for an itemset
  - `get_confidence(X, Y, transactions)`: Calculates confidence for rule X → Y
  - `get_lift(X, Y, transactions)`: Calculates lift for rule X → Y
  - `generate_candidate_itemsets(prev_itemsets)`: Generates candidate k-itemsets
  - `apriori(transactions, min_support, min_confidence, min_lift)`: Main Apriori algorithm

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- NumPy
- Pandas (for data handling)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/machine_learning_models_from_scratch.git
cd machine_learning_models_from_scratch
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install numpy pandas
```



## 🔑 Key Features

✅ **From Scratch Implementation**: All models built using only NumPy and Python  
✅ **Educational**: Clear, well-commented code for learning  
✅ **Comprehensive**: Covers multiple ML domains and algorithms  
✅ **Flexible**: Easy to modify and extend  

---

## 💡 Learning Insights

This repository helps you understand:

1. **How algorithms work internally** - No black-box abstractions
2. **Mathematical foundations** - See the math translated to code
3. **Optimization techniques** - Gradient descent, SGD, etc.
4. **Neural network mechanics** - Backpropagation, forward/backward passes
5. **Data mining** - Association rules and pattern discovery

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Fix bugs
- Improve documentation
- Add new algorithms
- Optimize existing implementations

---

## 📝 License

This project is open source and available under the MIT License.

---

## 📮 Contact & Support

For questions or suggestions, please open an issue or contact me.

---

## 🎓 Educational Resources

This repository is designed for:
- Students learning machine learning fundamentals
- Data scientists wanting to understand algorithm internals
- Anyone interested in implementing ML from scratch
- Interview preparation for ML/DS roles

---

## ⚡ Performance Notes

- These implementations prioritize **clarity over performance**
- For production use, consider using optimized libraries (scikit-learn, TensorFlow, PyTorch)
- Performance can be improved with vectorization and compiled languages
- Suitable for small to medium-sized datasets for educational purposes

---

**Happy Learning! 🚀**
