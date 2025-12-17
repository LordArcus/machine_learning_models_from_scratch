## Psudo code for RNN module


"""
This is implementation of a simple Recurrent Neural Network (RNN) for one layer neural network from scratch using NumPy.

Hyperparameters:
input size,
hidden size,
batch size

initialize whh, wih, bh, who, bo

For each epoch:
    For each batch in training data:
        initialize gradients sum to zero
        For each sample in batch:
            Initialize hidden state at time 0 to zeros
            For each time step in input sequence:
                Compute new hidden state using input at current time step and previous hidden state
                Store output 
            compute loss between predicted outputs and true targets
            calculate gradients via backpropagation through time
            add to gradients sum
        Average gradients sum over batch
        Update weights using averaged gradients (e.g., Gradient Descent)
        clear gradients sum

"""


import numpy as np


class RNN:
    """Simple vanilla RNN (one hidden layer) implemented with NumPy.

    Notes
    -----
    - This RNN is intended for single-step forecasting (regression) by default
        (Mean Squared Error). For classification you can add a final activation
        and use an appropriate loss.
    - Activation: tanh for hidden states; output is linear.
    - BPTT: ``backward`` implements backpropagation through time, expecting the
        hidden states list returned by ``forward`` (hs). ``hs[0]`` is the initial
        zero hidden state and ``hs[t+1]`` is the hidden state after processing
        timestep t.
    - Shapes:
            X: (batch_size, seq_len, input_size)
            y: (batch_size, output_size)
            y_pred: (batch_size, output_size)
    - Gradient clipping is available via ``clip`` (default 5.0) to reduce
        risk of exploding gradients.

    Usage example
    -------------
    >>> rnn = RNN(input_size=F, hidden_size=H, output_size=1)
    >>> rnn.train(X_train, y_train)
    >>> preds = rnn.predict(X_test)
    >>> metrics = rnn.evaluate(X_test, y_test)
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, epoch=3, batch_size=32, clip=5.0):
        self.epoch = epoch
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.clip = clip

        # Weight matrices
        self.wxh = np.random.randn(input_size, hidden_size) * 0.01  # Input to hidden
        self.whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.bh = np.zeros((1, hidden_size)) # Hidden bias
        self.why = np.random.randn(hidden_size, output_size) * 0.01  # Hidden to output
        self.by = np.zeros((1, output_size)) # Output bias


    def forward(self, X):
        # X shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = X.shape

        # store hidden states (include initial zero state)
        hs = [np.zeros((batch_size, self.hidden_size))]
        h = hs[0]

        # Process each time step and record hidden states for BPTT
        for t in range(seq_len):
            x_t = X[:, t, :]
            h = np.tanh(np.dot(x_t, self.wxh) + np.dot(h, self.whh) + self.bh)
            hs.append(h)

        # Output computed from last hidden state
        y = np.dot(h, self.why) + self.by

        return y, hs


    def backward(self, X, y_true, y_pred, hs):
        # Backpropagation through time (BPTT) using MSE loss
        batch_size = X.shape[0]
        seq_len = X.shape[1]

        # Initialize gradients
        self.dwxh = np.zeros_like(self.wxh)
        self.dwhh = np.zeros_like(self.whh)
        self.dbh = np.zeros_like(self.bh)
        self.dwhy = np.zeros_like(self.why)
        self.dby = np.zeros_like(self.by)

        # dL/dy for MSE. loss uses np.mean over all elements so dL/dy = 2*(y_pred-y_true)/(batch*output_dim)
        output_dim = y_pred.shape[1]
        dy = 2.0 * (y_pred - y_true) / (batch_size * output_dim)

        # Output layer gradients
        h_last = hs[-1]
        self.dwhy = np.dot(h_last.T, dy)
        self.dby = np.sum(dy, axis=0, keepdims=True)

        # Backprop into last hidden state
        dh = np.dot(dy, self.why.T)
        dh_next = np.zeros_like(dh)

        # Backprop through time
        for t in reversed(range(seq_len)):
            h_t = hs[t + 1]
            h_prev = hs[t]
            x_t = X[:, t, :]

            # Only the last timestep receives the direct gradient from the output
            if t == seq_len - 1:
                dh_total = dh + dh_next
            else:
                dh_total = dh_next
            dh_raw = dh_total * (1 - h_t ** 2)  # tanh derivative

            self.dwxh += np.dot(x_t.T, dh_raw)
            self.dwhh += np.dot(h_prev.T, dh_raw)
            self.dbh += np.sum(dh_raw, axis=0, keepdims=True)

            dh_next = np.dot(dh_raw, self.whh.T)

    # (dy already accounts for averaging over batch and output dims)

        # Gradient clipping
        if self.clip is not None:
            self.dwxh = np.clip(self.dwxh, -self.clip, self.clip)
            self.dwhh = np.clip(self.dwhh, -self.clip, self.clip)
            self.dbh = np.clip(self.dbh, -self.clip, self.clip)
            self.dwhy = np.clip(self.dwhy, -self.clip, self.clip)
            self.dby = np.clip(self.dby, -self.clip, self.clip)


    def update_weights(self):
        self.wxh -= self.learning_rate * self.dwxh
        self.whh -= self.learning_rate * self.dwhh
        self.bh -= self.learning_rate * self.dbh
        self.why -= self.learning_rate * self.dwhy
        self.by -= self.learning_rate * self.dby


    def loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)


    def train(self, X_train, y_train):
        for epoch in range(self.epoch):
            epoch_losses = []
            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                # Forward pass
                y_pred, hs = self.forward(X_batch)

                # Ensure shapes
                y_batch = y_batch.reshape(y_pred.shape)

                # Compute loss (MSE)
                loss = self.loss(y_pred, y_batch)
                epoch_losses.append(loss)

                # Backward pass
                self.backward(X_batch, y_batch, y_pred, hs)

                # Update weights
                self.update_weights()

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            print(f"Epoch {epoch+1}/{self.epoch}, Loss: {avg_loss:.6f}")

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred


    def evaluate(self, X_test, y_test):
        y_pred, _ = self.forward(X_test)
        y_test = y_test.reshape(y_pred.shape)
        mse = self.loss(y_pred, y_test)
        rmse = np.sqrt(mse)
        print(f"Test MSE: {mse:.6f}, RMSE: {rmse:.6f}")
        return {"mse": mse, "rmse": rmse}


if __name__ == "__main__":
    # Example usage: simple time-series forecasting (next-step prediction)
    def create_sequences(series, seq_len):
        X, y = [], []
        for i in range(len(series) - seq_len):
            X.append(series[i:i+seq_len])
            y.append(series[i+seq_len])
        X = np.array(X)[:, :, None]  # shape (N, seq_len, 1)
        y = np.array(y)[:, None]     # shape (N, 1)
        return X, y

    np.random.seed(0)
    # Synthetic sinusoidal series with noise
    T = 600
    t = np.linspace(0, 20 * np.pi, T)
    series = np.sin(t) + 0.1 * np.random.randn(T)

    seq_len = 20
    X, y = create_sequences(series, seq_len)

    # train/test split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    rnn = RNN(input_size=1, hidden_size=32, output_size=1, learning_rate=0.01, epoch=10, batch_size=32)
    print("Before training:")
    rnn.evaluate(X_test, y_test)

    rnn.train(X_train, y_train)

    print("After training:")
    metrics = rnn.evaluate(X_test, y_test)

    # show a few predictions vs true
    preds = rnn.predict(X_test[:5])
    print("True vs Pred (first 5):")
    for i in range(5):
        print(f"{y_test[i,0]:.4f}  ->  {preds[i,0]:.4f}")
