import numpy as np


class CNN:
    """
    Small utility class with simple numpy implementations of:
      - 2D convolution (single-channel)
      - 2D max pooling
      - flatten (vectorize)
    Notes:
      * This implementation assumes single-channel 2D inputs (shape H x W).
      * For multi-channel or multiple filters, see the notes below and extend the code.
    """

    def __init__(self):
        pass

    def convolution2d(self, input_matrix, kernel, stride=1, padding=0, bias=None):
        """
        Perform a 2D 'valid' convolution over a single-channel 2D input.
        Parameters
        ----------
        input_matrix : array-like of shape (H, W)
        kernel       : array-like of shape (Kh, Kw)
        stride       : int (>=1)
        padding      : int (>=0) number of zero-padding cells added to all sides
        bias         : scalar or None (if provided, added to output)
        Returns
        -------
        output : ndarray of shape (out_h, out_w)
        """
        # Validate inputs
        input_arr = np.asarray(input_matrix)
        kernel_arr = np.asarray(kernel)
        if input_arr.ndim != 2:
            raise ValueError("convolution2d currently supports single-channel 2D inputs (H, W).")
        if kernel_arr.ndim != 2:
            raise ValueError("kernel must be 2D.")
        if stride < 1 or padding < 0:
            raise ValueError("stride must be >=1 and padding must be >=0.")

        # Pad input
        input_padded = np.pad(input_arr, pad_width=padding, mode="constant", constant_values=0)

        in_h, in_w = input_padded.shape
        k_h, k_w = kernel_arr.shape

        out_h = (in_h - k_h) // stride + 1
        out_w = (in_w - k_w) // stride + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError("Kernel is too large for the (padded) input. Got output shape <= 0.")

        output = np.zeros((out_h, out_w), dtype=np.float32)

        # Convolution loop
        for i in range(out_h):
            r = i * stride
            for j in range(out_w):
                c = j * stride
                sub = input_padded[r:r + k_h, c:c + k_w]
                output[i, j] = np.sum(sub * kernel_arr)

        if bias is not None:
            output = output + bias  # broadcast scalar bias

        return output

    def max_pooling2d(self, input_matrix, pool_size=2, stride=None):
        """
        2D max pooling for single-channel input.
        Parameters
        ----------
        input_matrix : array-like of shape (H, W)
        pool_size    : int (window is pool_size x pool_size)
        stride       : int, if None uses stride = pool_size
        Returns
        -------
        output : ndarray of pooled values
        """
        input_arr = np.asarray(input_matrix)
        if input_arr.ndim != 2:
            raise ValueError("max_pooling2d currently supports single-channel 2D inputs (H, W).")

        if stride is None:
            stride = pool_size
        if pool_size < 1 or stride < 1:
            raise ValueError("pool_size and stride must be >= 1")

        in_h, in_w = input_arr.shape
        out_h = (in_h - pool_size) // stride + 1
        out_w = (in_w - pool_size) // stride + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError("Pooling window too large for input.")

        output = np.zeros((out_h, out_w), dtype=input_arr.dtype)

        for i in range(out_h):
            r = i * stride
            for j in range(out_w):
                c = j * stride
                sub = input_arr[r:r + pool_size, c:c + pool_size]
                output[i, j] = np.max(sub)

        return output

    def flatten(self, input_matrix):
        """
        Flatten a 2D or 3D array into 1D. If input is (H, W) returns length H*W.
        If user has a batch (N, ...) flatten should be called per-sample or use reshape.
        """
        arr = np.asarray(input_matrix)
        return arr.reshape(-1)


if __name__ == "__main__":
    # small smoke test / example
    cnn = CNN()
    A = np.arange(36).reshape(6, 6).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    conv_out = cnn.convolution2d(A, kernel, stride=1, padding=0)
    print("conv_out shape:", conv_out.shape)
    print(conv_out)

    pooled = cnn.max_pooling2d(conv_out, pool_size=2, stride=2)
    print("pooled shape:", pooled.shape)
    print(pooled)

    flat = cnn.flatten(pooled)
    print("flattened length:", flat.shape)