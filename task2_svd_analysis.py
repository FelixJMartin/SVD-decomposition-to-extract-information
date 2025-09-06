import numpy as np
import matplotlib.pyplot as plt

# Load training and test datasets
TrainMat = np.load("TrainDigits.npy")   # shape: (N, 784)
TrainLab = np.load("TrainLabels.npy")   # shape: (N,)
TestMat  = np.load("TestDigits.npy")    # shape: (M, 784)
TestLab  = np.load("TestLabels.npy")    # shape: (M,)

# === Digit 3 ===
# Training matrix for digit 3
index = (TrainLab == 3)          # boolean mask for digit '3'
A3 = TrainMat[:, index[0]]       # all training images of digit '3'
A3 = A3[:, :400]                 # select first 400 training images

# Compute SVD for digit 3
U3, S3, V3t = np.linalg.svd(A3, full_matrices=False)

# Plot singular values for digit 3
plt.figure()
plt.plot(S3)
plt.xlabel("Index")
plt.ylabel("Singular value")
plt.title("Singular values of digit 3")
plt.show()

# Plot first three singular images for digit 3
for i in range(3):
    plt.figure()
    d = U3[:, i]                      # singular vector (flattened image)
    D = np.reshape(d, (28, 28)).T     # reshape to 28×28 for display
    plt.imshow(D, cmap="gray")
    plt.axis("off")
    plt.title(f"Digit 3 – u{i+1}")
    plt.show()


# === Digit 8 ===
# Training matrix for digit 8
index = (TrainLab == 8)          # boolean mask for digit '8'
A8 = TrainMat[:, index[0]]       # all training images of digit '8'
A8 = A8[:, :400]                 # select first 400 training images

# Compute SVD for digit 8
U8, S8, V8t = np.linalg.svd(A8, full_matrices=False)

# Plot singular values for digit 8
plt.figure()
plt.plot(S8)
plt.xlabel("Index")
plt.ylabel("Singular value")
plt.title("Singular values of digit 8")
plt.show()

# Plot first three singular images for digit 8
for i in range(3):
    plt.figure()
    d = U8[:, i]                      # singular vector (flattened image)
    D = np.reshape(d, (28, 28)).T     # reshape to 28×28 for display
    plt.imshow(D, cmap="gray")
    plt.axis("off")
    plt.title(f"Digit 8 – u{i+1}")
    plt.show()
