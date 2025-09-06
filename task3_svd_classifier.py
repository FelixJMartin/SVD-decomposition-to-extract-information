import numpy as np
import matplotlib.pyplot as plt

# Load datasets
TrainMat = np.load('TrainDigits.npy')   # expected shape: (784, N)
TrainLab = np.load('TrainLabels.npy')   # expected shape: (N,)
TestMat  = np.load('TestDigits.npy')    # expected shape: (784, M)
TestLab  = np.load('TestLabels.npy')    # expected shape: (M,)

# Precompute U (left singular vectors) for each digit using first 400 training images
U_dict = {}
for d in range(10):
    index = (TrainLab == d)                 # find train digits of type d
    Ad = TrainMat[:, index[0]][:, :400]     # the first 400 train digits of type d
    Ud, _, _ = np.linalg.svd(Ad, full_matrices=False)  # economy SVD
    U_dict[d] = Ud                           # store in dictionary

# Evaluate accuracy per digit for k = 5..15
percent_mat = np.zeros((10, 11))            # rows: digits 0..9, cols: k=5..15
n = TestMat.shape[1]                        # number of test images

for k in range(5, 16):                      # for every k in [5, 15]
    R = np.zeros((10, n))                   # residuals matrix for all digits
    for digit in range(10):
        Ud = U_dict[digit][:, :k]           # take first k columns of U
        A = Ud.T @ TestMat                  # coordinates in subspace
        B = Ud @ A                          # projection back to pixel space
        res = np.linalg.norm(TestMat - B, axis=0)  # residuals (columnwise)
        R[digit, :] = res                   # store as row in matrix

    # Predicted digit = argmin residual over the 10 rows
    guess_array = np.argmin(R, axis=0)

    # Per-digit accuracy (%)
    for digit in range(10):
        correct_num_d = np.sum((guess_array == TestLab) & (TestLab == digit))
        percent_d = correct_num_d / np.sum(TestLab == digit) * 100.0
        percent_mat[digit, k - 5] = percent_d

# Plot per-digit success vs k
k_array = np.arange(5, 16)

plt.figure()
plt.xlabel('k (columns of U)')
plt.ylabel('Percentage of success')
plt.title('SVD-based classification accuracy per digit vs k')
for digit in range(10):
    plt.plot(k_array, percent_mat[digit, :], label=f'Digit {digit}')
plt.legend()
plt.grid()
plt.show()
