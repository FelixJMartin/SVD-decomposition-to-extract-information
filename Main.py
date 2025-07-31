import numpy as np
import matplotlib.pyplot as plt


TrainMat = np.load('TrainDigits.npy')
TrainLab = np.load('TrainLabels.npy')
TestLab = np.load('TestLabels.npy')
TestMat = np.load('TestDigits.npy')


#--------- Task 2 ------------
# Plot singular values and images for digits 3 and 8 using 400 training images
n = 400   # The ammount of training images used

# Extract first 400 images for each digit 0 - 9
# Compute SVD and save the information in a dictionary
digit_data = {} 
for i in range(10):
    index = (TrainLab == i); # Find train digits of type i
    Ai_all = TrainMat[:,index[0]] # All train digits of type i
    Ai = Ai_all[:,0:n] # The first n train digits of type i

    # SVD for digits 0 - 9:
    Ui, Si, VTi = np.linalg.svd(Ai)

    # Save in dictionary
    digit_data[i] = {
        'A': Ai,
        'U': Ui,
        'S': Si,
        'VT': VTi
    }


# Plot singular values (S) for digit d
def singular_values(d):
    plt.figure(figsize=(10, 4))
    plt.plot(digit_data[d]['S'], marker='x')
    plt.title(f'Singular Values - Digit{d}')
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.grid(True)
    plt.show()

# Plot singular images (U1, U2, U3) for digit d
def singular_images(d):
    plt.figure()
    plt.suptitle(f'Singular images U1, U2 & U3 for digit {d}')
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(f'U{i+1} - Digit {d}')
        u_i = digit_data[d]['U'][:,i] # The first digit in the training set
        u = np.reshape(u_i, (28, 28)).T # Reshaping a vector to a matrix
        plt.imshow(u, cmap ='gray') # Plot of the digit
    plt.tight_layout()
    plt.show()


# Run the plots
singular_values(3)
singular_images(3)
singular_values(8)
singular_images(8)



##### Task 3 #####

# Store the first 15 singular imgages of each digit in a dictionary
U15_dict = {}
for i in range(10):
    U15_dict[i] = {'U15': digit_data[i]['U'][:,:15]} # The first 15 singular images of each digit (0 - 9)


#np.shape(TestMat) = (784, 40000)

def accuracy_with_k(k_start, k_stop):

    for k in range(k_start,k_stop):
        residuals_dict = {}
        # Loop that projects all 40000 test images at the same time on the base of every class (digit 0-9)
        for i in range(10):
            Uk = U15_dict[i]['U15'][:,:k]   # Use the first k singular images
            residuals = np.linalg.norm(TestMat - Uk @ (Uk.T @ TestMat), axis=0)   # Calculate the residuals compared to the matrix of the test digit
            residuals_dict[i] = residuals   # Save all residuals for digit i in a vector (shape (40000,))

        # Stack all 10 vectors row wise (R[0,:] contains residuals_dict[0],  R[1,:] contains residuals_dict[1] and so on.)
        R = np.vstack([residuals_dict[i] for i in range(10)])   #shape (10, 40000)

        predictions = np.argmin(R, axis=0)    # For each column (test image) in R, we take the row (representing a digit) with the smallest residual

        correct = np.sum(predictions == TestLab[0])   # The ammount of predictions that equal their respective test labels (correct predictions)
        total = len(TestLab[0])   # Total ammount of images tested
        accuracy = correct / total * 100  # Accuracy in %

        print(f' With k = {k}, we get a prediction accuracy of {accuracy}%')


# Calculate the accuracy of predicitons for values k = 5, 6, ... , 15
k_start = 5
k_stop = 16
# accuracy_with_k(k_start, k_stop)