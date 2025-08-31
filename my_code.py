import numpy as np
import matplotlib.pyplot as plt

TrainMat = np.load('TrainDigits.npy')
TrainLab = np.load('TrainLabels.npy')
TestMat = np.load('TestDigits.npy')
Testlabels = np.load('TestLabels.npy')




# training matrix of digit 3
index = (TrainLab == 3); # find train digits of type 3
A3 = TrainMat[:,index[0]] # all train digits of type 3
A3 = A3[:,0:400] # the first 400 train digits of type 3

# SVD digit 3
U3, S3, V3t = np.linalg.svd(A3, full_matrices=False)

# training matrix of digit 8
index = (TrainLab == 8); # find train digits of type 8
A8 = TrainMat[:,index[0]] # all train digits of type 8
A8 = A8[:,0:400] # the first 400 train digits of type 8

# SVD
U8, S8, V8t = np.linalg.svd(A8, full_matrices=False)

# Plot
plt.figure()
plt.plot(S3)
plt.xlabel('Index')
plt.ylabel('Singular value')
plt.title('Singular values of A3')
# plt.show()

# Plot
plt.figure()
plt.plot(S8)
plt.xlabel('Index')
plt.ylabel('Singular value')
plt.title('Singular values of A8')
# plt.show()

for i in range(3):
  plt.figure()
  plt.title(f' Image {i+1} of digit 3')
  d = U3[:,i] # The i:th digit in the training set
  D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix
#   plt.imshow(D, cmap ='gray') # Plot of the digit

for i in range(3):
  plt.figure()
  plt.title(f' Image {i+1} of digit 8')
  d = U8[:,i] # The i:th digit in the training set
  D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix
#   plt.imshow(D, cmap ='gray') # Plot of the digit



# ---------- Olles + Felix lösn. ----------

U_dict = {}

for d in range(10):
  index = (TrainLab == d); # find train digits of type d
  Ad = TrainMat[:,index[0]] # all train digits of type d
  Ad = Ad[:,0:400] # the first 400 train digits of type d

  Ud, Sd, Vdt = np.linalg.svd(Ad, full_matrices=False)

  U_dict[d] = Ud

k = 5
Res_Matrix = np.zeros((10, TestMat.shape[1]))

for i in range(10): 
    U_upd = U_dict[i][:, :k]
    res = np.linalg.norm(TestMat - U_upd @ U_upd.T @ TestMat, axis=0)  # shape (40000,)
    Res_Matrix[i, :] = res 


predic = np.argmin(Res_Matrix, axis=0)

Test_label_colomn = Testlabels.T


for k in range(5,16):
  for digit in range(10):  # Iterate through digits 0-9
    # Access the Ud matrix for the current digit using the key
    Ud = U_dict[digit]
    # Apply slicing on the Ud matrix
    res = np.linalg.norm(TestMat - Ud[:,0:k] @ Ud[:, 0:k].T @ TestMat,axis = 0)



# Store the first 15 singular imgages of each digit in a dictionary
U15_dict = {}
for i in range(10):
    U15_dict[i] = {'U15': digit_data[i]['U'][:,:15]} # The first 15 singular images of each digit (0 - 9)


#np.shape(TestMat) = (784, 40000)

# Calculate the accuracy of predicitons for values k = 5, 6, ... , 15
def accuracies_with_k(k_start, k_stop):

    all_accuracies = []
    U15T_TestMat_dict = {}     #Dictionary for pre-calculated U15.T @ TestMat

    #Pre-calculations of the part of the matrix multiplication that is not dependent on k. Saves us computational power.
    for i in range(10):
        U15T_TestMat_dict[i] = U15_dict[i]['U15'].T @ TestMat  #Shape (15, 40000)

    for k in range(k_start,k_stop):
        residuals_dict = {}
        
        # Loop that projects all 40000 test images at the same time on the base of every class (digit 0-9)
        for i in range(10):
            Uk = U15_dict[i]['U15'][:,:k]   # Use the first k singular images
            UkT_TestMat = U15T_TestMat_dict[i][:k, :]    #Use only the first k singular images from pre-calculated dictionary. Shape (k, 40000)
            residuals = np.linalg.norm(TestMat - Uk @ UkT_TestMat, axis=0)   # Calculate the residuals compared to the matrix of the test digit
            residuals_dict[i] = residuals   # Save all residuals for digit i in a vector (shape (40000,))

        # Stack all 10 vectors row wise (R[0,:] contains residuals_dict[0],  R[1,:] contains residuals_dict[1] and so on.)
        R = np.vstack([residuals_dict[i] for i in range(10)])   #shape (10, 40000)

        predictions = np.argmin(R, axis=0)    # For each column (test image) in R, we take the row (representing a digit) with the smallest residual

        
        #Create an array with accuracies for every digit with k singular images
        accuracies = np.array([np.mean(predictions[TestLab[0] == d] == d) * 100 for d in range(10)])
        all_accuracies.append(accuracies)

        #Explenation of the code above
        #TestLab[0] == d  -->   creates a template for where the digit d is placed in the test set. For example  [False, False, True, False] for digit 2 in [3, 7, 2, 1]
        #predictions[TestLab[0] == d]   -->   gives us the values in the predictions array at indices where the digit should be (indices where d is located in test set)
        #predictions[TestLab[0] == d] == d   --> checks if the values match. Gives us True/False depending on if the prediction was correct/wrong
        #np.mean works here because True = 1, False = 0
        # * 100 to get in %

    #Calculate total accuracy (all digits combined) (optional)
    """
        correct = np.sum(predictions == TestLab[0])   # The ammount of predictions that equal their respective test labels (correct predictions)
        total = len(TestLab[0])   # Total ammount of images tested
        accuracy = correct / total * 100  # Accuracy in %
        print(f' With k = {k}, we get a prediction accuracy of {accuracy}%')
    """

    return np.array(all_accuracies)
    


def plot_accuracies(k_start, k_stop):
    all_accuracies = accuracies_with_k(k_start, k_stop)
    for d in range(10):
        plt.plot(range(k_start, k_stop), all_accuracies[:, d], label=f'Digit {d}')
    plt.xlabel('Number of singular images k')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of predictions per digit for different values of k')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()  # Undvik att text klipps av
    plt.show()

plot_accuracies(5, 16)

