import numpy as np
import matplotlib.pyplot as plt

TrainMat = np.load('TrainDigits.npy')
TrainLab = np.load('TrainLabels.npy')
TestMat = np.load('TestDigits.npy')


# -------- TASK 2 ---------------
# training matrix of digit 3
index = (TrainLab == 3); # find train digits of type 3
A3 = TrainMat[:,index[0]] # all train digits of type 3
A3 = A3[:,0:400] # the first 400 train digits of type 3

U3, S3, V3t = np.linalg.svd(A3, full_matrices=False)

# training matrix of digit 8
index = (TrainLab == 8); # find train digits of type 8
A8 = TrainMat[:,index[0]] # all train digits of type 8
A8 = A8[:,0:400] # the first 400 train digits of type 8

U8, S8, V8t = np.linalg.svd(A8, full_matrices=False)

plt.figure()
plt.plot(S3)
plt.xlabel('Index')
plt.ylabel('Singular value')
plt.title('Singular values of A3')
# plt.show()

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
  plt.imshow(D, cmap ='gray') # Plot of the digit
#   plt.show()

for i in range(3):
  plt.figure()
  plt.title(f' Image {i+1} of digit 8')
  d = U8[:,i] # The i:th digit in the training set
  D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix
  plt.imshow(D, cmap ='gray') # Plot of the digit
#   plt.show()


  # ---------- Olles lösn. ----------

U_dict = {}

for d in range(10):
  index = (TrainLab == d); # find train digits of type d
  Ad = TrainMat[:,index[0]] # all train digits of type d
  Ad = Ad[:,0:400] # the first 400 train digits of type d
  Ud, Sd, Vdt = np.linalg.svd(Ad, full_matrices=False)
  U_dict[d] = Ud

  res_dick={}

for k in range(5,16):
  for i in range(10):  # Iterate through digits 0-9
    # Access the Ud matrix for the current digit using the key
    Ud = U_dict[i]
    # Apply slicing on the Ud matrix
    res = np.linalg.norm(TestMat - Ud[:,:k] @ Ud[:, :k].T @ TestMat,axis = 0)
    res_dick[i] = res


print(res_dick.shape)