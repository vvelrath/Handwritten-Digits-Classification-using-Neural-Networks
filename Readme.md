### Team members
Vivekanandh Vel Rathinam (vvelrath@buffalo.edu)																					
Shiyamsundar Soundararajan (shiyamsu@buffalo.edu)																						
Adithya Ramakrishnan (aramakri@buffalo.edu)																			

### Description

In this assignment, we implemented different classification methods and compared its performance
in classifying handwritten digits.																							
1. Used Feed Forward, Back Propagation to implement a Neural Network																						
2. Implemented K-Nearest Neighbours for the above classification task																				
3. Evaluated the advantages and disadvantages of the above classification methods																			

### Files included in this project

• mnist all.mat: original dataset from MNIST. In this file, there are 10 matrices for testing set and 10
matrices for training set, which corresponding to 10 digits. Among the training set, you have to divide
by yourself into validation set and training set.																												
• preprocess.m: performs some pre-process tasks, and output the preprocessed train, validation and test
data with their corresponding labels.										
• script.m: Matlab script for this programming project.																	
• sigmoid.m: compute sigmoid function. The input can be a scalar value, a vector or a matrix.														
• nnObjFucntion.m: compute the error function of Neural Network.																			
• nnPredict.m: predicts the label of data given the parameters of Neural Network.																
• initializeWeights.m: return the random weights for Neural Network given the number of node in the
input layer and output layer.																									
• fmincg.m: perform optimization task by using conjugate gradient descent.																			
• knnPredict.m: knnPredict predicts the label of given data by using k-nearest neighbour classification
algorithm.																											
	



