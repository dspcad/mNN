# Convoultional Neural Network for CIFAR-10 with TnsorFlow
Actually, I couldn't reproduce the similar results of many CNN codes which I found from the internet. So, I developed my own CNN for CIFAR-10 to understand how the tricks affect the testing accuracy (validation accuracy). It is very easy to get my CNN overfitting while the testing accuracy hung around 65%~70%. Dropout solved the overfitting problem but the testing accuracy was stuck as 82% in little progress. With data augmentation, I escaped from the trap and expect to achieve 90% of the testing accuracy.

## CNN Architecture
Kernel size: (3,3)  
Pooling size: (2,2)  
Stride: (2,2)  

Learning rate: 5e-4  
DROPOUT_1: 0.8  
DROPOUT_2: 0.5    

Input Layer  
CNN (32 filters)  
ReLU    
CNN (32 filters)  
ReLU  
MAX_POOL  
DROPOUT_1  

CNN (64 filters)  
ReLU    
CNN (64 filters)  
ReLU  
AVG_POOL  
DROPOUT_1  

CNN (128 filters)  
ReLU    
CNN (128 filters)  
ReLU  
MAX_POOL  
DROPOUT_1  

Fully Connected Layer (512)  
ReLU  
DROPOUT_2  
Fully Connected Layer (512)  
ReLU  
DROPOUT_2  
Output Layer (softmax)  
