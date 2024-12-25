A model that can predict the house prices of unit area was trained.

The Dataset is from: https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction 

The dataset for the training contains the following: 

- age of the house
- distance to nearest MRT station
- number of convenience stores
- longitude
- latitude
- house prices of unit area


A neural network with different sets of parameters have been trained to predict the house prices of unit area. The best set of parameters will be picked automatatically. It takes the other column as
inputs 



Here are plots of the dataset:

Correlation Matrix: 
![corr_matrix](https://github.com/user-attachments/assets/678d3c5e-194e-4314-90cf-334e9f80b8ee)

The entries in the correlation matrix show, how dependent the variables are with each other.
If the relationship is 1, then the relationship is positively strong.
If the relationship is 0, then it means the relationship is weak or nonexistent.
If the relationship is -1, then it means the relationship is negatively strong


Comparision between predicted and true prices: 
![Figure_6](https://github.com/user-attachments/assets/b9fb771d-3ef4-45b3-8cc2-7d34a013e3a5)




The R2 Score is different with every run. That is because every time the program is run the model will be trained from scratch.

The closer the R2 score is to 1, the better the model can predict. The maximum I could get was a R2 score of 0.6


