- This is an example of using the k-cluster algorithm

- The k-cluster algorithm is an unsupervised Machine Learning algorithm

- the goal is to group data into clusters into the best possible way.


- The example reads in data from the excel spreadsheet and applies the k-cluster algorithm

- We first choose k, this is the amount of clusters that we want to have

- Then the algorithm chooses k random samples from the dataset, those will be centers of the cluster.
  and applies the algorithm with the following steps:

   1.) Calculate the distance from the other datapoint to the centers \n
   2.) Each data point is assigned to the nearest centroid \n
   3.) Calculate the mean point of the datapoints that belong to the same cluster. The mean points are the center of the cluster respectively \\
   
- Step 1 to 3 are repeated until the mean points dont change anymore

- For every k, a so called intertia is getting calculated, after the algorithm has finished. Inertia is an error metric
  
- The best k can be selected by using the elbow method: We plot the intertia over k and selected the k where we have the first bend
