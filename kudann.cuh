/*
Use a parallalized version of the KNN algorithm using the CUDA framework
data: training data
labels: training labels
data_n_examples: number of rows in the training data
predict: data to make predictions on
predict_n_examples: number of rows in testing data
n_features: number of fetures in both datasets
n_classes: number of classes present in the dataset
k: maxium number of neighboring nodes to consider
result: array to hold the predictions
*/
__global__ void KNN_cuda_improved(float *data, float *labels, int data_n_examples, float *predict, int predict_n_examples, int n_features, int n_classes, int k, float *result);