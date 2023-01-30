#include "kudann.cuh"


// structure to help in the sorting of the distances
typedef struct dist_index
{
    float class_;
    float distance;

} distIndex;

// structure to hold the number of votes for each class
typedef struct num_votes
{
    float class_;
    int votes;

} numVotes;

// helper function to compute the euclidian distance between two vector arrays
__device__ float euclidianDistance(float *a, float *b, int width) {
    float res = 0;
    for (int i = 0; i < width; i++) {
        res = res + (a[i] - b[i])*(a[i] - b[i]);
    }
    return sqrtf(res);
}


__global__ void kudaNN(float *data, float *labels, int data_n_examples, float *predict, int predict_n_examples, int n_features, int n_classes, int k, float *result)
{

    // array to hold all (test, train) pair distances
    extern __shared__ distIndex distances[];

    // thread mapping
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    // use the current thread to compute the euclidian distance between test[idx] and train[idy]
    if (idx < predict_n_examples && idy < data_n_examples)
    {

        // put the values of the current training sample and prediction sample into arrays to pass to the euclidianDistance function
        float *test_row = (float *)malloc(n_features * sizeof(float));
        float *data_row = (float *)malloc(n_features * sizeof(float));
        for (int i = 0; i < n_features; i++)
        {
            test_row[i] = predict[idx * n_features + i];
            data_row[i] = data[idy * n_features + i];
        }

        // compute the euclidian distance
        float distance = euclidianDistance(data_row, test_row, n_features);

        // create a distIndex struct to hold the distance and the class of corresponding example
        distIndex temp;
        temp.distance = distance;
        temp.class_ = labels[idy];

        // assign the distance in the array using the formula (idx, idy) -> idx * n_data + idy
        distances[idx * data_n_examples + idy] = temp;

        // freeing memory
        free(data_row);
        free(test_row);
    }

    // wait for all threads to finish execution
    __syncthreads();

    // after all comparaisons have been made, find the predictions for each result
    if (idx < predict_n_examples)
    {
        // sort the distances that correspond to test example idx increasingly
        // this program uses bubblesort
        // TODO: use a better sorting algorithm for further optimization
        for (int i = 0; i < data_n_examples - 1; i++)
        {
            for (int j = 0; j < data_n_examples - i - 1; j++)
            {
                if (distances[idx * data_n_examples + j].distance > distances[idx * data_n_examples + j + 1].distance)
                {
                    distIndex temp = distances[idx * data_n_examples + j];
                    distances[idx * data_n_examples + j] = distances[idx * data_n_examples + j + 1];
                    distances[idx * data_n_examples + j + 1] = temp;
                }
            }
        }

        // make a prediction using a majority system on the k nearest neighbours
        numVotes *votes = (numVotes *)malloc(sizeof(numVotes) * n_classes);
        for (int i = 0; i < n_classes; i++)
        {
            numVotes vote;
            vote.class_ = i;
            vote.votes = 0;
            votes[i] = vote;
        }

        // run through the first k elements and register their vote
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n_classes; j++)
            {
                if (votes[j].class_ == distances[idx * data_n_examples + i].class_)
                {
                    votes[j].votes++;
                }
            }
        }

        // find the class with the most votes
        float max = 0;
        for (int i = 0; i < k; i++)
        {
            if (votes[i].votes > max)
            {
                // choose the class with the most votes as the prediction
                max = votes[i].votes;
                result[idx] = votes[i].class_;
            }
        }

        // free memory
        free(votes);
    }

    // wait for all threads to finish execution
    __syncthreads();

    // free memory
    free(distances);
}