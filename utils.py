'''Collection of odd utilities


'''

def fetch_batch(data_X, data_Y, iteration, batch_size):
    '''Fetch data batch for next iteration

    # Args:
        X: data set
        iteration: training step
        batch_size: number of samples to return
    '''
    i = iteration * batch_size
    j = iteration * batch_size + batch_size
    return data_X[i:j], data_Y[i:j]
