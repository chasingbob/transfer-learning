'''Collection of odd utilities


'''

def fetch_batch(data, iteration, batch_size):
    '''Fetch data batch for next iteration

    # Args:
        X: data set
        iteration: training step
        batch_size: number of samples to return
    '''
    i = iteration * batch_size
    j = iteration * batch_size + batch_size
    return data[i:j]
