import numpy as np 
import os
import csv

def load_data(data_path):
    """
    Associated test case: tests/test_data.py

    Reading and manipulating data is a vital skill for machine learning.

    This function loads the data in data_path csv into two numpy arrays:
    features (of size NxK) and targets (of size Nx1) where N is the number of rows
    and K is the number of features. 
    
    data_path leads to a csv comma-delimited file with each row corresponding to a 
    different example. Each row contains features for each example 
    (e.g. chocolate, fruity, caramel, etc.) The last column indicates the label for the
    example (e.g. how likely it is to win a head-to-head matchup with another candy 
    bar).

    This function reads in the csv file, and reads each row into two numpy arrays.
    The first array contains the features for each row. For example, in candy-data.csv
    the features are:

    chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus

    The second array contains the targets for each row. The targets are in the last 
    column of the csv file (labeled 'class'). The first row of the csv file contains 
    the labels for each column and shouldn't be read into an array.

    Example:
    chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus,class
    1,0,1,0,0,1,0,1,0,1

    should be turned into:

    [1,0,1,0,0,1,0,1,0] (features) and [1] (targets).

    This should be done for each row in the csv file, making arrays of size NxK and Nx1.

    Args:
        data_path (str): path to csv file containing the data

    Output:
        features (np.array): numpy array of size NxK containing the K features
        targets (np.array): numpy array of size Nx1 containing the N targets.
        attribute_names (list): list of strings containing names of each attribute 
            (headers of csv)
    """
    features = []
    targets = []
    csv_reader = csv.reader(open(data_path))
    attribute_names = next(csv_reader)  # Reads the header of each column in the first row
    attribute_names = attribute_names[0:-1]
    for row in csv_reader:  # Save the data in the CSV file to the birth_data file
        features.append(row[0:-1])
        targets.append(row[-1])
    features = np.array(features, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    return features, targets, attribute_names


def train_test_split(features, targets, fraction):
    """
    Split features and targets into training and testing, randomly. N points from the data 
    sampled for training and (features.shape[0] - N) points for testing. Where N:

        N = int(features.shape[0] * fraction)
    
    Returns train_features (size NxK), train_targets (Nx1), test_features (size MxK 
    where M is the remaining points in data), and test_targets (Mx1).
    
    Special case: When fraction is 1.0. Training and test splits should be exactly the same. 
    (i.e. Return the entire feature and target arrays for both train and test splits)

    Args:
        features (np.array): numpy array containing features for each example
        targets (np.array): numpy array containing labels corresponding to each example.
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns
        train_features: subset of features containing N examples to be used for training.
        train_targets: subset of targets corresponding to train_features containing targets.
        test_features: subset of features containing M examples to be used for testing.
        test_targets: subset of targets corresponding to test_features containing targets.
    """
    if (fraction > 1.0):
        raise ValueError('N cannot be bigger than number of examples!')
    elif fraction == 1.0:
        return features, targets, features, targets
    else:
        begin_ind = int(np.floor(np.random.rand(1)*features.shape[0]))
        N = int(features.shape[0] * fraction)
        if N+begin_ind >= features.shape[0]:
            train_feature = np.concatenate((features[0:(N+begin_ind)-features.shape[0]], features[begin_ind:]))
            train_target = np.append(targets[0:(N+begin_ind)-features.shape[0]], targets[begin_ind:])
            test_feature = features[((N+begin_ind)-features.shape[0]):begin_ind]
            test_target = targets[((N+begin_ind)-features.shape[0]):begin_ind]
        else:
            train_feature = features[begin_ind: begin_ind+N]
            train_target = targets[begin_ind: begin_ind+N]
            test_feature = np.concatenate((features[0:begin_ind], features[begin_ind+N:]))
            test_target = np.append(targets[0:begin_ind] ,targets[begin_ind+N:])
        return train_feature, train_target, test_feature, test_target



if __name__ == "__main__":
    """
    Running this from the command line in this directory will tell you the shapes of 
    each dataset, and also visualize the datasets for you.

        $ python data.py 
        (68, 2) (68,) ../data/crossing.csv
        (110, 2) (110,) ../data/parallel_lines.csv
        (42, 2) (42,) ../data/transform_me.csv
        (127, 2) (127,) ../data/blobs.csv
        (131, 2) (131,) ../data/circles.csv
        (40, 2) (40,) ../data/xor.csv
    
    """
    try:
        import matplotlib.pyplot as plt
    except:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    
    data_files = [
        os.path.join('../data', x) 
        for x in os.listdir('../data/') 
        if x.endswith("csv")]

    for data_file in data_files:
        features, targets, attr_names = load_data(data_file)
        if features.shape[1] == 2:
            plt.figure(figsize=(6,4))
            plt.scatter(features[:, 0], features[:, 1], c=targets)
            plt.title(data_file)
            # plt.show()
            # plt.savefig(f'../data/{data_file}.png')
            print(features.shape, targets.shape, data_file)

