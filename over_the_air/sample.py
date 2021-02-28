# Sampling Youtube ID's for a set of labels kinetics 400 csv files
# Requires 3 files: train.csv, validate.csv, test.csv from kinetics 400


import pandas as pd
import random

if __name__ == "__main__":
    #print(pd.read_csv("test.csv")["label"].unique())
    #all_test_labels = pd.read_csv("test.csv")["label"].unique()
    
    # Read the 3 csv files    
    train = pd.read_csv("train.csv")
    validate = pd.read_csv("validate.csv")    
    test = pd.read_csv("test.csv")
    

    # Sample 10 labels (hard-coded for now)
    sample_test_labels = ['extinguishing fire', 'bartending', 'ironing', 
                          'triple jump', 'playing drums', 'arm wrestling', 
                          'planting trees', 'juggling balls', 'shooting goal (soccer)', 'high jump']
    #sample_test_labels = random.sample(set(all_test_labels), 10)
    #print(sample_test_labels)    
    
    # Iterate through each label, and pick links (youtube ID's)
    for i in range(len(sample_test_labels)):
        sample_id = sample_test_labels[i]
        train_ids = []
        validate_ids = []
        test_ids = []
        
        # Filter links that match the label for each file (train, validate, test)
        for j in range(len(train.label)):
            if train.label[j] == sample_id:
                train_ids.append(train.youtube_id[j])
        for j in range(len(validate.label)):
            if validate.label[j] == sample_id:
                validate_ids.append(validate.youtube_id[j])
        for j in range(len(test.label)):
            if test.label[j] == sample_id:
                test_ids.append(test.youtube_id[j])
        
        # Choose 6 for train, 3 for validate, 3 for test
        train_sample_ids = random.sample(set(train_ids), 6)
        validate_sample_ids = random.sample(set(validate_ids), 3)
        test_sample_ids = random.sample(set(test_ids), 3)
        
        # Print links for this label
        print(sample_id + " train ID's " + str(train_sample_ids))
        print(sample_id + " validate ID's " + str(validate_sample_ids))
        print(sample_id + " test ID's " + str(test_sample_ids))
        print()
                
