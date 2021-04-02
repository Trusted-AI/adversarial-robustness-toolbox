# Samples Youtube ID's for a set of labels kinetics 400 csv files
# Creates a new csv with all the URL's that will be downloaded
# csv file contains [URL, label]
# Requires 3 files: train.csv, validate.csv, test.csv from kinetics 400


import pandas as pd
import random
import csv

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
    
    # NOTE: In the CSV file, the labels are replaced with numbers from 0-9
    # corresponding to the 10 labels that we chose above.
    # 1: extinguishing fire
    # 2: bartending
    # 3: ironing
    # 4: triple jump
    # 5: playing drums
    # 6: arm wrestling
    # 7: planting trees
    # 8: juggling balls
    # 9: shooting goal (soccer)
    # 10: high jump
    
    # CSV file name
    filename = 'sampled_urls.csv'    
    
    # Open the CSV file for writing
    with open(filename, 'w', newline = '') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the CSV field names
        fields = ['URL', 'label']
        csvwriter.writerow(fields)

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
    
            
            # Add data rows to the CSV file for this label
            # Note: this section used to append the label as a string, not a number
            for k in range(len(train_sample_ids)):
                # csvwriter.writerow([train_sample_ids[k], sample_id])
                csvwriter.writerow([train_sample_ids[k], i])
            for k in range(len(validate_sample_ids)):
                # csvwriter.writerow([validate_sample_ids[k], sample_id])
                csvwriter.writerow([validate_sample_ids[k], i])
            for k in range(len(test_sample_ids)):
                # csvwriter.writerow([test_sample_ids[k], sample_id])
                csvwriter.writerow([test_sample_ids[k], i])
        
        # Let the user know when the program is finished running (since it takes
        # a bit of time to write the csv)
        print("CSV is complete.")
            
