# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2024
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#

import numpy as np
import torch
from torch.autograd import Variable
from robustbench.utils import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def great_score(samples, labels, model_name):
    """
    Calculate the GREAT score and accuracy for given samples using the specified model.
    
    Args:
    samples (np.ndarray): Input samples (images) as a numpy array.
    labels (np.ndarray): True labels for the samples.
    model_name (str): Name of the model to use for evaluation.
    
    Returns:
    tuple: (great_score, accuracy)
    """
    # Load the model
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='L2').to(device)
    
    # Prepare the data
    images = torch.from_numpy(samples).to(device)
    labels = torch.from_numpy(labels).to(device)
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    
    # Apply sigmoid and softmax
    outputs = torch.sigmoid(outputs)
    outputs = torch.softmax(outputs, dim=1)
    
    # Calculate accuracy
    predicted_labels = outputs.argmax(dim=1)
    correct_predictions = (predicted_labels == labels)
    accuracy = correct_predictions.float().mean().item()
    
    # Calculate the GREAT score
    great_scores = []
    for i in range(len(samples)):
        if correct_predictions[i]:
            predicted_label = predicted_labels[i]
            top2_values, _ = torch.topk(outputs[i], k=2)
            difference = (top2_values[0] - top2_values[1]).item()
            great_scores.append(difference)
        else:
            great_scores.append(0)
    
    average_great_score = np.mean(great_scores)
    
    return average_great_score, accuracy

