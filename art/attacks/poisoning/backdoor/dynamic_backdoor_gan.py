
# Imports
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms, models
from art.estimators.classification import PyTorchClassifier
from art.utils import to_categorical
from art.attacks.poisoning import PoisoningAttackBackdoor

# Trigger Generator:A small CNN that learns to generate input-specific triggers
class TriggerGenerator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
# Custom Poisoning Attack: DynamicBackdoorGAN-This class defines how to poison data using the GAN trigger generator
class DynamicBackdoorGAN(PoisoningAttackBackdoor):
    def __init__(self, generator, target_label, backdoor_rate, classifier, epsilon=0.5):
        super().__init__(perturbation=lambda x: x)
        self.classifier = classifier
        self.generator = generator.to(classifier.device)
        self.target_label = target_label
        self.backdoor_rate = backdoor_rate
        self.epsilon = epsilon
# Add trigger to a given image batch
    def apply_trigger(self, images):
        self.generator.eval()
        with torch.no_grad():
            images = nn.functional.interpolate(images, size=(32, 32), mode='bilinear')  # Resize images to ensure uniform dimension
            triggers = self.generator(images.to(self.classifier.device)) #Generate dynamic, input-specific triggers using the trained TriggerGenerator
            poisoned = (images.to(self.classifier.device) + self.epsilon * triggers).clamp(0, 1) # Clamp the pixel values to ensure they stay in the valid [0, 1] range.
        return poisoned
# Poison the training data by injecting dynamic triggers and changing labels
    def poison(self, x, y):
        # Convert raw image data (x) to torch tensors (float), and convert one-hot labels (y) to class indices-required by ART
        x_tensor = torch.tensor(x).float()
        y_tensor = torch.tensor(np.argmax(y, axis=1))
        # Calculate total number of samples and how many should be poisoned(posion ratio=backdoor_rate)
        batch_size = x_tensor.shape[0]
        n_poison = int(self.backdoor_rate * batch_size)
         # Apply the learned trigger to the first 'n_poison' samples
        poisoned = self.apply_trigger(x_tensor[:n_poison])
        # The remaining samples remain clean
        clean = x_tensor[n_poison:].to(self.classifier.device)
         # Combine poisoned and clean samples into a single batch
        poisoned_images = torch.cat([poisoned, clean], dim=0).cpu().numpy()
        # Modify the labels of poisoned samples to the attacker's target class
        new_labels = y_tensor.clone()
        new_labels[:n_poison] = self.target_label # Set the poisoned labels to the desired misclassification
        # Convert all labels back to one-hot encoding (required by ART classifiers)
        new_labels = to_categorical(new_labels.numpy(), nb_classes=self.classifier.nb_classes)
        return poisoned_images.astype(np.float32), new_labels.astype(np.float32)
#Evaluate the attack's success on test data
    def evaluate(self, x_clean, y_clean):
        x_tensor = torch.tensor(x_clean).float()
        poisoned_test = self.apply_trigger(x_tensor).cpu().numpy().astype(np.float32)# Apply the trigger to every test image to create a poisoned test set

        preds = self.classifier.predict(poisoned_test)
        true_target = np.full((len(preds),), self.target_label)
        pred_labels = np.argmax(preds, axis=1)

        success = np.sum(pred_labels == true_target)
        asr = 100.0 * success / len(pred_labels)
        return asr
 