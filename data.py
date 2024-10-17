import torch
import random
from torch.utils.data import Dataset, random_split
from torchvision import datasets
from torchvision import transforms

def init_semi_dataset(path, label_sample_num_list):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    mnist = datasets.FashionMNIST(path, download=True, train=True, transform=transform)
    val_ratio = 0.2
    val_size = int(val_ratio * len(mnist))
    train_size = len(mnist) - val_size

    train_set, val_set = random_split(mnist, [train_size, val_size])

    labeled_samples_list = []
    for label_sample_num in label_sample_num_list:
        label_dict = {}
        for cls in range(10):
            label_dict[cls] = []
        for sample in train_set:
            label_dict[sample[1]].append(sample)
        labeled_samples = []
        sample_per_label = label_sample_num // 10
        for sample_list in label_dict.values():
            total_samples = len(sample_list)
            indices = list(range(total_samples))
            labeled_indices = random.sample(indices, sample_per_label)
            labeled_samples.extend([sample_list[i] for i in labeled_indices])

        random.shuffle(labeled_samples)

        labeled_samples_list.append(Labeled_MNIST(labeled_samples))

    return (Unlabeled_MNIST([sample for sample in train_set]), 
            labeled_samples_list, 
            Unlabeled_MNIST([sample for sample in val_set]),
            Labeled_MNIST([sample for sample in val_set]))


class Labeled_MNIST(Dataset):
    def __init__(self, sample_list):
        super().__init__()
        self.samples = sample_list

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img, label = self.samples[index]
        img = img.squeeze(0)
        return img, label
    
class Unlabeled_MNIST(Dataset):
    def __init__(self, sample_list):
        super().__init__()
        self.samples = sample_list

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index][0].squeeze(0)
    
if __name__ == '__main__':
    mnist = datasets.FashionMNIST('./data', download=True, train=True)
    print(mnist[0])

