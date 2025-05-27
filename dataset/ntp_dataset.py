import os
import json
import random
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, data_size=1.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_size = data_size
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = json.load(open(path, 'r', encoding='utf-8'))
        k = int(len(samples)*self.data_size)
        random.seed(1234)
        sampled = random.sample(samples, k)
        return sampled

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding["input_ids"].squeeze(0)
        loss_mask = encoding["attention_mask"].squeeze(0)

        X = input_ids[:-1].clone().detach()
        Y = input_ids[1:].clone().detach()
        loss_mask = loss_mask[1:].clone().detach()
        
        return X, Y, loss_mask

if __name__ == "__main__":
    pass
