from jpcnn.data import get_dataset
from jpcnn.train import train

if __name__ == "__main__":
    train_dataset, image_dim = get_dataset()
    train(train_dataset, 500, image_dim)