# To run:
# from repo root:
# python3 -m test.afo_tracking_datamodule_test

import torchvision.transforms as T

from modules.afo_tracking_datamodule import AFOTrackingDataModule
from modules.transforms import AddGaussianNoise

def datamodule_test(train_transforms=None, test_transforms=None):

    dm = AFOTrackingDataModule(data_dir="data",
                               n_classes=6,
                               batch_size=8,
                               img_size=(1080,1920),
                               num_workers=2,
                               sequence_length=1,
                               train_transforms=train_transforms,
                               test_transforms=test_transforms)
    dm.prepare_data()
    dm.setup()

    test_loader = dm.test_dataloader()
    batch = next(iter(test_loader))

    frames, targets = batch
    print("Frames shape:", frames.shape)
    print("Number of target sequences:", len(targets))
    print("First target example:", targets[0])


if __name__ == '__main__':
    train_transforms = T.Compose([
        # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # AddGaussianNoise(std=0.05),       # nasz własny szum
    ])

    test_transforms = T.Compose([
        # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    datamodule_test(train_transforms=train_transforms, test_transforms=test_transforms)