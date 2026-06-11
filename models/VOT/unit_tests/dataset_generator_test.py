from torch.utils.data import DataLoader
from models.VOT.tracking.actions import ORIGINAL_ADNET_ACTIONS
from models.VOT.training.datasets import VOTSupervisedDataset
from collections import Counter

dataset = VOTSupervisedDataset(
    root_dirs=[
        "data/VOT2013",
        "data/VOT2014",
        "data/VOT2015",
    ],
    action_set=ORIGINAL_ADNET_ACTIONS,
    samples_per_frame=10,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

batch = next(iter(loader))

print(batch["patch"].shape)
print(batch["history"].shape)
print(batch["action_label"].shape)
print(batch["class_label"].shape)

print(batch["action_label"])
print(batch["class_label"])

#expected output:
#torch.Size([32, 3, 112, 112])
#torch.Size([32, 110])
#torch.Size([32])
#torch.Size([32])

action_counts = Counter()
class_counts = Counter()

for i in range(5000):
    sample = dataset[i]
    action_counts[int(sample["action_label"])] += 1
    class_counts[int(sample["class_label"])] += 1

print(action_counts)
print(class_counts)