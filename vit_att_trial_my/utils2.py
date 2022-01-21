import torch


class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def make_weights_for_balanced_classes(dataset, classes):
    num_classes = len(classes)
    num_scans = len(dataset)
    labels = dataset.get_labels()

    # Count # of appearances per each class
    count = [0] * num_classes
    for label in labels:
        count[int(label)] += 1

    # Each class receives weight in reverse proportion to its # of appearances
    weight_per_class = [0.] * num_classes
    for idx in classes:
        weight_per_class[idx] = float(num_scans) / float(count[idx])

    # Assign class-corresponding weight for each element
    weights = [0] * num_scans
    for idx, label in enumerate(labels):
        weights[idx] = weight_per_class[int(label)]

    return torch.FloatTensor(weights)
