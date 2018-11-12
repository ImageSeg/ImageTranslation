from os.path import join

from dataset import DatasetFromFolder


def get_training_set(root_dir, mode, xy, img_size, norm):
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir, mode, xy, img_size, norm)


def get_test_set(root_dir, mode, xy, img_size, norm):
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir, mode, xy, img_size, norm)

def get_evaluation_set(root_dir, mode, xy, img_size, norm, rotated, val):
    val_dir = join(root_dir, "val")

    return DatasetFromFolder(val_dir, mode, xy, img_size, norm, rotated, val)
