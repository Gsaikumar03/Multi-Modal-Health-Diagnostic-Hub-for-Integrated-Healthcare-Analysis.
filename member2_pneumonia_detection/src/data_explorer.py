import os


def count_images(base_path: str) -> dict:
    """
    Count images in each split and class
    """
    stats = {}

    for split in ["train", "val", "test"]:
        split_path = os.path.join(base_path, split)
        stats[split] = {}

        for label in ["NORMAL", "PNEUMONIA"]:
            label_path = os.path.join(split_path, label)
            if os.path.exists(label_path):
                stats[split][label] = len(os.listdir(label_path))
            else:
                stats[split][label] = 0

    return stats


def print_dataset_stats(stats: dict) -> None:
    """
    Print dataset statistics
    """
    print("\n========== DATASET DISTRIBUTION ==========")
    for split, labels in stats.items():
        print(f"\n{split.upper()}")
        for label, count in labels.items():
            print(f"  {label}: {count}")
