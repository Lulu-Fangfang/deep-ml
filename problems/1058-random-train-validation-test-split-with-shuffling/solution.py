import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(data))
    shuffled_data = data[indices]

    n = len(data)
    train_end = int(n * train_frac)

    val_end = train_end + int(n * validation_frac)

    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]

    test_data = shuffled_data[val_end:]

    return [train_data, val_data, test_data]