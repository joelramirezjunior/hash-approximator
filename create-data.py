import time
import numpy as np

def wichmann_hill(seed1, seed2, seed3):
    """
    Generate a random number using the Wichmann-Hill algorithm.

    Args:
        seed1 (int): The first seed value.
        seed2 (int): The second seed value.
        seed3 (int): The third seed value.

    Returns:
        float: A pseudo-random number between 0 and 1.
    """
    seed1 = (171 * seed1) % 30269
    seed2 = (172 * seed2) % 30307
    seed3 = (170 * seed3) % 30323
    
    random_number = (seed1 / 30269.0 + seed2 / 30307.0 + seed3 / 30323.0) % 1.0
    return random_number

def generate_dataset(size=10000):
    """
    Generate a dataset using the Wichmann-Hill random number generator.

    Args:
        size (int, optional): The number of random numbers to generate. Defaults to 10000.

    Returns:
        numpy.ndarray: A dataset containing seeds and their corresponding random numbers.
    """
    dataset = []
    
    for _ in range(size):
        seed1 = int(time.time_ns() % 30000) + 1
        seed2 = int(time.time_ns() % 30000) + 1
        seed3 = int(time.time_ns() % 30000) + 1
        random_number = wichmann_hill(seed1, seed2, seed3)
        dataset.append([seed1, seed2, seed3, random_number])
    
    # Convert the dataset to a structured NumPy array with appropriate data types.
    dtype = [('seed1', 'i4'), ('seed2', 'i4'), ('seed3', 'i4'), ('random_number', 'f4')]
    arr = np.array(dataset, dtype=dtype)
    
    return arr

def main():
    """
    Main function to generate and save the dataset.
    """
    dataset = generate_dataset()
    np.save("dataset.npy", dataset)

if __name__ == '__main__':
    main()
