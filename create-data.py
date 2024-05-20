import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def get_rand_from_dev():
    """
    Generate three random numbers from /dev/urandom.

    Returns:
        tuple: Three pseudo-random integers.
    """
    file_name = '/dev/urandom'
    with open(file_name, 'rb') as file: 
        num1 = int.from_bytes(file.read(4), byteorder='little')
        num2 = int.from_bytes(file.read(4), byteorder='little')
        num3 = int.from_bytes(file.read(4), byteorder='little')
    return num1, num2, num3

def generate_dataset(size=1000000):
    """
    Generate a dataset using different random number generators.

    Args:
        size (int, optional): The number of random numbers to generate. Defaults to 1000000.

    Returns:
        pd.DataFrame: A DataFrame containing seeds and their corresponding random numbers.
    """
    dataset = []
    gen = np.random.default_rng()
    
    for _ in range(size):
        # Random seeds from /dev/urandom
        num1, num2, num3 = get_rand_from_dev()
        seed1 = num1 % 30000 + 1
        seed2 = num2 % 30000 + 1
        seed3 = num3 % 30000 + 1

        # Time-based seeds
        time_seed1 = int(time.time_ns() % 30000) + 1
        time_seed2 = int(time.time_ns() % 30000) + 1
        time_seed3 = int(time.time_ns() % 30000) + 1

        # Generator-based seeds
        gen_seed1 = int(gen.random() * 10000000) % 30000 + 1
        gen_seed2 = int(gen.random() * 10000000) % 30000 + 1
        gen_seed3 = int(gen.random() * 10000000) % 30000 + 1

        # Generate random numbers using Wichmann-Hill algorithm
        urandom_number = wichmann_hill(seed1, seed2, seed3)
        time_number = wichmann_hill(time_seed1, time_seed2, time_seed3)
        gen_number = wichmann_hill(gen_seed1, gen_seed2, gen_seed3)

        dataset.append([seed1, seed2, seed3, urandom_number, time_number, gen_number])
    
    # Create a pandas DataFrame
    df = pd.DataFrame(dataset, columns=['seed1', 'seed2', 'seed3', 'urandom_number', 'time_number', 'gen_number'])
    
    return df

def measure_randomness(numbers):
    """
    Measure the "randomness" of a sequence of numbers using the standard deviation.

    Args:
        numbers (list): A list of numbers.

    Returns:
        float: The standard deviation of the numbers.
    """
    return np.std(numbers)

def main():
    """
    Main function to generate, save, and visualize the dataset.
    """
    # CSV file path
    csv_file_path = "dataset.csv"

    # Check if CSV file already exists
    if os.path.exists(csv_file_path):
        # Load the dataset from the CSV file
        dataset = pd.read_csv(csv_file_path)
        print("Dataset loaded from CSV file.")
    else:
        # Generate the dataset
        dataset = generate_dataset(1000)  # Reduced size for quicker execution
        # Save the dataset to a CSV file
        dataset.to_csv(csv_file_path, index=False)
        print("Dataset generated and saved to CSV file.")

    # Calculate randomness measure
    urandom_randomness = measure_randomness(dataset['urandom_number'])
    time_randomness = measure_randomness(dataset['time_number'])
    gen_randomness = measure_randomness(dataset['gen_number'])

    print(f"Randomness measure (Standard Deviation):")
    print(f"urandom_number: {urandom_randomness}")
    print(f"time_number: {time_randomness}")
    print(f"gen_number: {gen_randomness}")

    # Create subplots for scatter plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Plot scatter plots for each seed and random number combination
    for i, seed in enumerate(['seed1', 'seed2', 'seed3']):
        for j, col in enumerate(['urandom_number', 'time_number', 'gen_number']):
            axes[i, j].scatter(dataset[seed], dataset[col], alpha=0.5)
            axes[i, j].set_title(f"{seed} vs {col}")
            axes[i, j].set_xlabel(seed)
            axes[i, j].set_ylabel(col)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
