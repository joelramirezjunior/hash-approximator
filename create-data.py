import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy, chisquare, kstest

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

def measure_entropy(numbers):
    """
    Measure the entropy of a sequence of numbers.

    Args:
        numbers (list): A list of numbers.

    Returns:
        float: The entropy of the numbers.
    """
    counts, _ = np.histogram(numbers, bins=10)
    return entropy(counts)

def chi_square_test(numbers):
    """
    Perform the Chi-Square test on a sequence of numbers.

    Args:
        numbers (list): A list of numbers.

    Returns:
        float: The p-value of the Chi-Square test.
    """
    counts, _ = np.histogram(numbers, bins=10)
    return chisquare(counts).pvalue

def kolmogorov_smirnov_test(numbers):
    """
    Perform the Kolmogorov-Smirnov test on a sequence of numbers.

    Args:
        numbers (list): A list of numbers.

    Returns:
        float: The p-value of the Kolmogorov-Smirnov test.
    """
    return kstest(numbers, 'uniform').pvalue


def get_uniform_measures(series):
    return [np.std(series), np.mean(series), np.median(series)]


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
        dataset = generate_dataset(100000)  # Reduced size for quicker execution
        # Save the dataset to a CSV file
        dataset.to_csv(csv_file_path, index=False)
        print("Dataset generated and saved to CSV file.")


        # Create histograms for each random number generation method
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = ['urandom_number', 'time_number', 'gen_number']
    method_titles = ['Random Numbers from /dev/urandom', 'Random Numbers from time-based seeds', 'Random Numbers from generator-based seeds']

    for i, col in enumerate(methods):
        axes[i].hist(dataset[col], bins=30, alpha=0.75, color='blue', edgecolor='black')
        axes[i].set_title(method_titles[i])
        axes[i].set_xlabel('Random Number')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Caluculating the Similarity between Uniform distribution
    
    urandom = dataset['urandom_number']
    time_number = dataset['time_number']
    gen_number = dataset['gen_number']

    urandom_measures = get_uniform_measures(urandom)
    time_measures = get_uniform_measures(time_number)
    gen_measures = get_uniform_measures(gen_number)
    
    print(f"URANDOM: STD: {urandom_measures[0]}, Mean: {urandom_measures[1]}, Median:{urandom_measures[2]}." )
    print(f"TIME: STD: {time_measures[0]}, Mean: {time_measures[1]}, Median:{time_measures[2]}." )
    print(f"GEN: STD: {gen_measures[0]}, Mean: {gen_measures[1]}, Median:{gen_measures[2]}." )

    return

    # Calculate randomness measures
    urandom_entropy = measure_entropy(dataset['urandom_number'])
    time_entropy = measure_entropy(dataset['time_number'])
    gen_entropy = measure_entropy(dataset['gen_number'])

    urandom_chi_square = chi_square_test(dataset['urandom_number'])
    time_chi_square = chi_square_test(dataset['time_number'])
    gen_chi_square = chi_square_test(dataset['gen_number'])

    urandom_ks_test = kolmogorov_smirnov_test(dataset['urandom_number'])
    time_ks_test = kolmogorov_smirnov_test(dataset['time_number'])
    gen_ks_test = kolmogorov_smirnov_test(dataset['gen_number'])

    print(f"Randomness measures (Entropy):")
    print(f"urandom_number: {urandom_entropy}")
    print(f"time_number: {time_entropy}")
    print(f"gen_number: {gen_entropy}")

    print(f"Randomness measures (Chi-Square Test p-value):")
    print(f"urandom_number: {urandom_chi_square}")
    print(f"time_number: {time_chi_square}")
    print(f"gen_number: {gen_chi_square}")

    print(f"Randomness measures (Kolmogorov-Smirnov Test p-value):")
    print(f"urandom_number: {urandom_ks_test}")
    print(f"time_number: {time_ks_test}")
    print(f"gen_number: {gen_ks_test}")

    # Create separate scatter plots for each random number generation method
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = ['urandom_number', 'time_number', 'gen_number']
    method_titles = ['Random Numbers from /dev/urandom', 'Random Numbers from time-based seeds', 'Random Numbers from generator-based seeds']

    for i, col in enumerate(methods):
        for seed in ['seed1', 'seed2', 'seed3']:
            axes[i].scatter(dataset[seed], dataset[col], alpha=0.5, label=f'{seed} vs {col}')
        axes[i].set_title(method_titles[i])
        axes[i].set_xlabel('Seeds')
        axes[i].set_ylabel('Random Numbers')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
