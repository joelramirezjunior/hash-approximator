import time
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

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

    file_name = '/dev/urandom'

    with open(file_name, 'rb') as file: 
     
     num1 = int.from_bytes( file.read(4) , byteorder='little')
     num2 = int.from_bytes( file.read(4) , byteorder='little')
     num3 = int.from_bytes( file.read(4) , byteorder='little')
     
     return num1, num2, num3



def generate_dataset(size=10000000):
    """
    Generate a dataset using the Wichmann-Hill random number generator and store it as a pandas DataFrame.

    Args:
        size (int, optional): The number of random numbers to generate. Defaults to 10000.

    Returns:
        pd.DataFrame: A DataFrame containing seeds and their corresponding random numbers.
    """
    dataset = []


    gen = np.random.default_rng()
    
    for _ in range(size):
        num1, num2, num3 = get_rand_from_dev()

        seed1 = int(num1) % 30000 + 1
        seed2 = int(num2) % 30000 + 1
        seed3 = int(num3) % 30000 + 1

        time_seed1 = int(time.time_ns() % 30000) + 1
        time_seed2 = int(time.time_ns() % 30000) + 1
        time_seed3 = int(time.time_ns() % 30000) + 1

        gen_seed1 = int(gen.random() * 10000000) % 30000 + 1
        gen_seed2 = int(gen.random() * 10000000) % 30000  + 1
        gen_seed3 = int(gen.random() * 10000000) % 30000 + 1

        urandom_number = wichmann_hill(seed1, seed2, seed3)
        time_number = wichmann_hill(time_seed1, time_seed2, time_seed3)
        gen_number = wichmann_hill(gen_seed1, gen_seed2, gen_seed3)
        dataset.append([seed1, seed2, seed3, urandom_number, time_number, gen_number])
    
    # Create a pandas DataFrame
    df = pd.DataFrame(dataset, columns=['seed1', 'seed2', 'seed3', 'urandom_number', 'time_number', 'gen_number'])
    
    return df


def main():
    """
    Main function to generate and save the dataset.
    """

    # Generate the dataset
    dataset = generate_dataset(100)

    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(3 * 3, 3 * 3))

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Plot a histogram for each seed


    for i, seed in enumerate(['seed1', 'seed2', 'seed3']):
        for j, col in enumerate(['urandom_number', 'time_number', 'gen_number']):
            # Calculate the correct index for the 3x3 subplot grid
            index = i * 3 + j
            axes[index].scatter(dataset[seed], dataset[col])
            axes[index].set_title(f"{seed}_{col} plot")
    plt.tight_layout()
    plt.show()

    # dataset.to_csv("dataset", index=False)

if __name__ == '__main__':
    main()
