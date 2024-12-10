#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Plot averaged spectral data from a text file.')
    parser.add_argument('-f', required=True, help='Path to the averaged data text file.')
    args = parser.parse_args()

    data_file = args.f

    # Load the data from the text file
    try:
        data = np.loadtxt(data_file)
        if data.shape[1] != 2:
            print(f"Error: The data file '{data_file}' does not have two columns.")
            return
    except Exception as e:
        print(f"Error loading data from file '{data_file}': {e}")
        return

    wavelengths = data[:, 0]
    intensities = data[:, 1]

    # Plot the data
    plt.figure()
    plt.plot(wavelengths, intensities, label='Average Intensity')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity (Arbitrary Units)')
    plt.title('Averaged Intensity Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
