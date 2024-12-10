#!/usr/bin/env python3

import argparse
import numpy as np
import scipy.optimize as optimize
import sys

def lorentzian(x, x0, gamma, A, y0):
    """Lorentzian function.

    Parameters:
    - x: Independent variable (e.g., wavelength)
    - x0: Center position of the peak
    - gamma: Half-width at half-maximum (HWHM)
    - A: Amplitude (peak height above baseline)
    - y0: Baseline offset

    Returns:
    - Lorentzian function evaluated at x
    """
    return y0 + (A * gamma**2) / ((x - x0)**2 + gamma**2)

def round_to_thousandth(value):
    """Round a float to the nearest thousandth (3 decimal places)."""
    return round(value + 1e-10, 3)  # Adding a small epsilon to handle floating-point issues

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Fit a Lorentzian curve to spectral data within a specified wavelength window.')
    parser.add_argument('-i', required=True, help='Input averaged data text file.')
    parser.add_argument('-o', required=True, help='Output text file to save the Lorentzian fit parameters.')
    parser.add_argument('-l', required=True, type=float, help='Lower bound wavelength for the fitting window.')
    parser.add_argument('-u', required=True, type=float, help='Upper bound wavelength for the fitting window.')
    parser.add_argument('-p', action='store_true', help='Plot the fitted curve over the input data.')
    args = parser.parse_args()

    input_file = args.i
    output_file = args.o
    lower_bound_input = args.l
    upper_bound_input = args.u
    plot_flag = args.p  # True if -p is specified, else False

    # Load the averaged data from the input file
    try:
        data = np.loadtxt(input_file)
        if data.ndim != 2 or data.shape[1] != 2:
            print(f"Error: The data file '{input_file}' does not have two columns.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading data from file '{input_file}': {e}")
        sys.exit(1)

    wavelengths = data[:, 0]
    intensities = data[:, 1]

    # Ensure that lower_bound is less than upper_bound
    if lower_bound_input >= upper_bound_input:
        print("Error: The lower bound wavelength must be less than the upper bound wavelength.")
        sys.exit(1)

    # Round the input bounds to the nearest thousandth
    lower_bound_rounded = round_to_thousandth(lower_bound_input)
    upper_bound_rounded = round_to_thousandth(upper_bound_input)

    # Debug: Print rounded bounds
    print(f"Original Lower Bound: {lower_bound_input} -> Rounded Lower Bound: {lower_bound_rounded}")
    print(f"Original Upper Bound: {upper_bound_input} -> Rounded Upper Bound: {upper_bound_rounded}")

    # Find the smallest wavelength in the data >= rounded lower bound
    lower_index = np.searchsorted(wavelengths, lower_bound_rounded, side='left')
    if lower_index == len(wavelengths):
        print(f"Error: Rounded lower bound wavelength {lower_bound_rounded} is beyond the data range.")
        sys.exit(1)
    actual_lower_bound = wavelengths[lower_index]
    if actual_lower_bound < lower_bound_rounded:
        # Move to the next index if exact match not found
        lower_index += 1
        if lower_index == len(wavelengths):
            print(f"Error: No wavelength found >= rounded lower bound {lower_bound_rounded}.")
            sys.exit(1)
        actual_lower_bound = wavelengths[lower_index]

    # Find the smallest wavelength in the data >= rounded upper bound
    upper_index = np.searchsorted(wavelengths, upper_bound_rounded, side='left')
    if upper_index == len(wavelengths):
        upper_index = len(wavelengths)  # Use the last index
    actual_upper_bound = wavelengths[upper_index] if upper_index < len(wavelengths) else wavelengths[-1]

    # If the exact rounded upper bound isn't present, use the next higher wavelength
    if upper_index < len(wavelengths) and wavelengths[upper_index] < upper_bound_rounded:
        upper_index += 1
        if upper_index > len(wavelengths):
            upper_index = len(wavelengths)
        actual_upper_bound = wavelengths[upper_index-1] if upper_index-1 < len(wavelengths) else wavelengths[-1]

    # Debug: Print actual bounds used
    print(f"Using Lower Bound: {actual_lower_bound}")
    print(f"Using Upper Bound: {actual_upper_bound}")

    # Select the data within the fitting window
    x_data = wavelengths[lower_index:upper_index]
    y_data = intensities[lower_index:upper_index]

    if len(x_data) == 0:
        print("Error: No data points found within the specified wavelength window.")
        sys.exit(1)

    # Initial guesses for the Lorentzian parameters
    x0_guess = x_data[np.argmax(y_data)]  # Wavelength at maximum intensity within window
    gamma_guess = (x_data[-1] - x_data[0]) / 10 or 1e-6  # Avoid zero gamma
    A_guess = np.max(y_data) - np.min(y_data)  # Peak amplitude above baseline
    y0_guess = np.min(y_data)  # Baseline offset

    p0 = [x0_guess, gamma_guess, A_guess, y0_guess]

    # Fit the Lorentzian curve
    try:
        popt, pcov = optimize.curve_fit(lorentzian, x_data, y_data, p0=p0)
        x0_fit, gamma_fit, A_fit, y0_fit = popt
        fwhm = 2 * gamma_fit  # FWHM for Lorentzian is 2*gamma

        # Extract standard deviations of the parameters from the covariance matrix
        perr = np.sqrt(np.diag(pcov))

        # Write the fitted parameters to the output file
        with open(output_file, 'w') as f_out:
            f_out.write('Parameter\tValue\tStd Dev\n')
            f_out.write(f'Center (x0)\t{x0_fit:.6f}\t{perr[0]:.6f}\n')
            f_out.write(f'FWHM\t{fwhm:.6f}\t{(2 * perr[1]):.6f}\n')  # FWHM uncertainty is 2 * gamma uncertainty
            f_out.write(f'Amplitude (A)\t{A_fit:.6f}\t{perr[2]:.6f}\n')
            f_out.write(f'Baseline (y0)\t{y0_fit:.6f}\t{perr[3]:.6f}\n')

        print(f"\nLorentzian fit completed. Parameters saved to '{output_file}'.")
        print(f"Center (x0): {x0_fit:.6f} ± {perr[0]:.6f}")
        print(f"FWHM: {fwhm:.6f} ± {2 * perr[1]:.6f}")
        print(f"Amplitude (A): {A_fit:.6f} ± {perr[2]:.6f}")
        print(f"Baseline (y0): {y0_fit:.6f} ± {perr[3]:.6f}")

        # Plotting if the -p flag is specified
        if plot_flag:
            try:
                import matplotlib.pyplot as plt

                # Plot the data and the fit
                plt.figure(figsize=(8, 6))
                plt.plot(wavelengths, intensities, 'b-', label='Original Data')
                plt.plot(x_data, y_data, 'g.', label='Data Used for Fitting')
                x_fit = np.linspace(x_data[0], x_data[-1], 500)
                y_fit = lorentzian(x_fit, *popt)
                plt.plot(x_fit, y_fit, 'r-', label='Lorentzian Fit')
                plt.xlabel('Wavelength')
                plt.ylabel('Intensity')
                plt.title('Lorentzian Fit Over Input Data')
                plt.legend()
                plt.grid(True)
                plt.show()
            except ImportError:
                print("Matplotlib is not installed. Unable to plot the fitted curve.")
    except Exception as e:
        print(f"Error in fitting: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
