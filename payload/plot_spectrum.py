"""
AERO4701 Space Engineering 3
Lunar Atmospheric Investigations with CubeSats (LUNATICS)

Plotting the spectrum data from a CSV file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
from pathlib import Path
import plotting


def plot_spectrum(filename, rows_to_plot=None):
    """
    Plots the spectrum data from a CSV file.
    
    Args:
        filename (str): Name of CSV file containing spectrum data.
        rows_to_plot (list): List of row indices to plot.
    """

    # Read CSV file
    data = pd.read_csv(filename, sep=',')

    # Extract wavelengths from header row
    wavelengths = [int(col.strip('nm')) for col in data.columns]
    print(wavelengths)
    # For Gaussian
    FWHM = 20
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))

    # Filter dataframe for specified rows
    if rows_to_plot is not None:
        rows_to_plot = [row - 1 for row in rows_to_plot]  # Adjust for zero-based indexing
        data = data.iloc[rows_to_plot]

    for i, row in data.iterrows():
        spectrum = row.values

        # Interpolate spectrum
        # interp = sc.interpolate.interp1d(wavelengths, spectrum, kind='slinear', fill_value='extrapolate')
        interp = sc.interpolate.PchipInterpolator(wavelengths, spectrum)
        
        # Create new wavelength range for interpolation
        wavelength_range = np.linspace(min(wavelengths), max(wavelengths), 5000)
        spectrum_interp = interp(wavelength_range)
        normalised_spectrum = spectrum_interp / max(spectrum_interp)

        # Convert wavelengths to RGB
        # colours = [normalise_rgb(wavelength_to_rgb(w)) for w in wavelength_range]
        colours = [wavelength_to_rgb(w) for w in wavelength_range]
        # Normalise RGB values
        colours = [normalise_rgb(colour) for colour in colours]

        # Initialise figure
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6, 4),
                                       sharex=True, 
                                       gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
        ax0.grid(visible=True, which='both', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

        # Plot Gaussian
        for j, intensity in enumerate(spectrum):
            mu = wavelengths[j]
            A = intensity
            gauss_spectrum = gaussian(wavelength_range, A, mu, sigma)
            ax0.plot(wavelength_range, gauss_spectrum, color='black', linestyle='--', alpha=0.5, lw=1)

        # Plot the intensity spectrum
        ax0.scatter(wavelength_range, spectrum_interp, c=colours, s=10)
        ax0.set_ylabel('Reading (uW/cm²/count)')
        ax0.set_xlim([400, max(wavelength_range)])
        ax0.set_ylim([0, 1.1 * max(spectrum_interp)])

        # Plot absorption spectrum
        for j, intensity in enumerate(normalised_spectrum):
            colour = colours[j]
            ax1.axvline(wavelength_range[j], color=colour, alpha=intensity, lw=1)
        ax1.set_facecolor('black')
        ax1.get_yaxis().set_visible(False)
        ax1.set_xlim([400, max(wavelength_range)])
        ax1.grid(visible=False)
        ax1.set_xlabel('Wavelength (nm)')
    
    plt.subplots_adjust(left=0.1, right=0.97, top=0.97, bottom=0.18, hspace=0.025)
    plt.tight_layout()
    plt.show()


def wavelength_to_rgb(wavelength):
    """
    Converts wavelength (nm) into RGB colour.

    Args:
        wavelength (float): Wavelength in nanometres.
    """
    Gamma = 0.8
    IntensityMax = 255
    factor = 0
    # Initialise RGB
    red, green, blue = 0, 0, 0

    # Convert wavelength to RGB components
    if 380 <= wavelength < 440:
        red = (440 - wavelength) / (440 - 380)
        green = 0
        blue = 1
    elif 440 <= wavelength < 490:
        red = 0
        green = (wavelength - 440) / (490 - 440)
        blue = 1
    elif 490 <= wavelength < 510:
        red = 0
        green = 1
        blue = (510 - wavelength) / (510 - 490)
    elif 510 <= wavelength < 580:
        red = (wavelength - 510) / (580 - 510)
        green = 1
        blue = 0
    elif 580 <= wavelength < 645:
        red = 1
        green = (645 - wavelength) / (645 - 580)
        blue = 0
    elif 645 <= wavelength <= 780:
        red = 1
        green = 0
        blue = 0
    else:
        red = 0
        green = 0
        blue = 0

    # Adjust intensity
    if 380 <= wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 420 <= wavelength < 701:
        factor = 1.0
    elif 701 <= wavelength <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        factor = 0.0

    # Apply the intensity factor
    if red > 0:
        red = round(IntensityMax * ((red * factor) ** Gamma))
    if green > 0:
        green = round(IntensityMax * ((green * factor) ** Gamma))
    if blue > 0:
        blue = round(IntensityMax * ((blue * factor) ** Gamma))

    return red, green, blue


def normalise_rgb(rgb):
    """
    Normalises RGB values to be between 0 and 1.

    Args:
        wavelength (tuple): Tuple of RGB values.
    """
    
    r, g, b = rgb
    r /= 255
    g /= 255
    b /= 255

    return r, g, b


def gaussian(x, A, mu, sigma):
    """Gaussian function.
    Args:
        x (np.array): Array of x values [wavelengths].
        A (float): Amplitude of the Gaussian [intensity].
        mu (float): Mean of Gaussian [central wavelength].
        sigma (float): Standard deviation of Gaussian [from FWHM].
    """

    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

if __name__ == "__main__":
    filepath = Path(__file__).parent
    filename = filepath / 'spec_values.csv'

    wavelengths = [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940]

    plotting.startup_plotting(font_size=14, line_width=1.5, output_dpi=600, tex_backend=True)
    plot_spectrum(filename, [2])


