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

    data = pd.read_csv(filename, sep=',')

    # Extract wavelengths from header row
    wavelengths = [int(col.strip('nm')) for col in data.columns]

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
        wavelength_range = np.linspace(min(wavelengths), max(wavelengths), 1000)
        spectrum_interp = interp(wavelength_range)

        # Convert wavelengths to RGB
        # colours = [normalise_rgb(wavelength_to_rgb(w)) for w in wavelength_range]
        colours = [wavelength_to_rgb(w) for w in wavelength_range]
        # Normalise RGB values
        colours = [normalise_rgb(colour) for colour in colours]

        # Plot the spectrum
        fig, ax = plt.subplots(figsize=(6,3))
        ax.scatter(wavelength_range, spectrum_interp, c=colours)

        # Plot details
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reading (nW/cm²/count)')
        plt.ylim([0, 1.1 * max(spectrum_interp)])
        plt.grid(visible=True, which='both', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.subplots_adjust(left=0.1, right=0.97, top=0.97, bottom=0.18)


    
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


if __name__ == "__main__":
    filepath = Path(__file__).parent
    filename = filepath / 'spec_values.csv'

    plotting.startup_plotting(font_size=14, line_width=1.5, output_dpi=600, tex_backend=True)
    plot_spectrum(filename, [2])