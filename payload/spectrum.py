# Import modules
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from pathlib import Path
import plotting


# Gaussian 
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Wavelengths to RGB
def wavelength_to_rgb(wavelength):
    Gamma = 0.8
    IntensityMax = 255
    factor = 0
    red, green, blue = 0, 0, 0

    if 380 <= wavelength < 440:
        red, green, blue = (440 - wavelength) / 60, 0, 1
    elif 440 <= wavelength < 490:
        red, green, blue = 0, (wavelength - 440) / 50, 1
    elif 490 <= wavelength < 510:
        red, green, blue = 0, 1, (510 - wavelength) / 20
    elif 510 <= wavelength < 580:
        red, green, blue = (wavelength - 510) / 70, 1, 0
    elif 580 <= wavelength < 645:
        red, green, blue = 1, (645 - wavelength) / 65, 0
    elif 645 <= wavelength <= 780:
        red, green, blue = 1, 0, 0

    if 380 <= wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / 40
    elif 420 <= wavelength < 701:
        factor = 1.0
    elif 701 <= wavelength <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength) / 80

    r = round(IntensityMax * ((red * factor) ** Gamma)) if red > 0 else 0
    g = round(IntensityMax * ((green * factor) ** Gamma)) if green > 0 else 0
    b = round(IntensityMax * ((blue * factor) ** Gamma)) if blue > 0 else 0
    
    return r/255, g/255, b/255


def plot_spectrum(filename, wavelengths):
    # Read CSV
    df = pd.read_csv(filename, header=None, skiprows=1)
    time_stamps = pd.to_datetime(df.iloc[:, 0])
    data = df.iloc[:, 1:].values

    # Global value of max. spectrum during this group of readings
    max_spectrum_global = np.max(data)

    # Gaussian and colour settings
    FWHM = 20
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    wavelength_range = np.linspace(min(wavelengths), max(wavelengths), 5000)
    colours = [wavelength_to_rgb(w) for w in wavelength_range]

    # Initialise figure
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
    scatter = ax0.scatter([], [], s=10)
    gauss_lines = [ax0.plot([], [], 'k--', lw=1, alpha=0.4)[0] for _ in wavelengths]
    time_text = ax0.text(0.7, 0.9, '', transform=ax0.transAxes)

    # Initialisation function.
    def init():
        ax0.set_xlim(400, 950)
        ax0.set_ylim(0, np.max(data) * 1.1)
        ax0.set_ylabel('Sensor Reading')
        ax0.grid(True, linestyle='--', alpha=0.5)
        ax1.set_xlim(400, 950)
        ax1.set_facecolor('black')
        ax1.set_yticks([])
        ax1.set_xlabel('Wavelength (nm)')
        return [scatter, time_text] + gauss_lines

    def update(frame):
        readings = data[frame]
        interp = sc.interpolate.PchipInterpolator(wavelengths, readings)
        spectrum_interp = interp(wavelength_range)
        normalised_spectrum = spectrum_interp / max_spectrum_global
        time_text.set_text(f"{time_stamps[frame].strftime('%Y-%m-%d %H:%M:%S')}")

        y_max = np.max(spectrum_interp)
        ax0.set_ylim(0, 1.1 * y_max)

        # Update scatter with colour-coded wavelengths
        scatter.set_offsets(np.column_stack([wavelength_range, spectrum_interp]))
        scatter.set_array(np.linspace(0, 1, len(wavelength_range)))  # Dummy array for cmap
        scatter.set_color(colours)

        # Plot Gaussian curves
        for j, wl in enumerate(wavelengths):
            gauss_lines[j].set_data(wavelength_range, gaussian(wavelength_range, readings[j], wl, sigma))

        # Plot colour-coded absorption lines
        ax1.clear()
        ax1.set_xlim(400, 950)
        ax1.set_facecolor('black')
        ax1.set_xticks(np.arange(400, 951, 100))
        ax1.set_yticks([])
        ax1.set_ylabel("")
        for j, intensity in enumerate(normalised_spectrum):
            ax1.axvline(wavelength_range[j], color=colours[j], alpha=intensity, lw=1)

        return [scatter, time_text] + gauss_lines

    ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, blit=False, repeat=True, interval=5)

    ani.save("spectrum.gif", fps=4)

    plt.show()

if __name__ == "__main__":
    filepath = Path(__file__).parent
    filename = filepath / 'output1.csv'

    wavelengths = [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940]

    plotting.startup_plotting(font_size=14, line_width=1.5, output_dpi=600, tex_backend=True)

    plot_spectrum(filename, wavelengths)