"""
Example usage of the echem_analyzer library.
"""

from pathlib import Path

from src.panda_sdl_echem_analyzer import ElectrochemicalAnalyzer

# Set up paths
data_dir = Path("/data/CV_samples")
lookup_file = Path("/data/experiment_lookup.csv")

# Create analyzer
analyzer = ElectrochemicalAnalyzer(
    data_directory=data_dir, lookup_file=lookup_file, geometry="circular", diameter=6.5
)

# Scan for files in the directory
files = analyzer.scan_files()
print(f"Found {sum(len(files_list) for files_list in files.values())} total files")

# Plot CV second cycles sorted by voltage
analyzer.plot_cv_second_cycles(sort_by="voltage", ylim=(-0.6, 1.0), show_plot=True)

# Use a different color scheme
high_contrast_colors = analyzer.custom_colors(scheme="high_contrast")
analyzer.cv_plotter.plot_second_cycles_by_time(
    colors=[
        high_contrast_colors["five_color"]["c1"],
        high_contrast_colors["five_color"]["c2"],
        high_contrast_colors["five_color"]["c3"],
        high_contrast_colors["five_color"]["c4"],
        high_contrast_colors["five_color"]["c5"],
    ],
    ylim=(-0.5, 0.8),
    show_plot=True,
)

# Plot CA curves
analyzer.plot_ca_curves(sort_by="voltage", show_plot=True)

# Plot OCP data
analyzer.plot_ocp_curves(show_plot=True)

# Generate individual plots for all CV files
# This creates one plot per file in the data directory
analyzer.plot_cv_all_cycles(max_cycle=4, show_plot=False)

print("All plots generated successfully!")
