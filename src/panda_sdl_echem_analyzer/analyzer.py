"""
Main interface for electrochemical analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .colormaps import ColorMapGenerator
from .plotters import CAPlotter, CVPlotter, OCPPlotter
from .utils import load_lookup_table


class ElectrochemicalAnalyzer:
    """
    Main class for performing electrochemical data analysis.

    This class provides a high-level interface to analyze and visualize
    electrochemical data from different experiment types.
    """

    def __init__(
        self,
        data_directory: Union[str, Path],
        lookup_file: Optional[Union[str, Path]] = None,
        geometry: str = "circular",
        diameter: float = 6.5,
    ):
        """
        Initialize the analyzer with data directory and optional lookup table.

        Args:
            data_directory: Path to directory containing data files
            lookup_file: Optional path to CSV file with metadata
            geometry: Electrode geometry ('circular' or 'square')
            diameter: Electrode diameter or side length in mm
        """
        self.data_directory = Path(data_directory)
        self.geometry = geometry
        self.diameter = diameter

        # Load lookup table if provided
        self.lookup_df = None
        if lookup_file:
            self.lookup_df = load_lookup_table(lookup_file)

        # Initialize plotters
        self._init_plotters()

    def _init_plotters(self):
        """Initialize plotters for different experiment types."""
        self.cv_plotter = CVPlotter(
            self.data_directory, self.lookup_df, self.geometry, self.diameter
        )

        self.ca_plotter = CAPlotter(
            self.data_directory, self.lookup_df, self.geometry, self.diameter
        )

        self.ocp_plotter = OCPPlotter(self.data_directory, self.lookup_df)

    def scan_files(self) -> Dict[str, List[Path]]:
        """
        Scan data directory for experiment files.

        Returns:
            Dictionary of experiment types and their files
        """
        file_types = {
            "CV": list(self.data_directory.glob("*CV*.txt")),
            "CA": list(self.data_directory.glob("*CA*.txt")),
            "OCP": list(self.data_directory.glob("*OCP*.txt")),
        }

        # Clean our the CV OCP and CA OCP
        file_types["CV"] = [file for file in file_types["CV"] if "OCP" not in file.name]
        file_types["CA"] = [file for file in file_types["CA"] if "OCP" not in file.name]

        # Print summary of found files
        for exp_type, files in file_types.items():
            print(f"Found {len(files)} {exp_type} files")

        return file_types

    def plot_cv_second_cycles(
        self,
        sort_by: str = "voltage",
        ylim: Optional[Tuple[float, float]] = None,
        show_plot: bool = True,
        save_plot: bool = True,
        log_voltage: bool = False,
        log_current: bool = False,
        style: str = "scatter",
    ) -> None:
        """
        Plot second cycles from CV files.

        Args:
            sort_by: How to sort the files ('voltage' or 'time')
            ylim: Optional y-axis limits as (min, max)
            show_plot: Whether to display the plot
        """
        if sort_by.lower() == "voltage":
            self.cv_plotter.plot_second_cycles_by_voltage(
                ylim=ylim,
                show_plot=show_plot,
                save_plot=save_plot,
                log_voltage=log_voltage,
                log_current=log_current,
                style=style,
            )
        elif sort_by.lower() == "time":
            self.cv_plotter.plot_second_cycles_by_time(
                ylim=ylim,
                show_plot=show_plot,
                save_plot=save_plot,
                log_voltage=log_voltage,
                log_current=log_current,
                style=style,
            )
        else:
            raise ValueError(
                f"Unknown sort option: {sort_by}. Use 'voltage' or 'time'."
            )

    def plot_cv_all_cycles(
        self,
        max_cycle: int = 4,
        show_plot: bool = True,
        save_plot: bool = True,
        log_voltage: bool = False,
        log_current: bool = False,
        style: str = "scatter",
    ) -> None:
        """
        Plot all cycles for each CV file.

        Args:
            max_cycle: Maximum cycle number to plot
            show_plot: Whether to display the plots
        """
        self.cv_plotter.plot_all_cycles_individual(
            max_cycle=max_cycle,
            show_plot=show_plot,
            save_plot=save_plot,
            log_voltage=log_voltage,
            log_current=log_current,
            style=style,
        )

    def plot_cv_second_cycle_individual(
        self,
        show_plot: bool = True,
        save_plot: bool = True,
        log_voltage: bool = False,
        log_current: bool = False,
        style: str = "scatter",
    ) -> None:
        """
        Plot second cycle for each CV file individually.

        Args:
            show_plot: Whether to display the plots
        """
        self.cv_plotter.plot_second_cycle_individual(
            show_plot=show_plot,
            save_plot=save_plot,
            log_voltage=log_voltage,
            log_current=log_current,
            style=style,
        )

    def plot_ca_curves(
        self,
        sort_by: Optional[str] = None,
        show_plot: bool = True,
        save_plot: bool = True,
        excluded_ids: Optional[List[str | int]] = None,
        log_time: bool = False,
        log_current: bool = False,
        style: str = "scatter",
    ) -> None:
        """
        Plot chronoamperometry curves.

        Args:
            sort_by: How to sort the files ('voltage', 'time', or None)
            show_plot: Whether to display the plot
        """
        if sort_by is None:
            self.ca_plotter.plot_all_curves(
                show_plot=show_plot,
                save_plot=save_plot,
                excluded_ids=excluded_ids,
                log_time=log_time,
                log_current=log_current,
                style=style,
            )
        elif sort_by.lower() == "voltage":
            self.ca_plotter.plot_by_voltage(
                show_plot=show_plot,
                save_plot=save_plot,
                excluded_ids=excluded_ids,
                log_time=log_time,
                log_current=log_current,
                style=style,
            )
        elif sort_by.lower() == "time":
            self.ca_plotter.plot_by_time(
                show_plot=show_plot,
                save_plot=save_plot,
                excluded_ids=excluded_ids,
                log_time=log_time,
                log_current=log_current,
                style=style,
            )
        else:
            raise ValueError(
                f"Unknown sort option: {sort_by}. Use 'voltage', 'time', or None."
            )

    def plot_ca_individual(
        self,
        show_plot: bool = True,
        save_plot: bool = True,
        excluded_ids: Optional[List[str | int]] = None,
        log_time: bool = False,
        log_current: bool = False,
        style: str = "scatter",
    ) -> None:
        """
        Plot each CA file individually.

        Args:
            show_plot: Whether to display the plots
        """
        self.ca_plotter.plot_individual(
            show_plot=show_plot,
            save_plot=save_plot,
            excluded_ids=excluded_ids,
            log_time=log_time,
            log_current=log_current,
            style=style,
        )

    def plot_ocp_curves(
        self,
        show_plot: bool = True,
        save_plot: bool = True,
        log_voltage: bool = False,
        log_current: bool = False,
        style: str = "scatter",
    ) -> None:
        """
        Plot all OCP curves in one figure.

        Args:
            show_plot: Whether to display the plot
        """
        self.ocp_plotter.plot_all_curves(
            show_plot=show_plot,
            save_plot=save_plot,
            log_voltage=log_voltage,
            log_current=log_current,
            style=style,
        )

    def custom_colors(self, scheme: str = "default") -> Dict:
        """
        Get a color scheme for plotting.

        Args:
            scheme: The color scheme name ('default', 'high_contrast', 'blue_shades')

        Returns:
            Dictionary with color values
        """
        generator = ColorMapGenerator()
        default_colors = generator.get_default_colors()

        schemes = {
            "default": default_colors,
            "high_contrast": {
                "five_color": {
                    "c1": "#3B0F70",  # Deep purple
                    "c2": "#8C2981",  # Medium purple
                    "c3": "#DE4968",  # Rose
                    "c4": "#FE9F6D",  # Peach
                    "c5": "#FCFDBF",  # Pale yellow
                },
                "three_color": {
                    "start": "#3B0F70",  # Deep purple
                    "middle": "#DE4968",  # Rose
                    "end": "#FCFDBF",  # Pale yellow
                },
                "two_color": {
                    "start": "#3B0F70",  # Deep purple
                    "end": "#FCFDBF",  # Pale yellow
                },
            },
            "blue_shades": {
                "five_color": {
                    "c1": "#03071E",  # Navy
                    "c2": "#023E8A",  # Royal blue
                    "c3": "#0096C7",  # Sky blue
                    "c4": "#48CAE4",  # Light blue
                    "c5": "#ADE8F4",  # Pale blue
                },
                "three_color": {
                    "start": "#03071E",  # Navy
                    "middle": "#0096C7",  # Sky blue
                    "end": "#ADE8F4",  # Pale blue
                },
                "two_color": {
                    "start": "#03071E",  # Navy
                    "end": "#ADE8F4",  # Pale blue
                },
            },
        }

        if scheme not in schemes:
            print(f"Unknown scheme: {scheme}. Using default.")
            return default_colors

        return schemes[scheme]
