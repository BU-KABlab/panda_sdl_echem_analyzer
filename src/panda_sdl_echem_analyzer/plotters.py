"""
Classes for plotting different types of electrochemical data.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .colormaps import ColorMapGenerator
from .utils import (
    load_experiment_data,
    load_experiment_files,
    modify_current_density,
    parse_file_metadata,
)


class BasePlotter:
    """Base class for all plotters with common functionality."""

    def __init__(
        self,
        folder_path: Union[str, Path],
        experiment_type: str,
        lookup_df: Optional[pd.DataFrame] = None,
        geometry: str = "circular",
        diameter: float = 6.5,
    ):
        """
        Initialize the plotter.

        Args:
            folder_path: Path to folder containing data files
            experiment_type: Type of experiment (CV, CA, OCP)
            lookup_df: Optional DataFrame with metadata for the samples
            geometry: Electrode geometry ('circular' or 'square')
            diameter: Electrode diameter or side length in mm
        """
        self.folder_path = Path(folder_path)
        self.experiment_type = experiment_type
        self.lookup_df = lookup_df
        self.geometry = geometry
        self.diameter = diameter
        self.color_generator = ColorMapGenerator()

        # Configure default plot settings
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["figure.facecolor"] = "white"

    def _apply_current_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert current to current density."""
        df_copy = df.copy()
        df_copy["Im"] = df_copy["Im"].apply(
            lambda x: modify_current_density(x, self.geometry, self.diameter)
        )
        return df_copy

    def _setup_figure(self, figsize: Tuple[float, float] = (8, 6)) -> None:
        """Create a new figure for plotting."""
        plt.figure(figsize=figsize)

    def _save_figure(self, filename: str, formats: List[str] = ["png", "svg"]) -> None:
        """Save the current figure in multiple formats."""
        for fmt in formats:
            output_path = self.folder_path / f"{filename}.{fmt}"
            plt.savefig(output_path)
            print(f"Saved plot to {output_path}")

    def _close_figure(self) -> None:
        """Close the current figure."""
        plt.close()


class CVPlotter(BasePlotter):
    """Plotter for cyclic voltammetry data."""

    def __init__(
        self,
        folder_path: Union[str, Path],
        lookup_df: Optional[pd.DataFrame] = None,
        geometry: str = "circular",
        diameter: float = 6.5,
    ):
        """Initialize the CV plotter."""
        super().__init__(folder_path, "CV", lookup_df, geometry, diameter)

    def plot_second_cycles_by_voltage(
        self,
        colors: List[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        save_formats: List[str] = ["png", "svg"],
        show_plot: bool = True,
        save_plot: bool = True,
    ) -> None:
        """
        Plot second cycles from all files, sorted by experiment id.

        Args:
            colors: List of 5 colors for the gradient (if None, uses defaults)
            ylim: Optional y-axis limits as (min, max)
            save_formats: List of formats to save the plot in
            show_plot: Whether to display the plot
        """
        self._setup_figure()

        # Use default colors if none provided
        if colors is None:
            default_colors = self.color_generator.get_default_colors()["five_color"]
            c1, c2, c3, c4, c5 = (
                default_colors["c1"],
                default_colors["c2"],
                default_colors["c3"],
                default_colors["c4"],
                default_colors["c5"],
            )
        else:
            c1, c2, c3, c4, c5 = colors

        files = load_experiment_files(self.folder_path, f"*{self.experiment_type}*.txt")
        for file in files:
            if "OCP" in file.stem:
                files.remove(file)
        # Get files with voltage information
        file_dep_v = []
        for file_path in files:
            metadata = parse_file_metadata(file_path, self.lookup_df)
            if "dep_V" in metadata:
                file_dep_v.append((file_path, metadata["dep_V"], metadata))

        # Sort files by experiment id, lowest to highest
        sorted_files = sorted(file_dep_v, key=lambda x: int(x[0].stem.split("_")[2]))

        # Create colormap
        custom_cmap = self.color_generator.create_five_color_gradient(
            len(sorted_files), c1, c2, c3, c4, c5, "custom_colormap"
        )

        legend_handles = []
        legend_labels = []

        # Plot each file
        for index, (file_path, dep_V, metadata) in enumerate(sorted_files):
            reference = metadata["reference"]

            # Get deposition time if available
            dep_T = metadata.get("dep_T", "?")

            # Label for the legend
            label = f"{reference}_{dep_V}V_{dep_T}s"

            # Load and process data
            df = load_experiment_data(file_path, self.experiment_type)

            # Get second cycle (index 1)
            df_second_cycle = df[df["Cycle"] == 1]

            if len(df_second_cycle) == 0:
                continue  # Skip files without a second cycle

            # Apply current density conversion
            df_second_cycle = self._apply_current_density(df_second_cycle)

            # Plot with color from the colormap
            color = custom_cmap(index / len(sorted_files))
            plt.scatter(
                df_second_cycle["Vsig"],
                df_second_cycle["Im"],
                color=color,
                s=5,
            )

            # Add to legend
            legend_handles.append(Line2D([0], [0], color=color, lw=2))
            legend_labels.append(label)

        # Set plot labels and legend
        plt.xlabel("V vs V_Ag/AgCl (V)")
        plt.ylabel("Current Density (mA/cm²)")
        plt.legend(
            legend_handles, legend_labels, bbox_to_anchor=(1.05, 0.5), loc="center left"
        )
        plt.tight_layout()

        # Set y-axis limits if provided
        if ylim:
            plt.ylim(ylim)

        # Save the figure
        if save_plot:
            self._save_figure(
                f"all_second_cycles_{self.experiment_type}_sortV", save_formats
            )

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            self._close_figure()

    def plot_second_cycles_by_time(
        self,
        colors: List[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        save_formats: List[str] = ["png", "svg"],
        show_plot: bool = True,
        save_plot: bool = True,
    ) -> None:
        """
        Plot second cycles from all files, sorted by experiment id.

        Args:
            colors: List of 5 colors for the gradient (if None, uses defaults)
            ylim: Optional y-axis limits as (min, max)
            save_formats: List of formats to save the plot in
            show_plot: Whether to display the plot
        """
        self._setup_figure()

        # Use default colors if none provided
        if colors is None:
            default_colors = self.color_generator.get_default_colors()["five_color"]
            c1, c2, c3, c4, c5 = (
                default_colors["c1"],
                default_colors["c2"],
                default_colors["c3"],
                default_colors["c4"],
                default_colors["c5"],
            )
        else:
            c1, c2, c3, c4, c5 = colors

        files = load_experiment_files(self.folder_path, f"*{self.experiment_type}*.txt")

        # Get files with time information
        file_dep_t = []
        for file_path in files:
            metadata = parse_file_metadata(file_path, self.lookup_df)
            if "dep_T" in metadata:
                file_dep_t.append((file_path, metadata["dep_T"], metadata))

        # Sort files by experiment id, lowest to highest
        sorted_files = sorted(file_dep_t, key=lambda x: int(x[0].stem.split("_")[2]))

        # Create colormap
        custom_cmap = self.color_generator.create_five_color_gradient(
            len(sorted_files), c1, c2, c3, c4, c5, "custom_colormap"
        )

        legend_handles = []
        legend_labels = []

        # Plot each file
        for index, (file_path, dep_T, metadata) in enumerate(sorted_files):
            reference = metadata["reference"]

            # Get deposition voltage if available
            dep_V = metadata.get("dep_V", "?")

            # Label for the legend
            label = f"{reference}_{dep_V}V_{dep_T}s"

            # Load and process data
            df = load_experiment_data(file_path, self.experiment_type)

            # Get second cycle (index 1)
            df_second_cycle = df[df["Cycle"] == 1]

            if len(df_second_cycle) == 0:
                continue  # Skip files without a second cycle

            # Apply current density conversion
            df_second_cycle = self._apply_current_density(df_second_cycle)

            # Plot with color from the colormap
            color = custom_cmap(index / len(sorted_files))
            plt.scatter(
                df_second_cycle["Vsig"],
                df_second_cycle["Im"],
                color=color,
                s=5,
            )

            # Add to legend
            legend_handles.append(Line2D([0], [0], color=color, lw=2))
            legend_labels.append(label)

        # Set plot labels and legend
        plt.xlabel("V vs V_Ag/AgCl (V)")
        plt.ylabel("Current Density (mA/cm²)")
        plt.legend(
            legend_handles, legend_labels, bbox_to_anchor=(1.05, 0.5), loc="right"
        )
        plt.tight_layout()

        # Set y-axis limits if provided
        if ylim:
            plt.ylim(ylim)

        # Save the figure
        if save_plot:
            self._save_figure(
                f"all_second_cycles_{self.experiment_type}_sortT", save_formats
            )

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            self._close_figure()

    def plot_all_cycles_individual(
        self, max_cycle: int = 4, show_plot: bool = True, save_plot: bool = True
    ) -> None:
        """
        Create individual plots for each file showing all cycles.

        Args:
            max_cycle: Maximum cycle number to plot
            show_plot: Whether to display the plots
        """
        files = load_experiment_files(self.folder_path, f"*{self.experiment_type}*.txt")
        files = [file for file in files if "OCP" not in file.stem]
        files.sort(key=lambda x: int(x.stem.split("_")[2]))

        for file_path in files:
            id = file_path.stem.split("_")[0]
            # Load data
            df = load_experiment_data(file_path, self.experiment_type)

            # Apply current density conversion
            df = self._apply_current_density(df)

            # Create new figure
            self._setup_figure()

            # Get colors for the cycles
            # colors = cm.winter_r(np.linspace(0, 1, max_cycle))

            # Plot each cycle
            for i in range(max_cycle):
                df_cycle = df[df["Cycle"] == i]
                if len(df_cycle) > 0:
                    plt.scatter(
                        df_cycle["Vf"],
                        df_cycle["Im"],
                        linestyle="--",
                        label=f"{id} Cycle {i}",
                        s=5,
                    )

            # Set labels and title
            plt.xlabel("V vs Ag (V)")
            plt.ylabel("Current Density (mA/cm²)")
            plt.legend()
            plt.title(file_path.stem)
            plt.tight_layout()

            # Save the figure
            if save_plot:
                self._save_figure(f"{file_path.stem}_allcycles", ["png"])

            # Show or close
            if show_plot:
                plt.show()
            else:
                self._close_figure()

    def plot_second_cycle_individual(
        self, show_plot: bool = True, save_plot: bool = True, style: str = "scatter"
    ) -> None:
        """
        Create individual plots for the second cycle of each file.

        Args:
            show_plot: Whether to display the plots
        """
        files = load_experiment_files(self.folder_path, f"*{self.experiment_type}*.txt")
        files = [file for file in files if "OCP" not in file.stem]
        files.sort(key=lambda x: int(x.stem.split("_")[2]))
        for file_path in files:
            try:
                # Create new figure
                plt.figure(figsize=(6, 4))

                # Load data
                df = load_experiment_data(file_path, self.experiment_type)

                # Get second cycle (index 1)
                df_second_cycle = df[df["Cycle"] == 1]

                if len(df_second_cycle) == 0:
                    print(f"No second cycle data for {file_path.stem}")
                    continue

                # Apply current density conversion
                df_second_cycle = self._apply_current_density(df_second_cycle)

                # Extract label from filename
                match = re.search(r"[A-H]\d{1,12}_CV", file_path.stem)
                legend_label = match.group(0) if match else "Unknown"

                # Plot data
                if style == "scatter":
                    plt.scatter(
                        df_second_cycle["Vsig"],
                        df_second_cycle["Im"],
                        label=legend_label,
                        s=5,
                    )
                else:
                    plt.plot(
                        df_second_cycle["Vsig"],
                        df_second_cycle["Im"],
                        label=legend_label,
                    )

                # Set labels and legend
                plt.xlabel("V vs Ag/Ag+ (V)")
                plt.ylabel("Current Density (mA/cm²)")
                plt.ylim(-0.6, 1.0)
                plt.xlim(-0.9, 0.9)
                plt.legend()
                plt.tight_layout()

                # Save and show
                if save_plot:
                    plt.savefig(self.folder_path / f"{file_path.stem}_second_cycle.png")
                    print(f"{file_path.stem}_second_cycle plot saved")

                if show_plot:
                    plt.show()
                else:
                    plt.close()

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")


class CAPlotter(BasePlotter):
    """Plotter for chronoamperometry data."""

    def __init__(
        self,
        folder_path: Union[str, Path],
        lookup_df: Optional[pd.DataFrame] = None,
        geometry: str = "circular",
        diameter: float = 6.5,
    ):
        """Initialize the CA plotter."""
        super().__init__(folder_path, "CA", lookup_df, geometry, diameter)

    def plot_all_curves(
        self,
        colors: List[str] = None,
        save_formats: List[str] = ["png", "svg"],
        show_plot: bool = True,
        save_plot: bool = True,
        excluded_ids: List[int] = None,
        log_time: bool = False,
        log_current: bool = False,
    ) -> None:
        """
        Plot all CA curves in one figure.

        Args:
            colors: List of colors for the gradient (if None, uses defaults)
            save_formats: List of formats to save the plot in
            show_plot: Whether to display the plot
        """
        self._setup_figure()

        # Use default colors if none provided
        if colors is None:
            default_colors = self.color_generator.get_default_colors()["two_color"]
            start_color = default_colors["start"]
            end_color = default_colors["end"]
        else:
            start_color, end_color = colors

        files = load_experiment_files(self.folder_path, f"*{self.experiment_type}*.txt")
        files = [file for file in files if "OCP" not in file.stem]
        files.sort(key=lambda x: int(x.stem.split("_")[2]))

        # Filter out excluded ids if provided
        if excluded_ids:
            files = [
                file
                for file in files
                if int(file.stem.split("_")[2]) not in excluded_ids
            ]

        if not files:
            print(f"No files found with the pattern: {self.experiment_type}")
            return

        # Create colormap
        custom_cmap = self.color_generator.create_colormap(
            len(files), start_color, end_color, "custom_colormap"
        )

        legend_handles = []
        legend_labels = []

        # Plot each file
        for index, file_path in enumerate(files):
            file_name = file_path.stem
            parts_of_name = file_name.split("_")

            # Extract relevant parts for the label
            if len(parts_of_name) >= 8:
                relevant_part = "_".join(parts_of_name[2:4])
            else:
                relevant_part = file_name

            # Load and process data
            df = load_experiment_data(file_path, self.experiment_type)

            # Apply current density conversion
            df = self._apply_current_density(df)

            # Plot with color from the colormap
            color = custom_cmap(index / len(files))
            plt.scatter(
                df["Time"],
                df["Im"],
                color=color,
                s=5,
            )
            if not log_time and not log_current:
                pass
            elif log_time and not log_current:
                plt.xscale("log")
            elif log_current and not log_time:
                plt.yscale("log")
            elif log_time and log_current:
                plt.xscale("log")
                plt.yscale("log")
            else:
                pass

            # Add to legend
            legend_handles.append(Line2D([0], [0], color=color, lw=2))
            legend_labels.append(relevant_part)

        # Set plot labels and legend
        plt.xlabel("Time (s)")
        plt.ylabel("Current Density (mA/cm²)")
        plt.legend(legend_handles, legend_labels)
        plt.tight_layout()

        # Save the figure
        if save_plot:
            self._save_figure(f"{self.experiment_type}_plotall", save_formats)

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            self._close_figure()

    def plot_by_voltage(
        self,
        colors: List[str] = None,
        save_formats: List[str] = ["png", "svg"],
        show_plot: bool = True,
        save_plot: bool = True,
        excluded_ids: List[int] = None,
    ) -> None:
        """
        Plot CA curves sorted by voltage.
        Args:
            colors: List of 5 colors for the gradient (if None, uses defaults)
            save_formats: List of formats to save the plot in
            show_plot: Whether to display the plot
        """
        self._setup_figure()

        # Use default colors if none provided
        if colors is None:
            default_colors = self.color_generator.get_default_colors()["five_color"]
            c1, c2, c3, c4, c5 = (
                default_colors["c1"],
                default_colors["c2"],
                default_colors["c3"],
                default_colors["c4"],
                default_colors["c5"],
            )
        else:
            c1, c2, c3, c4, c5 = colors

        files = load_experiment_files(self.folder_path, f"*{self.experiment_type}*.txt")
        files = [file for file in files if "OCP" not in file.stem]

        if not files:
            print(f"No files found with the pattern: {self.experiment_type}")
            return

        # Get files with voltage information
        file_dep_v = []
        for file_path in files:
            metadata = parse_file_metadata(file_path, self.lookup_df)
            if "dep_V" in metadata:
                file_dep_v.append((file_path, metadata["dep_V"], metadata))

        # Sort files by voltage lowest to highest
        sorted_files = sorted(file_dep_v, key=lambda x: x[1])

        # Create colormap
        custom_cmap = self.color_generator.create_five_color_gradient(
            len(sorted_files), c1, c2, c3, c4, c5, "custom_colormap"
        )

        legend_handles = []
        legend_labels = []

        # Plot each file
        for index, (file_path, dep_V, metadata) in enumerate(sorted_files):
            reference = metadata["reference"]

            # Get deposition time if available
            dep_T = metadata.get("dep_T", "?")

            # Label for the legend
            label = f"{reference}_{dep_V}V_{dep_T}s"

            # Load and process data
            df = load_experiment_data(file_path, self.experiment_type)

            # Apply current density conversion
            df = self._apply_current_density(df)

            # Plot with color from the colormap
            color = custom_cmap(index / len(sorted_files))
            plt.scatter(
                df["Time"],
                df["Im"],
                color=color,
                s=5,
            )

            # Add to legend
            legend_handles.append(Line2D([0], [0], color=color, lw=2))
            legend_labels.append(label)

        # Set plot labels and legend
        plt.xlabel("Time (s)")
        plt.ylabel("Current Density (mA/cm²)")
        plt.legend(legend_handles, legend_labels)
        plt.tight_layout()

        # Save the figure
        if save_plot:
            self._save_figure(f"{self.experiment_type}_all_sortV", save_formats)

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            self._close_figure()

    def plot_by_time(
        self,
        colors: List[str] = None,
        save_formats: List[str] = ["png", "svg"],
        show_plot: bool = True,
        save_plot: bool = True,
        excluded_ids: List[int] = None,
    ) -> None:
        """
        Plot CA curves sorted by duration.

        Args:
            colors: List of 5 colors for the gradient (if None, uses defaults)
            save_formats: List of formats to save the plot in
            show_plot: Whether to display the plot
        """
        self._setup_figure()

        # Use default colors if none provided
        if colors is None:
            default_colors = self.color_generator.get_default_colors()["five_color"]
            c1, c2, c3, c4, c5 = (
                default_colors["c1"],
                default_colors["c2"],
                default_colors["c3"],
                default_colors["c4"],
                default_colors["c5"],
            )
        else:
            c1, c2, c3, c4, c5 = colors

        files = load_experiment_files(self.folder_path, f"*{self.experiment_type}*.txt")
        files = [file for file in files if "OCP" not in file.stem]

        if not files:
            print(f"No files found with the pattern: {self.experiment_type}")
            return

        # Get files with time information
        file_dep_t = []
        for file_path in files:
            metadata = parse_file_metadata(file_path, self.lookup_df)
            if "dep_T" in metadata:
                file_dep_t.append((file_path, metadata["dep_T"], metadata))

        # Sort files by duration, shortest to longest
        sorted_files = sorted(file_dep_t, key=lambda x: int(x[1]))

        # Create colormap
        custom_cmap = self.color_generator.create_five_color_gradient(
            len(sorted_files), c1, c2, c3, c4, c5, "custom_colormap"
        )

        legend_handles = []
        legend_labels = []

        # Plot each file
        for index, (file_path, dep_T, metadata) in enumerate(sorted_files):
            reference = metadata["reference"]

            # Get deposition voltage if available
            dep_V = metadata.get("dep_V", "?")

            # Label for the legend
            label = f"{reference}_{dep_V}V_{dep_T}s"

            # Load and process data
            df = load_experiment_data(file_path, self.experiment_type)

            # Apply current density conversion
            df = self._apply_current_density(df)

            # Plot with color from the colormap
            color = custom_cmap(index / len(sorted_files))
            plt.scatter(
                df["Time"],
                df["Im"],
                color=color,
                s=5,
            )

            # Add to legend
            legend_handles.append(Line2D([0], [0], color=color, lw=2))
            legend_labels.append(label)

        # Set plot labels and legend
        plt.xlabel("Time (s)")
        plt.ylabel("Current Density (mA/cm²)")
        plt.legend(legend_handles, legend_labels)
        plt.tight_layout()

        # Save the figure
        if save_plot:
            self._save_figure(f"{self.experiment_type}_all_sortT", save_formats)

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            self._close_figure()

    def plot_individual(
        self,
        show_plot: bool = True,
        save_plot: bool = True,
        excluded_ids: Optional[List[str | int]] = None,
        log_time: bool = False,
        log_current: bool = False,
        style: str = "scatter",
    ) -> None:
        """
        Create individual plots for each CA file.

        Args:
            show_plot: Whether to display the plots
        """
        files = load_experiment_files(self.folder_path, f"*{self.experiment_type}*.txt")
        files = [file for file in files if "OCP" not in file.stem]
        files.sort(key=lambda x: int(x.stem.split("_")[2]))
        for file_path in files:
            try:
                # Create new figure
                self._setup_figure()

                # Load data
                df = load_experiment_data(file_path, self.experiment_type)

                # Apply current density conversion
                df = self._apply_current_density(df)

                # Plot data
                if style == "scatter":
                    # Plot data
                    plt.scatter(
                        df["Time"],
                        df["Im"],
                        s=5,
                    )
                else:
                    # Plot data
                    plt.plot(
                        df["Time"],
                        df["Im"],
                        s=5,
                    )
                if not log_time and not log_current:
                    pass
                elif log_time and not log_current:
                    plt.xscale("log")
                elif log_current and not log_time:
                    plt.yscale("log")
                elif log_time and log_current:
                    plt.xscale("log")
                    plt.yscale("log")
                else:
                    pass

                # Set labels
                plt.xlabel("Time (s)")
                plt.ylabel("Current Density (mA/cm²)")
                plt.legend([file_path.stem])
                plt.tight_layout()

                # Save the figure
                if save_plot:
                    self._save_figure(f"{file_path.stem}_plot", ["png", "svg"])

                # Show or close
                if show_plot:
                    plt.show()
                else:
                    self._close_figure()

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")


class OCPPlotter(BasePlotter):
    """Plotter for open circuit potential data."""

    def __init__(
        self, folder_path: Union[str, Path], lookup_df: Optional[pd.DataFrame] = None
    ):
        """Initialize the OCP plotter."""
        super().__init__(folder_path, "OCP", lookup_df)

    def plot_all_curves(
        self,
        colors: List[str] = None,
        save_formats: List[str] = ["png", "svg"],
        show_plot: bool = True,
        save_plot: bool = True,
    ) -> None:
        """
        Plot all OCP curves in one figure.

        Args:
            colors: List of colors for the gradient (if None, uses defaults)
            save_formats: List of formats to save the plot in
            show_plot: Whether to display the plot
        """
        self._setup_figure()

        # Use default colors if none provided
        if colors is None:
            default_colors = self.color_generator.get_default_colors()["two_color"]
            start_color = default_colors["start"]
            end_color = default_colors["end"]
        else:
            start_color, end_color = colors

        files = load_experiment_files(self.folder_path, "*OCP*.txt")
        files.sort(key=lambda x: int(x.stem.split("_")[2]))

        if not files:
            print("No OCP files found")
            return

        # Create colormap
        custom_cmap = self.color_generator.create_colormap(
            len(files), start_color, end_color, "custom_colormap"
        )

        legend_handles = []
        legend_labels = []

        # Plot each file
        for index, file_path in enumerate(files):
            file_name = file_path.stem
            parts_of_name = file_name.split("_")

            # Extract relevant parts for the label
            if len(parts_of_name) >= 3:
                relevant_part = "_".join(parts_of_name[2:4])
            else:
                relevant_part = file_name

            # Load data
            df = load_experiment_data(file_path, self.experiment_type)

            # Plot with color from the colormap
            color = custom_cmap(index / len(files))
            plt.scatter(
                df["Time"],
                df["Vf"],
                color=color,
                s=5,
            )

            # Add to legend
            legend_handles.append(Line2D([0], [0], color=color, lw=2))
            legend_labels.append(relevant_part)

        # Set plot labels and legend
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.tight_layout()
        plt.legend(
            legend_handles, legend_labels, bbox_to_anchor=(1.05, 0.5), loc="center left"
        )

        # Save the figure
        if save_plot:
            self._save_figure("OCP_ALL", save_formats)

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            self._close_figure()
