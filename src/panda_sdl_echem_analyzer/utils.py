"""
Utility functions for electrochemical data processing.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def modify_current_density(
    value: float, geometry: str = "circular", diameter: float = 6.5
) -> float:
    """
    Convert current to current density based on electrode geometry.

    Args:
        value: The current value in amps
        geometry: The electrode geometry ('circular' or 'square')
        diameter: The diameter (for circular) or side length (for square) in mm

    Returns:
        The current density value in mA/cm²
    """
    radius = diameter / 2

    if geometry.lower() == "square":
        # Square wells - area in cm²
        area = (diameter / 10) ** 2
    else:
        # Circular wells - area in cm²
        area = math.pi * (radius / 10) ** 2

    # Convert A to mA and divide by area in cm²
    return value * 1000 / area


def load_experiment_files(
    folder_path: Union[str, Path], pattern: str = "*CV*.txt"
) -> List[Path]:
    """
    Find all experimental data files matching a pattern in a directory.

    Args:
        folder_path: Directory containing data files
        pattern: Glob pattern to match specific files

    Returns:
        List of Path objects for matching files
    """
    folder_path = Path(folder_path)
    return list(folder_path.glob(pattern))


def load_lookup_table(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV lookup table containing experimental parameters.

    Args:
        file_path: Path to the CSV lookup file

    Returns:
        DataFrame containing the lookup data
    """
    return pd.read_csv(file_path)


def parse_file_metadata(
    file_path: Path, lookup_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Extract metadata from a file name and optional lookup table.

    Args:
        file_path: Path to data file
        lookup_df: Optional lookup table for additional metadata

    Returns:
        Dictionary of metadata values
    """
    file_name = file_path.stem
    parts = file_name.split("_")

    metadata = {
        "file_name": file_name,
        "file_path": file_path,
    }

    if len(parts) > 3:
        metadata["reference"] = parts[3]

    # Add lookup data if available
    if lookup_df is not None and "reference" in metadata:
        reference = metadata["reference"]
        if reference in lookup_df["well_loc"].values:
            metadata["dep_V"] = lookup_df.loc[
                lookup_df["well_loc"] == reference, "dep_V"
            ].values[0]
            metadata["dep_T"] = lookup_df.loc[
                lookup_df["well_loc"] == reference, "dep_T"
            ].values[0]

    return metadata


def load_experiment_data(file_path: Path, experiment_type: str = "CV") -> pd.DataFrame:
    """
    Load and format data from an experiment file based on experiment type.

    Args:
        file_path: Path to data file
        experiment_type: Type of experiment ('CV', 'CA', or 'OCP')

    Returns:
        DataFrame containing formatted experimental data
    """
    if experiment_type == "CV":
        df = pd.read_csv(
            file_path,
            sep=" ",
            header=None,
            names=[
                "Time",
                "Vf",
                "Vu",
                "Im",
                "Vsig",
                "Ach",
                "IERange",
                "Overload",
                "StopTest",
                "Cycle",
                "Ach2",
            ],
        )
        # Clean up cycle data
        if "Cycle" in df.columns:
            df = df.dropna(subset=["Cycle"])
            df["Cycle"] = df["Cycle"].astype(int)

    elif experiment_type == "CA":
        df = pd.read_csv(
            file_path,
            sep=" ",
            header=None,
            names=[
                "Time",
                "Vf",
                "Vu",
                "Im",
                "Q",
                "Vsig",
                "Ach",
                "IERange",
                "Over",
                "StopTest",
            ],
        )

    elif experiment_type == "OCP":
        df = pd.read_csv(
            file_path,
            sep=" ",
            header=None,
            names=["Time", "Vf", "Vu", "Vsig", "Ach", "Overload", "StopTest", "Temp"],
        )

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    return df
