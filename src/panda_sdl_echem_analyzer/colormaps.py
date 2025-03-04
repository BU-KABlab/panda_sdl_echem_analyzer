"""
Functions for creating custom colormaps for electrochemical data visualization.
"""

from typing import Optional

import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap


class ColorMapGenerator:
    """
    Generate custom colormaps for data visualization.
    """

    @staticmethod
    def create_colormap(
        num_colors: int, start_color: str, end_color: str, name: Optional[str] = None
    ) -> ListedColormap:
        """
        Create a two-color gradient colormap.

        Args:
            num_colors: Number of color steps in the gradient
            start_color: Starting color in hex format (e.g., '#9c00ff')
            end_color: Ending color in hex format (e.g., '#3c943c')
            name: Name for the colormap

        Returns:
            A matplotlib ListedColormap
        """
        start_rgb = mcolors.hex2color(start_color)
        end_rgb = mcolors.hex2color(end_color)

        color_list = []
        for i in range(num_colors):
            r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * i / (num_colors - 1)
            g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * i / (num_colors - 1)
            b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * i / (num_colors - 1)
            color_list.append((r, g, b))

        if name is None:
            name = f"gradient_{start_color}_{end_color}"

        return ListedColormap(color_list, name=name)

    @staticmethod
    def create_three_color_gradient(
        num_colors: int,
        start_color: str,
        middle_color: str,
        end_color: str,
        name: Optional[str] = None,
    ) -> ListedColormap:
        """
        Create a three-color gradient colormap.

        Args:
            num_colors: Number of color steps in the gradient
            start_color: Starting color in hex format
            middle_color: Middle color in hex format
            end_color: Ending color in hex format
            name: Name for the colormap

        Returns:
            A matplotlib ListedColormap
        """
        start_rgb = mcolors.hex2color(start_color)
        middle_rgb = mcolors.hex2color(middle_color)
        end_rgb = mcolors.hex2color(end_color)

        # Calculate the number of colors for each segment
        num_colors_first_half = num_colors // 2
        num_colors_second_half = num_colors - num_colors_first_half

        color_list = []

        # First segment: from start to middle
        for i in range(num_colors_first_half):
            r = start_rgb[0] + (middle_rgb[0] - start_rgb[0]) * i / max(
                num_colors_first_half - 1, 1
            )
            g = start_rgb[1] + (middle_rgb[1] - start_rgb[1]) * i / max(
                num_colors_first_half - 1, 1
            )
            b = start_rgb[2] + (middle_rgb[2] - start_rgb[2]) * i / max(
                num_colors_first_half - 1, 1
            )
            color_list.append((r, g, b))

        # Second segment: from middle to end
        for i in range(num_colors_second_half):
            r = middle_rgb[0] + (end_rgb[0] - middle_rgb[0]) * i / max(
                num_colors_second_half - 1, 1
            )
            g = middle_rgb[1] + (end_rgb[1] - middle_rgb[1]) * i / max(
                num_colors_second_half - 1, 1
            )
            b = middle_rgb[2] + (end_rgb[2] - middle_rgb[2]) * i / max(
                num_colors_second_half - 1, 1
            )
            color_list.append((r, g, b))

        if name is None:
            name = f"gradient3_{start_color}_{middle_color}_{end_color}"

        return ListedColormap(color_list, name=name)

    @staticmethod
    def create_five_color_gradient(
        num_colors: int,
        c1: str,
        c2: str,
        c3: str,
        c4: str,
        c5: str,
        name: Optional[str] = None,
    ) -> ListedColormap:
        """
        Create a gradient colormap with five colors.

        Args:
            num_colors: Number of color steps in the gradient
            c1, c2, c3, c4, c5: Color hex values defining the gradient points
            name: Name for the colormap

        Returns:
            A matplotlib ListedColormap
        """
        # Convert the hex colors to RGB
        c1_rgb = mcolors.hex2color(c1)
        c2_rgb = mcolors.hex2color(c2)
        c3_rgb = mcolors.hex2color(c3)
        c4_rgb = mcolors.hex2color(c4)
        c5_rgb = mcolors.hex2color(c5)

        # Calculate the number of colors for each segment
        num_colors_per_segment = num_colors // 4
        remaining_colors = num_colors % 4  # This will be added to the last segment

        color_list = []

        # First segment: from c1 to c2
        for i in range(num_colors_per_segment):
            r = c1_rgb[0] + (c2_rgb[0] - c1_rgb[0]) * i / max(
                num_colors_per_segment - 1, 1
            )
            g = c1_rgb[1] + (c2_rgb[1] - c1_rgb[1]) * i / max(
                num_colors_per_segment - 1, 1
            )
            b = c1_rgb[2] + (c2_rgb[2] - c1_rgb[2]) * i / max(
                num_colors_per_segment - 1, 1
            )
            color_list.append((r, g, b))

        # Second segment: from c2 to c3
        for i in range(num_colors_per_segment):
            r = c2_rgb[0] + (c3_rgb[0] - c2_rgb[0]) * i / max(
                num_colors_per_segment - 1, 1
            )
            g = c2_rgb[1] + (c3_rgb[1] - c2_rgb[1]) * i / max(
                num_colors_per_segment - 1, 1
            )
            b = c2_rgb[2] + (c3_rgb[2] - c2_rgb[2]) * i / max(
                num_colors_per_segment - 1, 1
            )
            color_list.append((r, g, b))

        # Third segment: from c3 to c4
        for i in range(num_colors_per_segment):
            r = c3_rgb[0] + (c4_rgb[0] - c3_rgb[0]) * i / max(
                num_colors_per_segment - 1, 1
            )
            g = c3_rgb[1] + (c4_rgb[1] - c3_rgb[1]) * i / max(
                num_colors_per_segment - 1, 1
            )
            b = c3_rgb[2] + (c4_rgb[2] - c3_rgb[2]) * i / max(
                num_colors_per_segment - 1, 1
            )
            color_list.append((r, g, b))

        # Fourth segment: from c4 to c5
        num_colors_last_segment = num_colors_per_segment + remaining_colors
        for i in range(num_colors_last_segment):
            r = c4_rgb[0] + (c5_rgb[0] - c4_rgb[0]) * i / max(
                num_colors_last_segment - 1, 1
            )
            g = c4_rgb[1] + (c5_rgb[1] - c4_rgb[1]) * i / max(
                num_colors_last_segment - 1, 1
            )
            b = c4_rgb[2] + (c5_rgb[2] - c4_rgb[2]) * i / max(
                num_colors_last_segment - 1, 1
            )
            color_list.append((r, g, b))

        if name is None:
            name = f"gradient5_{c1}_{c5}"

        return ListedColormap(color_list, name=name)

    @classmethod
    def get_default_colors(cls) -> dict:
        """
        Get a set of default colors for different types of plots.

        Returns:
            Dictionary of default color schemes
        """
        return {
            # Five color gradient
            "five_color": {
                "c1": "#50325E",  # Deep purple
                "c2": "#8474A1",  # Light purple
                "c3": "#41A69C",  # Teal
                "c4": "#3c943c",  # Green
                "c5": "#FF9C79",  # Coral
            },
            # Three color gradient
            "three_color": {
                "start": "#623954",  # Purple
                "middle": "#CA7682",  # Coral
                "end": "#3c943c",  # Green
            },
            # Two color gradient
            "two_color": {
                "start": "#9c00ff",  # Purple
                "end": "#3c943c",  # Green
            },
        }
