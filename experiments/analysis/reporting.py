"""Shared plotting setup and the LaTeX table renderer used by the RQ notebooks."""

from __future__ import annotations

import matplotlib as mpl
import pandas as pd
import seaborn as sns

PALETTE = sns.color_palette("tab10")


def tex(s: str) -> str:
    """Escape a plain string for LaTeX text mode (underscores -> spaces)."""
    return s.replace("_", " ")


def setup_matplotlib() -> None:
    """Configure matplotlib: LaTeX rendering, serif font, 18 pt."""
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "font.family": "serif",
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "legend.title_fontsize": 16,
            "figure.titlesize": 22,
        }
    )


class MetricTable:
    """Render one or more metric groups in a shared LaTeX table."""

    MODES = ["multi", "image", "text"]

    MODE_LABELS = {
        "multi": r"\faImage\;+ \faFont",
        "image": r"\faImage",
        "text": r"\faFont",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        title: str,
        model_mapping: dict[str, str],
        metric_mapping: dict[str, str],
        scene_mapping: dict[str, str],
        shared_metrics: set[str] | None = None,
    ) -> None:
        self.model_mapping = model_mapping
        self.scene_mapping = scene_mapping

        self.tables = [
            {
                "title": title,
                "df": df.copy(),
                "metrics": metric_mapping,
                "shared_metrics": shared_metrics or set(),
            }
        ]

    def __add__(self, other: MetricTable) -> MetricTable:
        if self.model_mapping != other.model_mapping:
            raise ValueError("Model mappings differ.")

        if self.scene_mapping != other.scene_mapping:
            raise ValueError("Scene mappings differ.")

        merged = MetricTable.__new__(MetricTable)
        merged.model_mapping = self.model_mapping
        merged.scene_mapping = self.scene_mapping
        merged.tables = self.tables + other.tables
        return merged

    @staticmethod
    def _value(value: object) -> str:
        if value is None:
            return "---"

        if isinstance(value, str):
            return value if value else "---"

        if pd.isna(value):
            return "---"

        return str(value)

    def __repr__(self) -> str:
        metric_count = sum(len(table["metrics"]) for table in self.tables)

        separator_count = len(self.tables) - 1
        column_count = 2 + metric_count + separator_count
        alignment = "ll" + "c" * (metric_count + separator_count)

        lines = [
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
        ]

        if len(self.tables) == 1:
            table = self.tables[0]
            metric_titles = " & ".join(table["metrics"].values())

            lines.append(f"Model & Scene & {metric_titles} \\\\")
        else:
            group_headers = []
            detail_headers = ["", ""]
            cmidrules = []

            column = 3

            for index, table in enumerate(self.tables):
                width = len(table["metrics"])
                title = table["title"]

                group_headers.append(rf"\multicolumn{{{width}}}{{c}}{{\textsc{{{title}}}}}")

                detail_headers.extend(table["metrics"].values())

                cmidrules.append(rf"\cmidrule(lr){{{column}-{column + width - 1}}}")

                column += width

                if index < len(self.tables) - 1:
                    group_headers.append("")
                    detail_headers.append("")
                    column += 1

            lines.append(
                r"\multirow{2}{*}{Model} & "
                r"\multirow{2}{*}{Scene} & " + " & ".join(group_headers) + r" \\"
            )
            lines.extend(cmidrules)
            lines.append(" & ".join(detail_headers) + r" \\")

        lines.append(r"\midrule")

        model_items = list(self.model_mapping.items())
        scene_items = list(self.scene_mapping.items())

        for model_index, (model_key, model_label) in enumerate(model_items):
            model_span = 3 * len(scene_items)
            first_model_row = True

            for scene_index, (scene_key, scene_label) in enumerate(scene_items):
                for mode_index, mode in enumerate(self.MODES):
                    cells = []

                    if first_model_row:
                        cells.append(rf"\multirow{{{model_span}}}{{*}}{{{model_label}}}")
                        first_model_row = False
                    else:
                        cells.append("")

                    if mode_index == 0:
                        cells.append(rf"\multirow{{3}}{{*}}{{{scene_label}}}")
                    else:
                        cells.append("")

                    for table_index, table in enumerate(self.tables):
                        df = table["df"]
                        metrics = table["metrics"]
                        shared_metrics = table["shared_metrics"]

                        selected = df[
                            (df["model"] == model_key)
                            & (df["scene"] == scene_key)
                            & (df["genome_mode"] == mode)
                        ]

                        row = selected.iloc[0] if not selected.empty else None

                        for metric in metrics:
                            if metric in shared_metrics and mode_index > 0:
                                cells.append("")
                                continue

                            value = self._value(row[metric]) if row is not None else "---"

                            if metric in shared_metrics:
                                value = rf"\multirow{{3}}{{*}}{{{value}}}"

                            cells.append(value)

                        if table_index < len(self.tables) - 1:
                            cells.append("")

                    lines.append(" & ".join(cells) + r" \\")

                if scene_index < len(scene_items) - 1:
                    lines.append(rf"\cmidrule(lr){{2-{column_count}}}")

            if model_index < len(model_items) - 1:
                lines.append(r"\midrule")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )

        return "\n".join(lines)
