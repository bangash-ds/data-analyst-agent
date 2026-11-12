import json
import re
import logging
import pandas as pd
import numpy as np
from langchain.tools import tool
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
import matplotlib

matplotlib.use("Agg")

log = logging.getLogger(__name__)

from .schema import (
    ColumnAnalysisRequest,
    GroupAnalysisRequest,
    ChartRequest,
    ReportRequest
)

from .config import (
    CHART_DIR,
    REPORT_DIR
)


# Factory function that contains all tools
def create_tools(dataset_manager) -> List:
    """
    Factory function that creates all agent tools.

    Uses closure to inject dataset_manager into every tool
    keeping tools decoupled from dataset creation logic.

    Args:
        dataset_manager: An instance of DatasetManager.

    Returns:
        A list of LangChain tool objects.
    """

    @tool
    def dataset_overview() -> str:
        """
        Get dataset dimensions, column names,
        data types and sample records.
        """
        log.info("Executing tool: dataset_overview")
        try:
            df = dataset_manager.get_dataframe()
        except ValueError as error:
            log.warning("Dataset overview failed: %s", error)
            return f"Error: {error}"

        result = {
            "rows": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "sample_records": df.head(5).to_dict(
                orient="records"
            )
        }

        log.debug(
            "Overview generated: %d rows, %d columns",
            df.shape[0],
            df.shape[1]
        )

        return json.dumps(result, default=str)

    @tool
    def data_quality() -> str:
        """
        Identify missing values,
        duplicated rows and missing percentages.
        """

        log.info("Executing tool: data_quality")

        try:
            df = dataset_manager.get_dataframe()
        except ValueError as error:
            log.warning("data_quality failed: %s", error)
            return f"Error: {error}"

        missing = df.isnull().sum()

        result = {
            "missing_values": missing.to_dict(),
            "missing_percentage": (missing / len(df) * 100).round(2).to_dict(),
            "duplicated_rows": int(df.duplicated().sum()),
            "total_rows": int(len(df))
        }

        log.debug(
            "Quality check completed: %d missing cells and %d duplicates",
            int(missing.sum()),
            int(df.duplicated().sum())
        )

        return json.dumps(result, default=str)

    @tool
    def statistical_analysis() -> str:
        """
        Calculate descriptive statistics
        for numerical columns.
        """

        log.info("Executing tool: statistical_analysis")
        try:
            df = dataset_manager.get_dataframe()
        except ValueError as error:
            log.warning("statistical_analysis failed: %s", error)
            return f"Error: {error}"

        numerical_df = df.select_dtypes(include=np.number)

        if numerical_df.empty:
            log.warning("No numerical columns found for statistics")
            return "No numerical columns found."

        result = numerical_df.describe().round(2)

        log.debug(
            "Statistics computed for %d numerical columns",
            numerical_df.shape[1]
        )

        return result.to_json()

    # TOOL 4: Individual Column Analysis
    @tool(args_schema=ColumnAnalysisRequest)
    def column_analysis(column: str) -> str:
        """Analyze an individual dataset column.

        Returns its data type, unique values,
        missing values, frequency and statistics."""

        log.info("Executing tool: column_analysis for '%s'", column)
        try:
            df = dataset_manager.get_dataframe()
        except ValueError as error:
            log.warning("column_analysis failed: %s", error)
            return f"Error: {error}"

        if column not in df.columns:
            log.warning("Column '%s' not found in dataset", column)
            return f"Column '{column}' does not exist."

        series = df[column]

        result = {
            "column": column,
            "data_type": str(series.dtype),
            "unique_values": int(series.nunique()),
            "missing_values": int(series.isnull().sum()),
            "value_counts": series.value_counts(
                dropna=False
            ).head(20).to_dict()
        }

        if pd.api.types.is_numeric_dtype(series):
            result["mean"] = (
                float(series.mean())
                if series.notna().any()
                else None
            )

            result["median"] = (
                float(series.median())
                if series.notna().any()
                else None
            )

            result["standard_deviation"] = (
                float(series.std())
                if series.notna().any()
                else None
            )

        log.debug(
            "Column %s analyzed: dtype=%s, unique=%d",
            column,
            series.dtype,
            series.nunique()
        )

        return json.dumps(result, default=str)

        # Return all tools
    return [
        dataset_overview,
        data_quality,
        statistical_analysis,
        column_analysis,
    ]