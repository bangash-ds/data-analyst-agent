import json
import logging
import re
from typing import List
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from langchain.tools import tool

from .schemas import (
    ColumnAnalysisRequest,
    GroupAnalysisRequest,
    ChartRequest,
    ReportRequest,
    CleaningRequest  
)

from .config import (
    CHART_DIR,
    REPORT_DIR,
    DATA_DIR  
)

logger = logging.getLogger(__name__)


def create_tools(dataset_manager) -> List:
    """
    Factory function that creates all agent tools.

    Uses closure to inject dataset_manager into every tool,
    keeping tools decoupled from dataset creation logic.

    Args:
        dataset_manager: An instance of DatasetManager.

    Returns:
        A list of LangChain tool objects.
    """


    # TOOL 1: Dataset Overview

    @tool
    def dataset_overview() -> str:
        """
        Get dataset dimensions, column names,
        data types and sample records.
        """
        logger.info("Executing tool: dataset_overview")

        try:
            df = dataset_manager.get_dataframe()
        except ValueError as error:
            logger.warning("dataset_overview failed: %s", error)
            return f"Error: {error}"

        result = {
            "rows": int(df.shape[0]),
            "columns_count": int(df.shape[1]),
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "sample_records": df.head(5).to_dict(
                orient="records"
            ),
        }

        logger.debug(
            "Overview generated: %d rows, %d columns",
            df.shape[0],
            df.shape[1],
        )

        return json.dumps(result, default=str)


    # TOOL 2: Data Quality

    @tool
    def data_quality() -> str:
        """
        Identify missing values,
        duplicate rows and missing percentages.
        """
        logger.info("Executing tool: data_quality")

        try:
            df = dataset_manager.get_dataframe()
        except ValueError as error:
            logger.warning("data_quality failed: %s", error)
            return f"Error: {error}"

        missing = df.isnull().sum()

        result = {
            "missing_values": missing.to_dict(),
            "missing_percentage": (
                missing.div(len(df)).mul(100)
                .round(2).to_dict()
            ),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_rows": int(len(df)),
        }

        logger.debug(
            "Quality check complete: %d missing cells, %d duplicates",
            int(missing.sum()),
            int(df.duplicated().sum()),
        )

        return json.dumps(result, default=str)


    # TOOL 3: Statistical Analysis

    @tool
    def statistical_analysis() -> str:
        """
        Calculate descriptive statistics
        for numerical dataset columns.
        """
        logger.info("Executing tool: statistical_analysis")

        try:
            df = dataset_manager.get_dataframe()
        except ValueError as error:
            logger.warning("statistical_analysis failed: %s", error)
            return f"Error: {error}"

        numerical_df = df.select_dtypes(include=np.number)

        if numerical_df.empty:
            logger.warning("No numerical columns found for statistics")
            return "No numerical columns found."

        result = numerical_df.describe().round(2)

        logger.debug(
            "Statistics computed for %d numerical columns",
            numerical_df.shape[1],
        )

        return result.to_json()


    # TOOL 4: Individual Column Analysis

    @tool(args_schema=ColumnAnalysisRequest)
    def column_analysis(column: str) -> str:
        """
        Analyze an individual dataset column.

        Returns its data type, unique values,
        missing values, frequency and statistics.
        """
        logger.info("Executing tool: column_analysis for '%s'", column)

        try:
            df = dataset_manager.get_dataframe()
        except ValueError as error:
            logger.warning("column_analysis failed: %s", error)
            return f"Error: {error}"

        if column not in df.columns:
            logger.warning("Column '%s' not found in dataset", column)
            return f"Column '{column}' does not exist."

        series = df[column]

        result = {
            "column": column,
            "data_type": str(series.dtype),
            "unique_values": int(series.nunique()),
            "missing_values": int(series.isnull().sum()),
            "value_counts": series.value_counts(
                dropna=False
            ).head(20).to_dict(),
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

        logger.debug(
            "Column '%s' analyzed: dtype=%s, unique=%d",
            column,
            series.dtype,
            series.nunique(),
        )

        return json.dumps(result, default=str)


    # TOOL 5: Correlation Analysis

    @tool
    def correlation_analysis() -> str:
        """
        Calculate correlations between
        numerical dataset columns.
        """
        logger.info("Executing tool: correlation_analysis")

        try:
            df = dataset_manager.get_dataframe()
        except ValueError as error:
            logger.warning("correlation_analysis failed: %s", error)
            return f"Error: {error}"

        numerical_df = df.select_dtypes(include=np.number)

        if numerical_df.shape[1] < 2:
            logger.warning("Fewer than 2 numerical columns for correlation")
            return "At least two numerical columns are required."

        correlation = numerical_df.corr()

        logger.debug(
            "Correlation matrix computed: %dx%d",
            correlation.shape[0],
            correlation.shape[1],
        )

        return correlation.round(3).to_json()


    # TOOL 6: Group Analysis

    @tool(args_schema=GroupAnalysisRequest)
    def group_analysis(
        group_column: str,
        target_column: str,
        operation: str = "mean",
    ) -> str:
        """
        Group data using one column and
        aggregate another column.

        Supported operations:
        mean, sum, count, min, max.
        """
        logger.info(
            "Executing tool: group_analysis | group=%s, target=%s, op=%s",
            group_column,
            target_column,
            operation,
        )

        try:
            df = dataset_manager.get_dataframe()
        except ValueError as error:
            logger.warning("group_analysis failed: %s", error)
            return f"Error: {error}"

        if group_column not in df.columns:
            logger.warning("Group column '%s' not found", group_column)
            return "Grouping column not found."

        if target_column not in df.columns:
            logger.warning("Target column '%s' not found", target_column)
            return "Target column not found."

        valid_operations = ["mean", "sum", "count", "min", "max"]

        if operation not in valid_operations:
            logger.warning("Unsupported operation: %s", operation)
            return "Unsupported aggregation operation."

        if operation in ["mean", "sum", "min", "max"]:
            if not pd.api.types.is_numeric_dtype(df[target_column]):
                logger.warning(
                    "Operation '%s' requires numerical target, got '%s'",
                    operation,
                    df[target_column].dtype,
                )
                return (
                    f"{operation} requires a numerical "
                    "target column."
                )

        result = (
            df.groupby(group_column)[target_column]
            .agg(operation)
            .sort_values(ascending=False)
        )

        logger.debug(
            "Group analysis complete: %d groups found",
            len(result),
        )

        return result.head(50).to_json()


    # TOOL 7: Visualization

    @tool(args_schema=ChartRequest)
    def visualize_data(
        column: str,
        chart_type: str = "histogram",
    ) -> str:
        """
        Generate a histogram, boxplot or countplot
        for a dataset column.
        """
        logger.info(
            "Executing tool: visualize_data | column=%s, type=%s",
            column,
            chart_type,
        )

        try:
            df = dataset_manager.get_dataframe()
        except ValueError as error:
            logger.warning("visualize_data failed: %s", error)
            return f"Error: {error}"

        if column not in df.columns:
            logger.warning("Column '%s' not found for visualization", column)
            return "Column not found."

        series = df[column]

        if chart_type in ["histogram", "boxplot"]:
            if not pd.api.types.is_numeric_dtype(series):
                logger.warning(
                    "%s requires numerical column, got '%s'",
                    chart_type,
                    series.dtype,
                )
                return (
                    "Histogram and boxplot require "
                    "numerical columns."
                )

        if chart_type == "countplot":
            order = series.value_counts().head(20).index

        plt.figure(figsize=(10, 6))

        if chart_type == "histogram":
            sns.histplot(data=df, x=column, kde=True)

        elif chart_type == "boxplot":
            sns.boxplot(data=df, y=column)

        elif chart_type == "countplot":
            sns.countplot(data=df, x=column, order=order)
            plt.xticks(rotation=45)

        else:
            plt.close()
            logger.warning("Unsupported chart type: %s", chart_type)
            return "Unsupported chart type."

        plt.title(f"{chart_type.title()} of {column}")
        plt.tight_layout()

        safe_column = re.sub(r"[^a-zA-Z0-9_-]+", "_", column)
        filename = f"{safe_column}_{chart_type}.png"
        save_path = CHART_DIR / filename

        plt.savefig(save_path, dpi=150)
        plt.close()

        logger.info("Chart saved: %s", save_path)

        return f"Chart generated successfully: {save_path}"


    # TOOL 8: Save Analysis Report

    @tool(args_schema=ReportRequest)
    def save_report(title: str, report: str) -> str:
        """
        Save the completed analysis report
        as a Markdown file.
        """
        logger.info("Executing tool: save_report | title='%s'", title)

        safe_title = re.sub(
            r"[^a-zA-Z0-9_-]+", "_", title
        ).strip("_")

        if not safe_title:
            safe_title = "analysis_report"

        report_path = REPORT_DIR / f"{safe_title}.md"

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(f"# {title}\n\n")
            file.write(report)

        logger.info("Report saved: %s", report_path)

        return (
            f"Report saved successfully at: {report_path}"
        )

    @tool(args_schema=CleaningRequest)
    def clean_dataset(strategy: str, columns: list[str] = None) -> str:
        """Clean missing values using an approved strategy.

        Strategies:
        - drop_missing_rows: Remove rows with any missing value in target columns
        - fill_numeric_mean: Fill numeric columns with their mean
        - fill_numeric_median: Fill numeric columns with their median  
        - fill_categorical_mode: Fill categorical columns with most frequent value

        Saves cleaned CSV to data/ directory without modifying original dataset.
        Returns path to cleaned file and summary statistics.
        """
        if columns is None:
            columns = []

        logger.info("Executing tool: clean_dataset | strategy=%s | columns=%s", strategy, columns)

        try:
            df = dataset_manager.get_dataframe().copy()
        except ValueError as error:
            logger.warning("clean_dataset failed: %s", error)
            return f"Error: {error}"

        available_columns = df.columns.tolist()

        # Validate requested columns
        invalid_columns = [col for col in columns if col not in available_columns]
        if invalid_columns:
            return f"Invalid columns: {invalid_columns}. Available: {available_columns}"

        # Auto-select applicable columns if none specified
        if not columns:
            if strategy == "drop_missing_rows":
                columns = available_columns
            elif strategy in ("fill_numeric_mean", "fill_numeric_median"):
                columns = df.select_dtypes(include=np.number).columns.tolist()
            elif strategy == "fill_categorical_mode":
                columns = df.select_dtypes(exclude=np.number).columns.tolist()

        if not columns:
            return f"No applicable columns found for strategy '{strategy}'."

        # Track missing counts before cleaning
        missing_before = int(df[columns].isnull().sum().sum())

        # Apply cleaning strategy
        if strategy == "drop_missing_rows":
            df = df.dropna(subset=columns)
        elif strategy == "fill_numeric_mean":
            for col in columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
        elif strategy == "fill_numeric_median":
            for col in columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
        elif strategy == "fill_categorical_mode":
            for col in columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])

        missing_after = int(df[columns].isnull().sum().sum())

        # Save cleaned dataset
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        safe_name = Path(dataset_manager.file_path).stem if hasattr(dataset_manager, 'file_path') else "dataset"
        output_path = DATA_DIR / f"{safe_name}_cleaned_{timestamp}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(
            "Dataset cleaned: %d → %d missing values | Saved to %s",
            missing_before, missing_after, output_path
        )

        result = {
            "strategy": strategy,
            "columns_cleaned": columns,
            "missing_values_before": missing_before,
            "missing_values_after": missing_after,
            "rows_removed": int(len(dataset_manager.get_dataframe()) - len(df)),
            "cleaned_file_path": str(output_path),
        }

        return json.dumps(result, default=str)


    # Return all tools

    return [
        dataset_overview,
        data_quality,
        statistical_analysis,
        column_analysis,
        correlation_analysis,
        group_analysis,
        visualize_data,
        save_report,
        clean_dataset,  
    ]