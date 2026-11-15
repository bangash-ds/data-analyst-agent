from pydantic import BaseModel, Field
from typing import Literal


# 1. Column Analysis Schema
class ColumnAnalysisRequest(BaseModel):
    column: str = Field(description="Column name to analyze")


# 2. Group Analysis Schema
class GroupAnalysisRequest(BaseModel):
    group_column: str = Field(description="Column used for grouping")
    target_column: str = Field(
        description="Numerical or categorical column to aggregate"
    )
    operation: Literal["mean", "sum", "count", "min", "max"] = Field(
        default="mean", description="Aggregation operation"
    )


# 3. Visualization Schema
class ChartRequest(BaseModel):
    column: str = Field(description="Column to visualize")
    chart_type: Literal["histogram", "boxplot", "countplot"] = Field(
        default="histogram", description="Type of visualization"
    )


# 4. Report Schema
class ReportRequest(BaseModel):
    title: str = Field(description="Title of the analysis report")
    report: str = Field(
        min_length=20, description="Complete analysis report in markdown"
    )
