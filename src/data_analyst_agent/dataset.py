from pathlib import Path
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DatasetManager():
    """Manages Dataset Lifecycle: Loading, Viladation and safe access."""

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._file_path: Optional[pd.DataFrame] = None

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load or viladate CSV or Excel dataset.
       
        Arg:
            file_path: path to the dataset file.
         
        Returns:
            Viladated Pandas DataFrame.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: If format is unsupported or dataset is empty. 
        """

        path = Path(file_path)


        if not path.exists():
            logger.error("Dataset file not found: %s", path.resolve())
            raise FileNotFoundError(f"Dataset not found: {path}")

        extension = path.suffix.lower()

        if extension == ".csv":
          self._df = pd.read_csv(path)
        elif extension in  {".xlsx","xlx"}:
            self._df = pd.read_excel(path)
        else:
            logger.error("Unsupported file format %s for path: %s", extension, path)
            raise ValueError(
                f"Unsupprted  file format '{extension}'."
                "Support only CSV or Excel file format."
            )      
        
        # Validate immediately after load
        if self._df.empty:
            logger.warning("Loaded dataset contain no data rows: %s", path.name)
            self._df = None
            raise ValueError("The uploaded dataset contains no data rows")

        self._file_path = path

        logger.info(
            "Dataset loaded successfully: %s (%d rows, %d columns)",         
            path.name,
            self._df.shape[0],
            self._df.shape[1]

        )
        return self._df

    def get_dataframe(self) -> pd.DataFrame:
        """Return a COPY of the loaded DataFrame to prevent accidental mutation.

        Returns:
            A copy of the current dataset.

        Raises:
            ValueError: If no dataset has been loaded yet.
        """
        if self._df is None:
            logger.error("get_dataframe() called before loaded_dataset()")
            raise ValueError("No dataset has been loaded.Call load_dataset() first")
        return self._df.copy()
