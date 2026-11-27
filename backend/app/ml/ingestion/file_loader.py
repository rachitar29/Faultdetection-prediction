
"""
Dynamic file ingestion for the predictive maintenance application.

Currently supports:
- CSV files
- Excel files (.xlsx and .xls)
- PDF files containing extractable tables

All processing is local and offline.

Scanned PDFs require a separate OCR stage.
"""

from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".pdf",
}


class FileIngestionError(Exception):
    """Raised when an uploaded file cannot be processed."""


def validate_file(file_path, max_file_size_bytes=None):
    """
    Validate file existence, extension, file type and optional size.

    Args:
        file_path: Path to the uploaded file.
        max_file_size_bytes: Optional maximum permitted size in bytes.
            None means no application-level size limit.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    if not path.is_file():
        raise FileIngestionError(
            "The supplied path is not a file."
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise FileIngestionError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    file_size = path.stat().st_size

    if file_size == 0:
        raise FileIngestionError(
            "The uploaded file is empty."
        )

    if max_file_size_bytes is not None:
        if (
            not isinstance(max_file_size_bytes, int)
            or isinstance(max_file_size_bytes, bool)
            or max_file_size_bytes <= 0
        ):
            raise ValueError(
                "max_file_size_bytes must be a positive integer or None."
            )

        if file_size > max_file_size_bytes:
            raise FileIngestionError(
                f"File size ({file_size} bytes) exceeds the permitted "
                f"limit ({max_file_size_bytes} bytes)."
            )

    return path


def load_csv(path):
    """Load CSV data with common encoding and delimiter detection."""

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            dataframe = pd.read_csv(
                path,
                encoding=encoding,
                sep=None,
                engine="python",
            )

            if dataframe.empty:
                raise FileIngestionError(
                    "The CSV file contains no data rows."
                )

            return dataframe

        except UnicodeDecodeError as error:
            last_error = error

        except pd.errors.EmptyDataError as error:
            raise FileIngestionError(
                "The CSV file contains no readable data."
            ) from error

        except pd.errors.ParserError as error:
            last_error = error

        except FileIngestionError:
            raise

    raise FileIngestionError(
        f"Unable to parse CSV file: {last_error}"
    ) from last_error


def load_excel(path, sheet_name=0):
    """
    Load an Excel worksheet.

    Pandas selects the appropriate Excel engine based on the file
    extension. For .xlsx files, openpyxl is commonly required.
    For legacy .xls files, an appropriate engine such as xlrd may
    be required.
    """

    try:
        dataframe = pd.read_excel(
            path,
            sheet_name=sheet_name,
        )

    except ImportError as error:
        raise FileIngestionError(
            "Excel support requires an appropriate pandas Excel engine. "
            "Install openpyxl for .xlsx files or xlrd for .xls files."
        ) from error

    except Exception as error:
        raise FileIngestionError(
            f"Unable to read Excel file: {error}"
        ) from error

    if dataframe.empty:
        raise FileIngestionError(
            "The selected Excel worksheet is empty."
        )

    return dataframe


def _validate_pdf_table_headers(headers, page_number, table_number):
    """Validate and normalize headers extracted from a PDF table."""

    if not headers:
        raise FileIngestionError(
            f"PDF page {page_number}, table {table_number}: "
            "No table headers were found."
        )

    normalized_headers = []

    for index, header in enumerate(headers):
        if header is None:
            raise FileIngestionError(
                f"PDF page {page_number}, table {table_number}: "
                f"Header at position {index} is empty."
            )

        header_name = str(header).strip()

        if not header_name:
            raise FileIngestionError(
                f"PDF page {page_number}, table {table_number}: "
                f"Header at position {index} is blank."
            )

        normalized_headers.append(header_name)

    if len(normalized_headers) != len(set(normalized_headers)):
        raise FileIngestionError(
            f"PDF page {page_number}, table {table_number}: "
            f"Duplicate headers detected: {normalized_headers}"
        )

    if "source_page" in normalized_headers:
        raise FileIngestionError(
            f"PDF page {page_number}, table {table_number}: "
            "'source_page' is reserved for ingestion metadata."
        )

    return normalized_headers


def load_pdf(path):
    """
    Extract tables from a PDF using pdfplumber.

    Tables from different pages are combined only when they have
    the same set of column names. Column order is normalized to
    the first successfully extracted table.
    """

    try:
        import pdfplumber

    except ImportError as error:
        raise FileIngestionError(
            "PDF table extraction requires pdfplumber. "
            "Install it from an offline wheel when available."
        ) from error

    extracted_tables = []
    expected_headers = None

    try:
        with pdfplumber.open(path) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1,
            ):
                tables = page.extract_tables()

                for table_number, table in enumerate(
                    tables,
                    start=1,
                ):
                    if not table or len(table) < 2:
                        continue

                    headers = _validate_pdf_table_headers(
                        table[0],
                        page_number,
                        table_number,
                    )

                    if expected_headers is None:
                        expected_headers = headers

                    elif set(headers) != set(expected_headers):
                        missing_headers = sorted(
                            set(expected_headers) - set(headers)
                        )

                        unexpected_headers = sorted(
                            set(headers) - set(expected_headers)
                        )

                        raise FileIngestionError(
                            f"PDF page {page_number}, table "
                            f"{table_number} has an inconsistent schema. "
                            f"Missing headers: {missing_headers}. "
                            f"Unexpected headers: {unexpected_headers}."
                        )

                    dataframe = pd.DataFrame(
                        table[1:],
                        columns=headers,
                    )

                    # Normalize column order before concatenation.
                    dataframe = dataframe.reindex(
                        columns=expected_headers
                    )

                    dataframe["source_page"] = page_number

                    extracted_tables.append(dataframe)

    except FileIngestionError:
        raise

    except Exception as error:
        raise FileIngestionError(
            f"Unable to extract PDF tables: {error}"
        ) from error

    if not extracted_tables:
        raise FileIngestionError(
            "No readable tables were found in the PDF. "
            "Scanned PDFs require an offline OCR stage."
        )

    combined_dataframe = pd.concat(
        extracted_tables,
        ignore_index=True,
    )

    if combined_dataframe.empty:
        raise FileIngestionError(
            "PDF tables were extracted, but they contain no data rows."
        )

    return combined_dataframe


def load_machine_data(
    file_path,
    sheet_name=0,
    max_file_size_bytes=None,
):
    """
    Load machine sensor data from a supported file.

    Args:
        file_path: Input file path.
        sheet_name: Excel worksheet name or index.
        max_file_size_bytes: Optional file-size limit in bytes.

    Returns:
        pandas.DataFrame
    """

    path = validate_file(
        file_path,
        max_file_size_bytes=max_file_size_bytes,
    )

    extension = path.suffix.lower()

    if extension == ".csv":
        dataframe = load_csv(path)

    elif extension in {".xlsx", ".xls"}:
        dataframe = load_excel(
            path,
            sheet_name=sheet_name,
        )

    elif extension == ".pdf":
        dataframe = load_pdf(path)

    else:
        raise FileIngestionError(
            f"No loader available for {extension}"
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    if dataframe.columns.duplicated().any():
        duplicate_columns = (
            dataframe.columns[
                dataframe.columns.duplicated(keep=False)
            ]
            .tolist()
        )

        raise FileIngestionError(
            f"Duplicate column names were detected: {duplicate_columns}"
        )

    if len(dataframe.columns) == 0:
        raise FileIngestionError(
            "No columns were found in the uploaded file."
        )

    return dataframe


def get_file_summary(dataframe):
    """Generate a basic summary of loaded machine data."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": list(dataframe.columns),
        "missing_values": int(
            dataframe.isna().sum().sum()
        ),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "data_types": {
            str(column): str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
    }


if __name__ == "__main__":
    print("DYNAMIC FILE LOADER")
    print("=" * 55)

    print(
        "Supported formats:",
        sorted(SUPPORTED_EXTENSIONS),
    )

    print("CSV support: Available through pandas")

    try:
        import openpyxl  # noqa: F401
        print("XLSX support: openpyxl available")
    except ImportError:
        print("XLSX support: openpyxl not installed")

    try:
        import xlrd  # noqa: F401
        print("XLS support: xlrd available")
    except ImportError:
        print("XLS support: xlrd not installed")

    try:
        import pdfplumber  # noqa: F401
        print("PDF support: pdfplumber available")
    except ImportError:
        print("PDF support: pdfplumber not installed")

    print("\nFILE LOADER MODULE READY")
