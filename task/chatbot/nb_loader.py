"""Loads .ipynb notebook files."""
import json
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


def concatenate_cells(
    cell: dict,
    include_outputs: bool,
    max_output_length: int,
    cell_delim: str,
    traceback: bool,
) -> str:
    """Combine cells information in a readable format ready to be used.

    Args:
        cell: A dictionary
        include_outputs: Whether to include the outputs of the cell.
        max_output_length: Maximum length of the output to be displayed.
        traceback: Whether to return a traceback of the error.

    Returns:
        A string with the cell information.

    """
    cell_type = cell["cell_type"]
    source = cell["source"]
    output = cell["outputs"]
    
    if include_outputs and cell_type == "code" and output:
        if "ename" in output[0].keys():
            error_name = output[0]["ename"]
            error_value = output[0]["evalue"]
            if traceback:
                traceback = output[0]["traceback"]
                return (
                    f"'{cell_type}' cell: '{source}'\n, gives error '{error_name}',"
                    f" with description '{error_value}'\n"
                    f"and traceback '{traceback}'\n\n"
                ) + cell_delim
            else:
                return (
                    f"'{cell_type}' cell: '{source}'\n, gives error '{error_name}',"
                    f"with description '{error_value}'\n\n"
                ) + cell_delim
        elif output[0]["output_type"] == "stream":
            output = output[0]["text"]
            min_output = min(max_output_length, len(output))
            return (
                f"'{cell_type}' cell: '{source}'\n with "
                f"output: '{output[:min_output]}'\n\n"
            ) + cell_delim
    else:
        return f"'{cell_type}' cell: '{source}'\n\n" + cell_delim

    return ""


def concatenate_cells_to_md(
    cell: dict,
    include_outputs: bool = False,
    max_output_length: int = 10,
    cell_delim: str = "",
    traceback: bool = False,
) -> str:
    """
    Convert cell information into a markdown format.

    Args:
        cell: A dictionary representing the Jupyter notebook cell.
        include_outputs: Whether to include the outputs of the cell.
        max_output_length: Maximum length of the output to be displayed.
        traceback: Whether to return a traceback of the error.

    Returns:
        A markdown-formatted string with the cell information.
    """
    cell_type = cell["cell_type"]
    source = ''.join(cell.get("source", []))
    outputs = cell.get("outputs", [])
    
    if cell_type == "code":
        markdown_output = f"\n```python\n{source}\n```\n" 
    else: 
        markdown_output = f"\n{source}\n"

    if include_outputs and cell_type == "code" and output:
        for output in outputs: 
            if "ename" in output:
                error_name = output["ename"]
                error_value = output["evalue"]
                if traceback and "traceback" in output:
                    error_traceback = ''.join(output["traceback"])
                    error_section = (
                        f"#### Error\n"
                        f"- **Name**: {error_name}\n"
                        f"- **Description**: {error_value}\n"
                        f"- **Traceback**:\n```\n{error_traceback}\n```\n"
                    )
                else:
                    error_section = (
                        f"#### Error\n"
                        f"- **Name**: {error_name}\n"
                        f"- **Description**: {error_value}\n"
                    )
                markdown_output += error_section
            elif output["output_type"] == "stream":
                text_output = ''.join(output.get("text", []))
                min_output = min(max_output_length, len(text_output))
                markdown_output += f"\n```output\n{text_output[:min_output]}\n```\n"
    
    return markdown_output + cell_delim


def remove_newlines(x: Any) -> Any:
    """Recursively remove newlines, no matter the data structure they are stored in."""
    import pandas as pd

    if isinstance(x, str):
        return x.replace("\n", "")
    elif isinstance(x, list):
        return [remove_newlines(elem) for elem in x]
    elif isinstance(x, pd.DataFrame):
        return x.applymap(remove_newlines)
    else:
        return x


class NotebookLoader(BaseLoader):
    """Load `Jupyter notebook` (.ipynb) files."""

    def __init__(
        self,
        path: str,
        include_outputs: bool = False,
        max_output_length: int = 10,
        remove_newline: bool = False,
        as_markdown: bool = False,
        cell_delim: str = "",
        traceback: bool = False,
    ):
        """Initialize with a path.

        Args:
            path: The path to load the notebook from.
            include_outputs: Whether to include the outputs of the cell.
                Defaults to False.
            max_output_length: Maximum length of the output to be displayed.
                Defaults to 10.
            remove_newline: Whether to remove newlines from the notebook.
                Defaults to False.
            traceback: Whether to return a traceback of the error.
                Defaults to False.
        """
        self.file_path = path
        self.include_outputs = include_outputs
        self.max_output_length = max_output_length
        self.remove_newline = remove_newline
        self.traceback = traceback
        self.cell_delim = cell_delim
        self.as_markdown = as_markdown

    def load(
        self,
    ) -> List[Document]:
        """Load documents."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is needed for Notebook Loader, "
                "please install with `pip install pandas`"
            )
        p = Path(self.file_path)

        with open(p, encoding="utf8") as f:
            d = json.load(f)

        data = pd.json_normalize(d["cells"])
        cell_metas = ["cell_type", "source", "outputs"]
        for meta in cell_metas:
            if meta not in data:
                data = data.assign(**{meta: ""})
        filtered_data = data[["cell_type", "source", "outputs"]]
        if self.remove_newline:
            filtered_data = filtered_data.applymap(remove_newlines)
            
        concat_fn = concatenate_cells_to_md if self.as_markdown else concatenate_cells
        concat_args = {
            "include_outputs": self.include_outputs, 
            "max_output_length": self.max_output_length, 
            "traceback": self.traceback,
            "cell_delim": self.cell_delim,
        }
        concat_pr = lambda x: concat_fn(x, **concat_args)

        text = filtered_data.apply(concat_pr, axis=1).str.cat(sep=" ")

        metadata = {"source": str(p)}

        return [Document(page_content=text, metadata=metadata)]

