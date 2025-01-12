from functools import partial
from typing import Literal
from langchain_core.tools import tool
from jupyter_tools import FileLister

####################################################################
## Chatbot is intentionally grounded on pre-constructed json 

import json

with open('notebook_chunks.json', 'r') as fp:
    nbsummary = json.load(fp)

filenames = nbsummary.get("filenames")
outlines = "\n\n".join([v.get("outline") for k,v in nbsummary.items() if isinstance(v, dict)])

####################################################################

@tool
def read_notebook(
    filename: str, 
) -> str:
    """Displays a file to yourself and the end-user. These files are long, so only use it as a last resort."""
    return FileLister().to_string(files=[filename], workdir="..")

## Advanced Note: The schema can be strategically modified to tell the server how to grammar enforce
## In this case, specifying the finite options for the files. 
## To discover this, try type-hinting filename: Literal["file1", "file2"] and printing schema
read_notebook.args_schema.schema()["properties"]["filename"]["enum"] = filenames
