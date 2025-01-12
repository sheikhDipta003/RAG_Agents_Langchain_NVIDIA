import itertools
import json
import operator
import os
from getpass import getpass

from typing import Annotated, ClassVar, List, Literal, Optional, Tuple, TypedDict, Union

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.documents import Document

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI

from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from typing import ClassVar, List, Literal
import os
try: 
    from .nb_loader import NotebookLoader
except: 
    from nb_loader import NotebookLoader
    
############################################################################

import nbformat

def read_notebook(path):
    """Read a Jupyter notebook from the specified path."""
    with open(path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def create_notebook():
    """Create a new empty Jupyter notebook."""
    return nbformat.v4.new_notebook()

def write_notebook(notebook, path):
    """Write a Jupyter notebook to the specified path."""
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)

def add_cell(notebook, source, cell_type="code"):
    """Add a code cell to a notebook."""
    if cell_type != "output":
        cell_type = f"{cell_type}_cell"
    new_cell_fn = getattr(nbformat.v4, f"new_{cell_type}")
    notebook.cells.append(new_cell_fn(source))

############################################################################

from langchain_unstructured import UnstructuredLoader
from functools import partial

NotebookMdLoader = partial(NotebookLoader, as_markdown=True, cell_delim="[[[CELL-BREAK]]")

class FileLister(BaseModel):

    files: List[str] = Field([], description="Files which you would like to see as context to make your decisions, in order of priority or chronology.")
    workdir: ClassVar[str] = "./temp_nbs"

    @classmethod
    def to_string(cls, files=None, workdir=None, return_in_context=True):
        buffer = "" if return_in_context else []
        workdir = workdir or cls.workdir
        files = files or os.listdir(workdir)
        for file in files:
            try: 
                if file.endswith(".ipynb"):
                    documents = NotebookMdLoader(os.path.join(workdir, file)).load()
                else: 
                    documents = UnstructuredLoader(os.path.join(workdir, file)).load()
            except Exception as e:
                documents = [Document(page_content=f"Exception While Reading File: {e}")]
            if return_in_context:
                contents = "\n\n".join(getattr(doc, "page_content", doc) for doc in documents)
                buffer += f"Contents of `{file}`:<contents>{contents}</contents>\n\n"
            else: 
                buffer += documents
        if return_in_context: 
            buffer = buffer.strip()
        return buffer

    @classmethod
    def get_filenames(cls, workdir=None):
        return os.listdir(workdir or cls.workdir)


class FileReader(BaseModel):
    """
    A Jupyter Notebook Cell, capable of holding markdown and code. 
    Code should go in ```python ...``` markdown scope when illustrating
    and code scope when it should actually be ran. 
    """
    files_to_read: List[Optional[str]] = Field(choices=[])

    def read_files(self):
        return FileLister().to_string(self.files_to_read)

    @classmethod
    def get_instruction(cls):
        parser = PydanticOutputParser(pydantic_object=cls)
        inst = parser.get_format_instructions()
        inst += f"\n\nAvailable Files: {FileLister.get_filenames()}"
        return inst

#####################################################################################
        
class JupyterCell(BaseModel):
    """
    A Jupyter Notebook Cell, capable of holding markdown and code. 
    Code should go in ```python ...``` markdown scope when illustrating
    and code scope when it should actually be ran. 
    """
    source: str
    cell_type: Literal["markdown", "code"]

    
class JupyterChunk(BaseModel):
    """
    List of Jupyter Notebook Cells
    """
    cells: List[JupyterCell]
    filename: Optional[str] = Field(description="filename to save chunks as notebook")

    work_dir: ClassVar[str] = "./temp_nbs"

    def to_json(self):
        return [cell.dict() for cell in self.cells]

    def to_markdown(self, delim="\n", as_list=False):
        md_list = [nb_loader.concatenate_cells_to_md(c) for c in self.to_json()]
        if as_list:
            return md_list
        return delim.join(md_list).strip()

    @classmethod
    def from_cells(cls, cells=[], filename=None):
        dict_cells = [
            (d if isinstance(d, dict) else getattr(d, "dict", lambda: d.__dict__)()) 
            for d in cells 
        ]
        new_cells = [JupyterCell(**cell) for cell in dict_cells]
        return cls(cells=new_cells, filename=filename)
    
    @classmethod
    def _get_nb_path(cls, filename=None, folder=None):
        folder = folder or cls.work_dir
        filename = filename or "temp.ipynb"
        os.makedirs(folder, exist_ok=True)
        if not filename.endswith(".ipynb"):
            filename += ".ipynb"
        return os.path.join(folder, filename)
    
    def save(self, filename=None, folder=None, identity=True):
        out_nb = create_notebook()
        path = self.__class__._get_nb_path(self.filename or filename, folder)
        for cell in self.cells:
            add_cell(out_nb, cell.source, cell.cell_type)
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(out_nb, f)
        if identity: 
            return self

    @classmethod
    def load(cls, filename=None, folder=None):
        path = cls._get_nb_path(filename, folder)
        with open(path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        return cls.from_cells(notebook.cells, filename)

    @classmethod
    def get_instruction(cls):
        parser = PydanticOutputParser(pydantic_object=cls)
        inst = parser.get_format_instructions()
        return inst

#####################################################################################

make_jupyter_notebook_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Develop a series of notebook cells, as a list of jsons."
        " Example code should go into markdown as ```python ...``` content (or other language)."
        " Code that should be ran should go into code cells. Code should never appear in both."
        " Every cell should have comments, discussions, and transitions, and should follow NVIDIA DLI Style."
        "\n\nFORMAT: {struct_format}"
        "\n\nHISTORY: {history}"
    )),
    # MessagesPlaceholder('history'),
    ('user', '{input}'),
])


def make_jupyter_notebook(prompt=make_jupyter_notebook_prompt, llm=None, inputs = {}) -> str:
    """Create a jupyter cell.
    Arguments: 
        directive: Description of notebook requirements/instructions on all the content which should be in it.
        history: Relevant context from the history which includes all relevant code/syntax/details. 
    Returns:
        A jupyter notebook, which will be written to file per filename field.
    """
    parser = PydanticOutputParser(pydantic_object=JupyterChunk)
    cell_llm = llm.with_structured_output(JupyterChunk) 
    chain = prompt | cell_llm | JupyterChunk.save
    output = chain.invoke({"struct_format": parser.get_format_instructions(), **inputs})
    return output


def read_jupyter_notebook(filename: str) -> str:
    """Read in a jupyter notebook. Results will be outputted to a file.
    Arguments:
        filename: ends in .ipynb
    Returns:
        A jupyter notebook.
    """
    return JupyterChunk.load(filename)