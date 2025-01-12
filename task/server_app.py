# https://python.langchain.com/docs/langserve#server
from fastapi import FastAPI
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langserve import add_routes

## May be useful later
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.document_transformers import LongContextReorder
from functools import partial
from operator import itemgetter

from langchain_community.vectorstores import FAISS

## TODO: Make sure to pick your LLM and do your prompt engineering as necessary for the final assessment
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

## Make sure you have docstore_index.tgz in your working directory
docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
docs = list(docstore.docstore._dict.values())

def format_chunk(doc):
    return (
        f"Paper: {doc.metadata.get('Title', 'unknown')}"
        f"\n\nSummary: {doc.metadata.get('Summary', 'unknown')}"
        f"\n\nPage Body: {doc.page_content}"
    )

## This printout just confirms that your store has been retrieved
print(f"Constructed aggregate docstore with {len(docstore.docstore._dict)} chunks")
# print(f"Sample Chunk:")
# print(format_chunk(docs[len(docs)//2]))

# The code confirms that the vector store has been successfully loaded and displays:
# The total number of document chunks.
# A sample document chunk with its title, summary, and content.

from functools import partial
from operator import itemgetter

#####################################################################
llm = instruct_llm | StrOutputParser()

#####################################################################

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name: out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

chat_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked you a question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational)"
    "\n\nUser Question: {input}"
)

def output_puller(inputs):
    """"Output generator. Useful if your chain returns a dictionary with key 'output'"""
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')

#####################################################################
## TODO: Pull in your desired RAG Chain. Memory not necessary

## Chain 1 Specs: "Hello World" -> retrieval_chain 
##   -> {'input': <str>, 'context' : <str>}
long_reorder = RunnableLambda(LongContextReorder().transform_documents)  ## GIVEN
context_getter = (
    lambda x: long_reorder.invoke(
        docstore.similarity_search(x, k=3)
    )
)
retrieval_chain = RunnableLambda(lambda x: context_getter(x))  # Return only the context value

## Chain 2 Specs: retrieval_chain -> generator_chain 
##   -> {"output" : <str>, ...} -> output_puller
generator_chain = (
    chat_prompt
    | llm
)
# generator_chain = {'output' : generator_chain} | RunnableLambda(output_puller)  ## GIVEN

## END TODO
#####################################################################


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

## PRE-ASSESSMENT: Run as-is and see the basic chain in action

add_routes(
    app,
    instruct_llm,
    path="/basic_chat",
)

## ASSESSMENT TODO: Implement these components as appropriate

add_routes(
    app,
    generator_chain,
    path="/generator",
)

add_routes(
    app,
    retrieval_chain,
    path="/retriever",
)

## Might be encountered if this were for a standalone python file...
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)
