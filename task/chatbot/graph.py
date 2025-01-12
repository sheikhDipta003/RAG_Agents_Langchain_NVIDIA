from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
import operator
import uuid
import datetime

##################################################################

class State(TypedDict):
    ## Dictates what kind of buffer the agent nodes can write to to pass information
    ## This one says "nodes can write to messages buffer, writing is equivalent to adding a message"
    ## NOTE: To override a message, you can add a message with the target message's ID. 
    ## NOTE: To delete a message, you can add a RemoveMessage with the target's message ID. 
    messages: Annotated[list[AnyMessage], add_messages]
    directives: Annotated[list[AnyMessage], add_messages]

def create_graph(nodes, edges, conditional_edges=[], state=State, thread_id="42"):
    graph = StateGraph(state)
    [graph.add_node(*node) for node in nodes]
    [graph.add_edge(*edge) for edge in edges]
    [graph.add_conditional_edges(*cedge) for cedge in conditional_edges]
    
    ## The checkpointer lets the graph persist its state
    ## Thread used to select buffer / memory compartment / etc to operate on 
    # config = {"configurable": {"thread_id": thread_id}}
    # memory = MemorySaver()
    app = graph.compile(
        # checkpointer=memory
    )
    memory = None
    config = {}

    return app, memory, config

#################################################################

from langchain.tools import tool
from typing import Literal
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from jupyter_tools import FileLister
from functools import partial


async def set_directive_fn(state, config: RunnableConfig):
    return {"directives": [state.get("messages")[-1]]}
    

async def agent_fn(state, config: RunnableConfig, llm, chat_prompt, **kwargs):
    chain = chat_prompt | llm.bind(config=config)
    response = await chain.ainvoke(state)
    ## This invocation makes a new message, so this return is an appending of a new message
    return {"messages": [response]}

    
async def tools_fn(state, config: RunnableConfig, tool_node = (lambda x: x), **kwargs):
    last_msg = state.get("messages")[-1]
    if last_msg.tool_calls:
        results = tool_node.invoke({"messages": [last_msg]})["messages"]
        for result in results:
            last_msg.content += f"\n<RESULT>{result.content}</RESULT>"

    directive = state.get("directives")[-1].content
    new_msgs = [last_msg, (
        "human", f"Great! Now continue responding to the original user directive: {directive}."
            " You've executed at least one tool, so continue your thought process. DO NOT redo any past processes."
    )]
    return {"messages": new_msgs}

################################################################################################

def loop_or_end(state: Literal["loop", "end"], config: RunnableConfig):
    ## Return the state to route to based on whether a tool is called
    return "loop" if state.get("messages")[-1].tool_calls else "end"

# app, memory, config = create_graph(
#     nodes = [
#         ("enter", set_directive_fn), 
#         ("agent", agent_fn), 
#         ("tools", tools_fn), 
#         # ("react", react_fn), 
#     ],
#     edges = [
#         (START, "enter"),
#         ("enter", "agent"),
#         ("tools", "agent"),
#         # (START, "react"), ("react", END),
#     ],
#     conditional_edges = [
#         ("agent", loop_or_end, {"loop": "tools", "end": END})
#     ]
# )

################################################################################################

async def stream_response(new_messages, app, config, yield_meta, silenced_nodes=[]):
    buffers = {}
    new_messages = [("human", new_messages)] if isinstance(new_messages, str) else new_messages
    new_messages = {"messages": new_messages} if isinstance(new_messages, dict) else new_messages
    async for msg, meta in app.astream(new_messages, stream_mode="messages", config=config):
        if meta.get("langgraph_node") in silenced_nodes: continue
        if msg.id not in buffers:
            delim = "*" * 84
            print(f"\n\n{delim}\n** Found {msg.__class__.__name__} with id {msg.id}\n{delim}")
            if show_meta: print(f"{meta}\n{delim}")
        buffers[msg.id] = msg if not buffers.get(msg.id) else (buffers.get(msg.id) + msg)
        if print_raw: 
            print(repr(msg) if not truncate else str(repr(msg))[:truncate])
        elif not isinstance(msg, ToolMessage):
            print(msg.content, end="")
    
# async def print_response_stream(
#     new_messages,
#     app, config,
#     print_raw=False,  ## If true, print messages from buffer. Otherwise, just prints tokens. 
#     truncate=200,        ## Maximum length to give to each streamed value
#     show_meta=True,      ## Whether to show message metadata i.e. buffer, producing node, etc.
#     silenced_nodes=[]    ## Nodes whos' results you don't want to see
# ):
#     buffers = {}
#     new_messages = [("human", new_messages)] if isinstance(new_messages, str) else new_messages
#     new_messages = {"messages": new_messages} if isinstance(new_messages, dict) else new_messages
#     async for msg, meta in app.astream(new_messages, stream_mode="messages", config=config):
#         if meta.get("langgraph_node") in silenced_nodes: continue
#         if msg.id not in buffers:
#             delim = "*" * 84
#             print(f"\n\n{delim}\n** Found {msg.__class__.__name__} with id {msg.id}\n{delim}")
#             if show_meta: print(f"{meta}\n{delim}")
#         buffers[msg.id] = msg if not buffers.get(msg.id) else (buffers.get(msg.id) + msg)
#         if print_raw: 
#             print(repr(msg) if not truncate else str(repr(msg))[:truncate])
#         elif not isinstance(msg, ToolMessage):
#             print(msg.content, end="")

################################################################################################

# await stream_response(
#     input("[Human]"), 
#     app, config, 
#     print_stream=True
# )

# from functools import partial
# from typing import Literal
# from jupyter_tools import FileLister

# @tool
# def read_notebook(
#     filename: str, 
# ) -> str:
#     """Displays a file to yourself and the end-user. These files are long, so only use it as a last resort."""
#     return FileLister().to_string(files=[filename], workdir=".")

# ## Advanced Note: The schema can be strategically modified to tell the server how to grammar enforce
# ## In this case, specifying the finite options for the files. 
# ## To discover this, try type-hinting filename: Literal["file1", "file2"] and printing schema
# read_notebook.args_schema.schema()["properties"]["filename"]["enum"] = filenames

################################################################################################

# toolset = [read_notebook]
# tooled_agent_fn = partial(agent_fn, llm = conv_llm.bind_tools(toolset))
# tooled_tools_fn = partial(tools_fn, tool_node = ToolNode(toolset))

################################################################################################

# app, memory, config = create_graph(
#     nodes = [
#         ("enter", set_directive_fn), 
#         ("agent", tooled_agent_fn), 
#         ("tools", tooled_tools_fn), 
#     ],
#     edges = [
#         (START, "enter"),
#         ("enter", "agent"),
#         ("tools", "agent"),
#     ],
#     conditional_edges = [
#         ("agent", loop_or_end, {"loop": "tools", "end": END})
#     ],
#     plot=False,
# )

# question = "Give me an interesting code snippet from Notebook 5."
# question = "Show me how the notebook explains diffusion."
# await stream_response(question, app, config, print_stream=False, show_meta=False)