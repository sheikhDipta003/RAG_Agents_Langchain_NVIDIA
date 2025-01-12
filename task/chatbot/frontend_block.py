
## NOTE: THIS SERVER IS RUNNING PERPETUALLY FOR THIS COURSE.
## DO NOT CHANGE CODE HERE; INSTEAD, INTERFACE WITH IT VIA USER INTERFACE
## AND BY DEPLOYING ON PORT :9012

import os
import random

from copy import deepcopy
from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA

import gradio as gr
from typing import List

current_dir = Path(__file__).parent

import logging
import traceback

def get_traceback(e):
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(lines)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#####################################################################
## Chain Dictionary

from conv_tool_caller import ConversationalToolCaller
from prompts import context_prompt, agent_prompt, tool_instruction, tool_prompt
from graph import create_graph, agent_fn, tools_fn, set_directive_fn, loop_or_end, START, END, ToolNode
from tools import read_notebook, filenames
from functools import partial
from jupyter_tools import FileLister

chat_llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
tool_llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

# conv_llm = ConversationalToolCaller(
#     tool_instruction=tool_instruction, 
#     tool_prompt=tool_prompt, 
#     llm=chat_llm,
#     toolable_llm = tool_llm,
# ).get_tooled_chain()

# toolset = [read_notebook]
# tooled_agent_fn = partial(agent_fn, llm = conv_llm.bind_tools(toolset), chat_prompt=agent_prompt)
# tooled_tools_fn = partial(tools_fn, tool_node = ToolNode(toolset))

########################################################################################

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
# )

# question = "Give me an interesting code snippet from Notebook 5."
# question = "Show me how the notebook explains diffusion."
# await stream_response(question, app, config, print_stream=False, show_meta=False)

########################################################################################

# async def graph_streamer(history, thread_id=42, **kwargs):
#     messages = {"messages": history}
#     new_config = {"configurable": {"thread_id": thread_id}}
#     async for chunk, meta in app.astream(messages, stream_mode="messages", config=new_config, **kwargs):
#         if meta.get("langgraph_node") == "agent":
#             if chunk.content:
#                 yield chunk

## Necessary Endpoints
chains_dict = {
    # 'Basic' : chat_llm,
    'Context' : (
        context_prompt 
        | RunnablePassthrough(lambda s: print(s))
        | chat_llm
    ),
    # 'Agent' : graph_streamer,
}

model_opts = {
    # "Llama-3.1 8B (local NIM)": {
    #     "model_name": "meta/llama-3.1-8b-instruct",
    #     "base_url": "http://nim:8000/v1",
    #     "toolable": True,
    # },
    "Llama-3.1 8B (remote)": {
        "model_name": "meta/llama-3.1-8b-instruct",
        "base_url": "http://llm_client:9000/v1",
        "toolable": True,
    },
    "Llama-3.1 70B (remote)": {
        "model_name": "meta/llama-3.1-70b-instruct",
        "base_url": "http://llm_client:9000/v1",
        "toolable": True,
    },
    "Llama-3.1 405B (remote)": {
        "model_name": "meta/llama-3.1-405b-instruct",
        "base_url": "http://llm_client:9000/v1",
        "toolable": True,
    },
    # "Llama 3.1 Nemotron 51B (remote)": {
    #     "model_name": "nvidia/llama-3.1-nemotron-51b-instruct",
    #     "base_url": "http://llm_client:9000/v1",
    # },
    "Mistral NeMo 12B (remote)": {
        "model_name": "nv-mistralai/mistral-nemo-12b-instruct",
        "base_url": "http://llm_client:9000/v1",
    },
    "Mistral-Large V2 (remote)": {
        "model_name": "mistralai/mistral-large-2-instruct",
        "base_url": "http://llm_client:9000/v1",
    },
}

#####################################################################
## ChatBot utilities

async def add_message(message, history: list[dict], role="assistant", preface=""):
    new_history = history + [{"role": role, "content": preface}]
    buffer = ""
    try:
        token = ""
        async for chunk in message:
            token = getattr(chunk, 'content', chunk)
            buffer += token
            new_history[-1]["content"] += token.replace("\n", " \n ")
            new_history[-1]["content"] = (
                new_history[-1]["content"]
                    .replace("\n<function=", "\n<b><function")
                    .replace("]</function", "] </function></b><hr/>")
            )
            yield new_history, buffer, False 
            if token.endswith("\n"): 
                new_history[-1]["content"] = new_history[-1]["content"][:-1]
    except Exception as e:
        logger.error(f"Gradio Stream failed:\n{get_traceback(e)}")
        new_history[-1]["content"] += f"...\nGradio Stream failed: {e}"
        yield new_history, buffer, True
    print(chat_llm._client.last_inputs)


async def add_text(history, text):
    history = history + [{"role": "user", "content": text}]
    return history, gr.Textbox(value="", interactive=False)


def set_user_thread(user_state, request: gr.Request):
    ## Give user a new thread to operate on in LangGraph
    global threads
    user_state["thread_id"] = len(threads)
    threads += [user_state["thread_id"]]
    return user_state


async def bot(
    history, 
    chain_key, 
    chat_model, 
    tool_model, 
    file_opts, 
    user_state
):
    chat_llm.model = model_opts.get(chat_model).get("model_name")
    tool_llm.model = model_opts.get(tool_model).get("model_name")
    chat_llm.base_url = model_opts.get(chat_model).get("base_url")
    tool_llm.base_url = model_opts.get(tool_model).get("base_url")
    chain = chains_dict.get(chain_key)
    history = [d for d in history if d.get("content")]
    [d.pop("metadata", "") for d in history]
    # print(FileLister().to_string(files=file_opts, workdir="/notebooks"))
    if chain_key == "Agent":
        msg_stream = chain(history, user_state.get("thread_id"))
    elif chain_key == "Context": 
        inputs = {"messages": history}
        inputs["full_context"] = [] if not file_opts else FileLister().to_string(files=file_opts, workdir="/notebooks")
        msg_stream = chain.astream(inputs)
    else:
        msg_stream = chain.astream(history)
    async for history, buffer, is_error in add_message(msg_stream, history):
        yield history


#####################################################################
## GRADIO EVENT LOOP

# https://github.com/gradio-app/gradio/issues/4001
CSS ="""
.contain { display: flex; flex-direction: column; height:80vh;}
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto; font-width: bold; }
#app-title h1, .prose h1 { font-width: bold; color: #76b900; }
.multi-dropdown label .wrap .wrap-inner { overflow-y: scroll !important; height: 10vh; resize: auto; }
""" 
THEME = gr.themes.Default(primary_hue="green")

threads = []

def get_demo():
    with gr.Blocks(css=CSS, theme=THEME) as demo:

        user_state = gr.State({"thread_id": -1})

        gr.Markdown("# DLI Course Chatbot [BETA]", elem_id="app-title")

        chatbot = gr.Chatbot(
            [{"role": "assistant", "content": "Hello! Welcome to the DLI Chatbot page! How can I help you?"}],
            elem_id="chatbot",
            bubble_full_width=False,
            avatar_images=(None, (os.path.join(os.path.dirname(__file__), "parrot.png"))),
            type="messages"
        )

        with gr.Row():
            with gr.Column(scale=4):    
                txt = gr.Textbox(
                    lines=4,
                    show_label=False,
                    placeholder="Enter text and press SHIFT + ENTER. Refresh window to clear conv history",
                    container=False,
                )
                file_opts = gr.Dropdown(
                    filenames, 
                    value=[],#filenames[1:-1], 
                    multiselect=True, 
                    label="Filebank", 
                    elem_classes=["multi-dropdown"],
                    # info="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor, nisl eget ultricies aliquam, nunc nisl aliquet nunc, eget aliquam nisl nunc vel nisl."
                )
            with gr.Column():
                # chain_btn  = gr.Radio(["Basic", "Context", "Agent"], value="Agent", label="Main Route")
                chain_btn = gr.State("Context")
                chat_llm_name = gr.Dropdown(
                    value="Mistral-Large V2 (remote)",
                    choices=list(model_opts.keys()),
                    scale=1,
                    label="Chatting LLM",
                )
                # tool_llm_name = gr.Dropdown(
                #     value="Mistral-Large V2 (remote)",
                #     choices=[k for k,v in model_opts.items() if v.get("toolable")],
                #     scale=1,
                #     label="Tooling LLM",
                # )
                tool_llm_name = gr.State("Mistral-Large V2 (remote)")

        # Reference: https://www.gradio.app/guides/blocks-and-event-listeners

        # This listener is triggered when the user presses the Enter key while the Textbox is focused.
        txt_msg = (
            # first update the chatbot with the user message immediately. Also, disable the textbox
            txt.submit(              ## On textbox submit (or enter)...
                fn=add_text,            ## Run the add_text function...
                inputs=[chatbot, txt],  ## Pass in the values of chatbot and txt...
                outputs=[chatbot, txt], ## Assign the results to the values of chatbot and txt...
                queue=False             ## And don't use the function as a generator (so no streaming)!
            )
            # then update the chatbot with the bot response (same variable logic)
            .then(bot, [chatbot, chain_btn, chat_llm_name, tool_llm_name, file_opts, user_state], [chatbot])
            ## Then, unblock the textbox by assigning an active status to it
            .then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
        )

        demo.load(set_user_thread, inputs=[user_state], outputs=[user_state])

    return demo

#####################################################################
## Final App Deployment

if __name__ == "__main__":


    demo = get_demo()
    demo.queue()

    logger.warning("Starting FastAPI app")

    app = FastAPI()
    app.mount("/imgs", StaticFiles(directory="/notebooks/imgs"), name="images")

    ## Allows images to be accessible via /file=imgs/...
    gr.set_static_paths(paths=["imgs"])
    app = gr.mount_gradio_app(app, demo, '/')

    @app.route("/health")
    async def health():
        return {"success": True}, 200
