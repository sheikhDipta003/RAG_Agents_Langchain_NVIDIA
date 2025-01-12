from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_core.messages.utils import message_chunk_to_message, _msg_to_chunk, _convert_to_message
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableAssign, RunnablePassthrough, RunnableLambda, RunnableGenerator
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing import List, Any, Optional, ClassVar
from langchain_core.utils.function_calling import convert_to_openai_tool
from langserve import RemoteRunnable
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config

import contextlib

# @contextlib.contextmanager
# def set_values(variables: list, keys: list, values: list):
#     originals = [getattr(val, key) for val, key in zip(variables, keys)]
#     try:     # Temporarily update the values by reference
#         [setattr(var, key, val) for var, key, val in zip(variables, keys, values)]
#         yield
#     finally: # Always restore the original value, no matter what happens
#         [setattr(var, key, val) for var, key, val in zip(variables, keys, originals)]

import asyncio
from typing import AsyncGenerator, Generator, TypeVar

T = TypeVar('T')

def agen_to_gen(agen: AsyncGenerator[T, None]) -> Generator[T, None, None]:
    loop = asyncio.get_event_loop()
    sentinel = object()
    while True:
        chunk = loop.run_until_complete(agen.__anext__())
        if chunk is sentinel:
            break
        yield chunk

async def gen_to_agen(gen: Generator[T, None, None]) -> AsyncGenerator[T, None]:
    sentinel = object()
    while True:
        chunk = await asyncio.to_thread(next, gen, sentinel)
        if chunk is sentinel:
            break
        yield chunk

class ConversationalToolCaller(BaseModel):
    
    tool_instruction: str = Field(
        "You have access to the tools listed in the toolbank. Use tools only within the \n<function></function> tags."
        " Select tools to handle uncertain, imprecise, or complex computations that an LLM would find it hard to answer."
        " You can only call one tool at a time, and the tool cannot accept complex multi-step inputs."
        "\n\n<toolbank>{toolbank}</toolbank>\n"
        "Examples (WITH HYPOTHETICAL TOOLS):"
        "\nSure, let me call the tool in question.\n<function=\"foo\">[\"input\": \"hello world\"]</function>"
        "\nSure, first, I need to calculate the expression of 5 + 10\n<function=\"calculator\">[\"expression\": \"5 + 10\"]</function>"
        "\nSure! Let me look up the weather in Tokyo\n<function=\"weather\">[\"location\"=\"Tokyo\"])</function>"
    )
    
    tool_prompt: str = Field(
        "You are an expert at selecting tools to answer questions. Consider the context of the problem,"
        " what has already been solved, and what the immediate next step to solve the problem should be."
        " Do not predict any arguments which are not present in the context; if there's any ambiguity, use no_tool."
        "\n\n<toolbank>{toolbank}</toolbank>\n"
        "\n\nSchema Instructions: The output should be formatted as a JSON instance that conforms to the JSON schema."
        "\n\nExamples (WITH HYPOTHETICAL TOOLS):"
        "\n<function=\"search\">[\"query\": \"current events in Japan\"]</function>"
        "\n<function=\"translation\">[\"text\": \"Hello, how are you?\", \"language\": \"French\"]</function>"
        "\n<function=\"calculator\">[\"expression\": \"5 + 10\"]</function>"
    )

    tools: Optional[List[Any]] = Field([])
    llm: Optional[Any] = Field()
    toolable_llm: Optional[Any] = Field()
    include_null_tool: bool = Field(True)

    @tool
    def no_tool(self) -> str:
        """Null tool; says no tool should be used"""
        return "No Tool Selected"

    def get_tools(self, include_null_tool = None, tools = None):
        tools = (tools or self.tools)[:]
        for tool in tools:
            if isinstance(tool, dict):
                return []
        include_null_tool = include_null_tool if include_null_tool is not None else self.include_null_tool
        if include_null_tool:
            tools += [self.no_tool]
        return tools

    def get_tool_signatures(self, include_null_tool = None, tools = None):
        tools = (tools or self.tools)[:]
        schemas = []
        for tool in tools:
            if isinstance(tool, dict):
                schemas += [tool]
            else:
                schemas += [convert_to_openai_tool(tool)]
        include_null_tool = include_null_tool if include_null_tool is not None else self.include_null_tool
        if include_null_tool:
            schemas += [convert_to_openai_tool(self.no_tool)]
        return schemas
    
    def get_tool_node(self, raw_node: bool = False, include_null_tool = None, tools = None):
        ## The thing that executes your tooling requests
        tools = self.get_tools(include_null_tool, tools)
        if raw_node:
            return ToolNode(tools)
        else: 
            return (
                RunnablePassthrough()
                | {"messages": lambda x: [x]} 
                | ToolNode(tools) 
                | RunnableLambda(lambda x: x.get("messages"))
            )


    def get_toolbank(self, tools=None):
        ## The context that tells your LLMs what tools they have access to
        toolsig = self.get_tool_signatures(tools=tools)
        return  f"{str(toolsig).replace('{', '[').replace('}', ']')}"

    def get_tool_instruction(self, fill_toolbank: bool = True, tools: Optional[list] = None):
        ## The instructions for what tools can be invoked inline and how
        if fill_toolbank:
            return self.tool_instruction.format(toolbank = self.get_toolbank(tools=tools))
        return self.tool_instruction

    def get_tool_prompt(self, fill_toolbank: bool = True, tools: Optional[list] = None):
        ## The instructions for how the tool caller should generate its calls based on the conversation
        if fill_toolbank:
            return self.tool_prompt.format(toolbank = self.get_toolbank(tools=tools))
        return self.tool_prompt

    ################################################################################################
    
    def get_tooled_chain(
        self, 
        llm: Optional[BaseChatModel] = None, 
        toolable_llm: Optional[BaseChatModel] = None, 
    ):
    
        def token_generator(
            state, 
            tools=None, 
            llm=llm,
            toolable_llm=toolable_llm,
            config: RunnableConfig = None,
        ):  
            tools = tools or self.tools
            llm = llm or self.llm
            toolable_llm = toolable_llm or self.toolable_llm or llm
            assert llm, "Please specify your llm on caller construction"

            if isinstance(state, str):
                state = {"messages": state}
            
            if isinstance(state, dict) and state.get("messages"):
                if isinstance(state.get("messages"), (list, tuple)):
                    messages = [_convert_to_message(msg) for msg in state.get("messages")]
                else: 
                    messages = [_convert_to_message(state.get("messages"))]
                state = ChatPromptValue(messages = messages)                    
                config = config or state.get("config")

            ## PART 1: Engage in discussions about which tools to call
            if tools:
                if state.messages and state.messages[0].type == "system":
                    state.messages[0].content += "\n\n" + self.get_tool_instruction(tools=tools)
                else: 
                    state.messages = [SystemMessage(content=self.get_tool_instruction(tools=tools))] + state.messages

            ## Stream tokens from the first chain as they get generated
            pre_tool_msg = None
            for token in llm.stream(state, config=config, stop="</function>"):
                pre_tool_msg = token if not pre_tool_msg else pre_tool_msg + token
                yield token

            ender = "]</function>"
            if "<function" in pre_tool_msg.content and "</function>" not in pre_tool_msg.content:
                token.content = ""
                for i in range(len(ender)):
                    if pre_tool_msg.content.strip().endswith(ender[:i+1]): 
                        token.content = ender[i+1:]
                        break
                if token.content:
                    pre_tool_msg += token
                    yield token
                
            ## If there are no possible tools to call, that's that
            if not tools: return
            assert pre_tool_msg.content, "Empty Content From First Section"
            
            if "</function>" in pre_tool_msg.content:

                ## PART 2: Actually predict the tool call via guided decoding based off the last message
                tool_prompt = ChatPromptTemplate.from_messages([
                    ("system", self.get_tool_prompt(tools = tools)),
                    ("placeholder", "{messages}"),
                ])
                
                bound_llm = toolable_llm.bind_tools(self.get_tools(tools=tools), tool_choice="any")
                tool_chain2 = tool_prompt | bound_llm
                tool_chain2_msgs = {"messages": [message_chunk_to_message(pre_tool_msg)]}

                ## Advanced concept; Streaming enforced at RunnableConfig context scope. Disable to force non-streaming
                old_config = var_child_runnable_config.get()
                try: 
                    var_child_runnable_config.set({})
                    tooled_msg = _msg_to_chunk(tool_chain2.invoke(tool_chain2_msgs))
                finally: 
                    var_child_runnable_config.set(old_config)

                if any(v["name"] != "no_tool" for v in tooled_msg.tool_calls):
                    yield RunnablePassthrough().invoke(tooled_msg)

        class FakeToolBind(Runnable):
            
            bind_kwargs: ClassVar[dict] = {}

            def __init__(self, generator, *args, **kwargs):
                self.generator = generator

            def invoke(self, *args, **kwargs): 
                buffer = None
                for token in self.generator(args[0], **kwargs):
                    buffer = token if not buffer else buffer + token
                return message_chunk_to_message(buffer)
            
            def stream(self, *args, **kwargs): 
                for token in self.generator(args[0], **kwargs):
                    yield token

            async def ainvoke(self, *args, **kwargs): 
                buffer = None
                async for token in gen_to_agen(self.generator(args[0], **kwargs)):
                    buffer = token if not buffer else buffer + token
                return message_chunk_to_message(buffer)
            
            async def astream(self, *args, **kwargs): 
                async for token in gen_to_agen(self.generator(args[0], **kwargs)):
                    yield token
            
            def bind_tools(self, tools, **kwargs):
                self.bind_kwargs["tools"] = tools
                return self.bind(**self.bind_kwargs)

        
        class InputSchema(BaseModel):
            messages: Any
            tools: Optional[List]
            llm: Optional[Any]
        
        output = FakeToolBind(token_generator).with_types(input_type=InputSchema)
        return output
        

class TooledRemoteRunnable(RemoteRunnable):

    def bind_tools(self, tools, **kwargs):

        class InputSchema(BaseModel):
            messages: Any
            tools: Optional[List]
            llm: Optional[Any]

        tools = [convert_to_openai_tool(tool) for tool in tools]
        return (
            RunnableLambda(lambda s: {"messages": s, "tools": tools}) 
            | self.with_types(input_type=InputSchema)
        )


if __name__ == "__main__":

    from fastapi import FastAPI
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
    from langchain_openai import ChatOpenAI
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
    
    import uvicorn
    from langchain_community.vectorstores import FAISS
    import os

    model_name = os.environ.get("NVIDIA_MODEL_NAME", "meta/llama-3.1-8b-instruct")
    model_path = os.environ.get("NVIDIA_BASE_URL", "http://llm_client:9000/v1")

    ## TODO: Make sure to pick your LLM and do your prompt engineering as necessary for the final assessment
    llm = ChatOpenAI(model=model_name, base_url=model_path, api_key="none")

    tool_caller = ConversationalToolCaller(llm=llm).get_tooled_chain()

    app = FastAPI(
      title="LangChain Server",
      version="1.0",
      description="A simple api server using Langchain's Runnable interfaces",
    )
    
    add_routes(
        app,
        tool_caller,
        path="/convtools",
    )

    uvicorn.run(app, host="0.0.0.0", port=9012)