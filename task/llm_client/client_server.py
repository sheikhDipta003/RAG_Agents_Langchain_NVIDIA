from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
import httpx
import os
from pydantic import BaseModel, Field
from typing import Dict, List
from collections import defaultdict
from functools import partial
import asyncio

import jinja2
import json
import time
import requests
import traceback

app = FastAPI()


###################################################################################
## Request Utilities

def get_missing_model_msg(model):
    return {"error": {
        "message": f"The model \'{model}\' does not exist", 
        "type": "invalid_request_error", 
        "param": "model", 
        "code": "model_not_found"}}

def get_timeout_msg(path):
    return {"error": {
        "message": f"Timeout occurred when trying to call \'{path}\'", 
        "type": "request_timeout_error", 
        "param": "path", 
        "code": "request_timed_out"}}


Client = partial(httpx.AsyncClient, timeout=httpx.Timeout(60))


###################################################################################
## Possible Routes For Model Discovery/Authentication


class RouteBase(BaseModel):
    base_url: str
    key: str = Field("")
    mapper: dict = Field({})
    models: dict = Field({})
    exclude_list: list = Field([])

    def map_ext(self, ext):
        ext = ext if not ext.startswith("/") else ext[1:]
        ext = ext if not ext.endswith("/") else ext[:-1]
        ext = ext if not ext.startswith("v1/") else ext[3:]
        return self.mapper.get(ext, ext)

    def url(self, extension, drop_base=False, model=None):
        ext = self.map_ext(extension)
        return ext if drop_base else os.path.join(self.base_url, ext)

    def headers(self, stream=False):
        headers = {"content-type": "application/json"}
        if stream:
            headers["accept"] = "text/event-stream"
        if self.key: 
            headers["authorization"] = f"Bearer {os.environ.get(self.key)}"
        return headers

    def content(self, content, extension, model=None):
        ext = self.url(extension, drop_base=True, model=model)
        if "chat" in ext:
            if "prompt" in content:
                raise ValueError("Cannot feed prompt into chat completions endpoint")
        else: 
            if "messages" in content:
                raise ValueError("Cannot feed messages into completions endpoint")
        return content


class RouteOpenAPI(RouteBase):

    async def get_models(self, headers={}, refresh=False):
        try: 
            async with Client() as client:
                response = await client.get(
                    self.url("models"),
                    headers={**self.headers(), **headers},
                )
        except httpx.TimeoutException as e:
            raise HTTPException(status_code=408, detail=get_timeout_msg(self.url("models")))
        try: 
            message = response.json()
            model_list = message.get('data', [])
            self.models = {m.get("id"): m for m in model_list if m.get("id") not in self.exclude_list}
            return model_list
        except Exception as e: 
            return response

    async def has_model(self, model_name, ext, force_refresh=False):
        if not self.models or force_refresh:
            await self.get_models()
        return model_name in self.models

    async def postprocess(self, response, extension, timeout=10, headers={}):
        return response


###################################################################################
## Global state to store API keys, endpoint URL, and models
app_state = {
    "default_model": "mistralai/mixtral-8x7b-instruct-v0.1",
    "endpoints": [
        RouteOpenAPI(
            base_url="https://integrate.api.nvidia.com/v1",
            key="NVIDIA_API_KEY",
            exclude_list=[
                "adept/fuyu-8b",
                "baai/bge-m3",
                "bigcode/starcoder2-15b",
                "bigcode/starcoder2-7b",
                "google/deplot",
                "google/gemma-2b",
                "google/paligemma",
                "liuhaotian/llava-v1.6-34b",
                "liuhaotian/llava-v1.6-mistral-7b",
                "microsoft/kosmos-2",
                "mistralai/mixtral-8x22b-v0.1",
                "nvidia/nemotron-4-340b-reward",
                "yentinglin/llama-3-taiwan-70b-instruct",
            ],
        ),
        RouteOpenAPI(
            base_url="https://api.openai.com/v1",
            key="OPENAI_API_KEY"
        ),
    ],
    "model_map": {},
    "valid_keys": ["OPENAI_API_KEY", "NVIDIA_API_KEY"]
}
app_state["initial_keys"] = {
    k: os.environ.get(k) 
    for k in app_state.get("valid_keys")
}


##########################################################################
## Simple Endpoints/Key Management


@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running."""
    return True


@app.get("/hello")
async def read_root():
    """Returns a simple greeting, useful for initial testing."""
    return {"Hello": "World"}


@app.get("/")
async def list_endpoints():
    """Lists all available endpoints with their descriptions."""
    exclude_list = ["/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"]
    return {
        route.path: route.description
        for route in app.routes if route.path not in exclude_list
    }


@app.post("/set_keys")
async def set_key(data: Dict[str, str]):
    """Update API keys for application"""
    try: 
        global app_state
        new_data = {k:v for k,v in data.items() if k in app_state.get("valid_keys", [])}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    os.environ.update(new_data)
    return {"result" : "Keys set successfully"}


@app.post("/revert")
async def set_key(data: Dict[str, str]):
    """Reverts API key to initial state"""
    global app_state
    os.environ.update(app_state.get("initial_keys", {}))
    return {"result" : "Keys reset successfully"}


# @app.get("/get_keys")
# async def get_key():
#     """Returns the current relevant API keys"""
#     global app_state
#     return {k:v for k,v in os.environ.items() if k in app_state.get("valid_keys", [])


###################################################################################
## OpenAPI-Style Endpoints


async def populate_model_list():
    try: 
        ## Schedule all the model discovery queries to run in tandem
        endpoints = app_state.get("endpoints")
        tasks = [ep.get_models() for ep in endpoints]
        results = await asyncio.gather(*tasks)
        
        ## Populate per-endpoint models and aggregate list of all models
        model_lists = []
        out_responses = []
        for ep, result in zip(endpoints, results):
            if isinstance(result, list):
                model_lists += [result]
                app_state["model_map"].update({m.get("id"): ep for m in model_lists[-1]})
            else: 
                print("Response Encountered:", msg or response.__dict__)
                out_responses += [response]
        
        ## If any responses were lodged (none should have) report the first one
        if out_responses:
            return out_responses[0]
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
@app.get("/v1/models/{model:path}")
async def list_models(model=""):
    """Returns the listing of available models (or a single model)"""
    try:
        response = await populate_model_list()
        ## TODO: What happens when a population request fails? Log it and move on?
        if not model:
            model_lists = [list(ep.models.values()) for ep in app_state.get("endpoints")]
            return {"object": "list", "data": sum(model_lists, [])}
        else:
            model_ep = app_state["model_map"].get(model)
            if model_ep:
                return model_ep.models.get(model)

        ## Base case; a requested model does not exist
        missing_msg = get_missing_model_msg(model)
        raise HTTPException(detail=missing_msg, status_code=404)
        
    except Exception as e:
        traceback.print_exc()
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))

##########################################################################
## Inference

async def create_completion_base(request: Request, extension: str):
    """Forwards chat completion requests to OpenAPI endpoint"""
    try: 
        content = await request.body()
        content = json.loads(content.decode())

        model = content.get("model")
        stream = content.get("stream")

        ## Slight processing of headers for passthrough
        headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in ["host", "content-length"]
        }

        endpoint_opts = [
            ep for ep in app_state.get("endpoints", []) 
            if await ep.has_model(model, extension)
        ]

        if endpoint_opts:
            endpoint = endpoint_opts[0]
        else:
            missing_msg = get_missing_model_msg(model)
            raise HTTPException(detail=missing_msg, status_code=404)

        call_kws = {
            "url": endpoint.url(extension),
            "content": json.dumps(content).encode(),
            "headers": {**headers, **endpoint.headers(stream)}, 
        }

        ############################################################
        ## Simple Use Case: Non-streaming (w/ early termination)

        if not stream:
            try: 
                async with Client() as client:
                    response = await client.post(**call_kws)
            except httpx.TimeoutException as e:
                raise HTTPException(status_code=408, detail=get_timeout_msg(call_kws.get("url")))

            filtered_headers = {
                key: value for key, value in response.headers.items() 
                if key.lower() not in ["content-length", "content-encoding", "transfer-encoding"]
            }

            return Response(content=response.content, status_code=response.status_code, headers=filtered_headers)

        ############################################################
        ## Simple Use Case: Streaming

        ## Create a generator to keep querying the response endpoint after initial response
        ## NOTE: This is a weird way for keeping stream client open both for a potential
        ##  initial exception raise and also as an argument to the StreamingResponse return.
        async def respond_and_stream():
            try: 
                async with Client().stream("POST", **call_kws) as response:
                    yield response
                    ## This stage only gets invoked it the response is valid + streaming is enabled
                    agen = response.aiter_bytes()
                    async for cbytes in agen:
                        yield cbytes
            except httpx.TimeoutException as e:
                raise HTTPException(status_code=408, detail=get_timeout_msg(call_kws.get("url")))

        ## Create response generator and process initial response
        agen = respond_and_stream()
        response = await agen.__anext__()

        if response.status_code != 200:
    
            response_bytes = await response.aread()
            content = response_bytes
            filtered_headers = {
                key: value for key, value in response.headers.items() 
                if key.lower() not in ["content-length", "content-encoding", "transfer-encoding"]
            }
            return Response(content=content, status_code=response.status_code, headers=filtered_headers)

        else: 

            return StreamingResponse(agen, media_type='text/event-stream')

    except Exception as e:
        traceback.print_exc()
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/{path:path}")
async def handle_request(request: Request, path: str):
    """Forwards requests based on the path to the appropriate OpenAPI endpoint."""
    segments = path.split('/')
    return await create_completion_base(request, extension=path)
