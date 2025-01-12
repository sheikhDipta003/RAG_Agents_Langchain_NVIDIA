from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
import httpx
import os
from pydantic import BaseModel
from typing import Dict, List

import jinja2
import json
import time

app = FastAPI()

@app.get("/")
async def list_endpoints():
    """Lists all available endpoints with their descriptions."""
    exclude_list = ["/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"]
    return [{
        route.path: getattr(route, "description", "No description"),
    } for route in app.routes if route.path not in exclude_list]

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running."""
    return True

@app.get("/hello")
async def read_root():
    """Returns a simple greeting, useful for initial testing."""
    return {"Hello": "World"}

# Global state to store API keys, endpoint URL, and models
app_state = {
    "key_dict": {
        "NVIDIA_API_KEY": os.environ.get("NVIDIA_API_KEY")
    },
    "default_model": "ai-mixtral-8x7b-instruct",
    "models": {
        # "mixtral_local": {
        #     "infer": "http://open_llm_cloud:9010/v1/completions",
        #     "health": "http://open_llm_cloud:9010/health",
        #     "wait": "",
        #     "chat_template_path": "/open_llm/mistral-instruct.jinja",
        #     "model_specs": {
        #         "model_type": "chat",
        #         "client": "ChatNVIDIA",
        #         "api_type": "open",
        #         "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        #     }
        # },
        # "sql_coder": {
        #     "infer": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/d776fe9c-38d6-4ba7-aa54-076893fa8705",
        #     "health": "https://api.nvcf.nvidia.com/v2/nvcf/functions/d776fe9c-38d6-4ba7-aa54-076893fa8705",
        #     "wait": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
        #     "key": "NVIDIA_API_KEY",
        #     "chat_template_path": "/open_llm/sqlcoder-chat.jinja",
        #     "model_specs": {
        #         "model_type": "chat",
        #         "client": "ChatNVIDIA",
        #         "api_type": "open",
        #         "model_name": "defog/sqlcoder-70b-alpha",
        #     }
        # },
        # "dbrx-instruct": {
        #     "infer": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/ecaea17a-8e58-445f-9730-184fac24ff71",
        #     "health": "https://api.nvcf.nvidia.com/v2/nvcf/functions/ecaea17a-8e58-445f-9730-184fac24ff71",
        #     "wait": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
        #     "key": "NVIDIA_API_KEY",
        #     "model_name": "alpindale/dbrx-instruct",
        #     "model_specs": {
        #         "model_type": "chat",
        #         "client": "ChatNVIDIA",
        #         "api_type": "open",
        #         "model_name": "dbrx-instruct",
        #     }
        # },
        "ai-llama2-70b" : {
            "infer": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/2fddadfb-7e76-4c8a-9b82-f7d3fab94471",
            "health": "https://api.nvcf.nvidia.com/v2/nvcf/functions/2fddadfb-7e76-4c8a-9b82-f7d3fab94471",
            "wait": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
            "key": "NVIDIA_API_KEY",
            "model_name": "meta/llama2-70b",
            "model_specs": {
                "model_type": "chat",
                "client": "ChatNVIDIA",
                "api_type": "open",
                "max_tokens": 1024,
                "model_name": "ai-llama2-70b",
            }
        },
        "ai-mixtral-8x7b-instruct": {
            "infer": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/a1e53ece-bff4-44d1-8b13-c009e5bf47f6",
            "health": "https://api.nvcf.nvidia.com/v2/nvcf/functions/a1e53ece-bff4-44d1-8b13-c009e5bf47f6",
            "wait": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
            "key": "NVIDIA_API_KEY",
            "model_name": "mistralai/mixtral-8x7b-instruct-v0.1",
            "model_specs": {
                "model_type": "chat",
                "client": "ChatNVIDIA",
                "api_type": "open",
                "max_tokens": 1024,
                "model_name": "ai-mixtral-8x7b-instruct",
            }
        },
        "ai-mistral-7b-instruct-v2": {
            "infer": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/d7618e99-db93-4465-af4d-330213a7f51f",
            "health": "https://api.nvcf.nvidia.com/v2/nvcf/functions/d7618e99-db93-4465-af4d-330213a7f51f",
            "wait": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
            "key": "NVIDIA_API_KEY",
            "model_name": "mistralai/mistral-7b-instruct-v0.2",
            "model_specs": {
                "model_type": "chat",
                "client": "ChatNVIDIA",
                "api_type": "open",
                "max_tokens": 1024,
                "model_name": "ai-mistral-7b-instruct-v2",
            }
        },
        "ai-mistral-large": {
            "infer": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/767b5b9a-3f9d-4c1d-86e8-fa861988cee7",
            "health": "https://api.nvcf.nvidia.com/v2/nvcf/functions/767b5b9a-3f9d-4c1d-86e8-fa861988cee7",
            "wait": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
            "key": "NVIDIA_API_KEY",
            "model_name": "mistralai/mistral-large",
            "model_specs": {
                "model_type": "chat",
                "client": "ChatNVIDIA",
                "api_type": "open",
                "max_tokens": 8096,
                "model_name": "ai-mistral-large",
            }
        },
        "ai-llama3-8b": {
            "infer": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/a5a3ad64-ec2c-4bfc-8ef7-5636f26630fe",
            "health": "https://api.nvcf.nvidia.com/v2/nvcf/functions/a5a3ad64-ec2c-4bfc-8ef7-5636f26630fe",
            "wait": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
            "key": "NVIDIA_API_KEY",
            "model_name": "meta/llama3-8b",
            "model_specs": {
                "model_type": "chat",
                "client": "ChatNVIDIA",
                "api_type": "open",
                "max_tokens": 1024,
                "model_name": "ai-llama3-8b",
            }
        },
        "ai-llama3-70b": {
            "infer": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/a88f115a-4a47-4381-ad62-ca25dc33dc1b",
            "health": "https://api.nvcf.nvidia.com/v2/nvcf/functions/a88f115a-4a47-4381-ad62-ca25dc33dc1b",
            "wait": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
            "key": "NVIDIA_API_KEY",
            "model_name": "meta/llama3-70b",
            "model_specs": {
                "model_type": "chat",
                "client": "ChatNVIDIA",
                "api_type": "open",
                "max_tokens": 1024,
                "model_name": "ai-llama3-70b",
            }
        },
        "ai-embed-qa-4": {
            "infer": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/09c64e32-2b65-4892-a285-2f585408d118",
            "health": "https://api.nvcf.nvidia.com/v2/nvcf/functions/09c64e32-2b65-4892-a285-2f585408d118",
            "wait": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
            "key": "NVIDIA_API_KEY",
            "model_name": "NV-Embed-QA",
            "model_specs": {
                "model_type": "embedding",
                "model_name": "ai-embed-qa-4",
                "client": "NVIDIAEmbeddings",
                "api_type": "open",
            }
        },
        "ai-rerank-qa-mistral-4b": {
            "infer": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0bf77f50-5c35-4488-8e7a-f49bb1974af6",
            "health": "https://api.nvcf.nvidia.com/v2/nvcf/functions/0bf77f50-5c35-4488-8e7a-f49bb1974af6",
            "wait": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
            "key": "NVIDIA_API_KEY",
            "model_name": "nv-rerank-qa-mistral-4b:1",
            "model_specs": {
                "model_type": "ranking",
                "client": "NVIDIARerank",
                "api_type": "open",
            }
        },
        # "ai-mixtral-8x22b-instruct": {
        #     "infer": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/39655fc1-9ebc-4b24-963e-6915ea6680de",
        #     "health": "https://api.nvcf.nvidia.com/v2/nvcf/functions/39655fc1-9ebc-4b24-963e-6915ea6680de",
        #     "wait": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
        #     "key": "NVIDIA_API_KEY",
        #     "model_name": "mistralai/mixtral-8x22b-instruct-v0.1",
        #     "model_specs": {
        #         "model_type": "chat",
        #         "client": "ChatNVIDIA",
        #         "api_type": "open",
        #         "max_tokens": 8096,
        #         "model_name": "ai-mixtral-8x22b-instruct",
        #     }
        # },
    },
}

class Url(BaseModel):
    endpoint_url: str

class ModelList(BaseModel):
    models: Dict[str, str]

###################################################################################
## OpenAPI-Style Endpoints

def raise_helper(msg):
    raise Exception(msg)


for model, spec in app_state.get("models").items():
    if "chat_template_path" in spec:
        path = spec.get("chat_template_path")
        spec["chat_template"] = jinja2.Template(open(path, "r").read())
        spec["chat_template"].environment.globals['raise_exception'] = raise_helper


@app.get("/v1/models")
async def get_models():
    """Returns the list of OpenAPI models."""
    model_sate = [
        {"id": model_name, **v.get("model_specs")} 
        for model_name, v in app_state.get('models').items()
    ]
    return {"type": "object", "data": model_sate}


async def create_completion_base(request: Request, is_chat_endpoint: bool):
    """Forwards chat completion requests to OpenAPI endpoint"""
    try: 
        body = await request.body()
        body = json.loads(body.decode())
        headers = request.headers.mutablecopy()
        if "host" in headers:
            del headers["host"]
        if "content-length" in headers:
            del headers["content-length"]

        model_name = body.pop("model", app_state.get("default_model"))
        model_spec = app_state.get("models").get(model_name, {})
        assert model_spec, f"Unknown Model Requested: {model_name}. Available: {list(app_state.get('models').keys())}"
        base_url = model_spec.get("infer")
        if model_spec.get("model_name"):
            body["model"] = model_spec["model_name"]
        stream = body.get("stream")

        ## Using our API key instead of yours
        if "key" in model_spec:
            key_dict = app_state.get('key_dict')
            key_name = model_spec.get('key')
            headers["authorization"] = f"Bearer {key_dict.get(key_name)}"
        if stream:
            headers["accept"] = "text/event-stream"

        if is_chat_endpoint:
            if model_spec.get("chat_template") and "messages" in body:
                body["prompt"] = model_spec.get("chat_template").render({"messages" : body.pop("messages")}).lstrip()
            else: 
                assert "prompt" not in body, "Cannot feed prompt into /chat/completions endpoint. Please use /completions"
            ## Special edge case for mistral/mixtral, since some implementations do not allow system messages
            messages = body.get("messages", [])
            if "mixtral" in model_name.lower() or "mistral" in model_name.lower():
                role_map = {"assistant": "assistant"}
                new_msgs = []
                for i, msg in enumerate(messages):
                    old_role = msg.get("role")
                    new_role = role_map.get(old_role, "user")
                    if new_msgs and new_msgs[-1].get("role") == new_role:
                        new_msgs[-1]["content"] += "\n" + "="*16 + "\n"
                        new_msgs[-1]["content"] += msg.get("content", "")
                    else: 
                        msg["role"] = new_role
                        new_msgs += [msg]
                body["messages"] = new_msgs

        else:
            assert "messages" not in body, "Cannot feed messages into /completions endpoint. Please use /chat/completions"

        body = json.dumps(body).encode()

        call_kws = {
            "url": base_url,
            "content": body,
            "headers": headers, 
            "timeout": None,
        }

        print(call_kws)

        timeout = httpx.Timeout(5, read=None)

        if not stream:

            async with httpx.AsyncClient(timeout=timeout) as client:

                response = await client.post(**call_kws)

                while response.status_code == 202:
                    request_id = response.headers.get("NVCF-REQID")
                    fetch_url = os.path.join(model_spec.get('wait'), request_id)
                    time.sleep(0.5)
                    response = await client.get(fetch_url, headers=headers)

                content = response.content
                
                if is_chat_endpoint and content != b"":
                    resp_dict = json.loads(content.decode())
                    if resp_dict.get("choices"):
                        for choice in resp_dict.get("choices"): 
                            if "text" in choice:
                                choice["message"] = {"role": "assistant", "content": choice.pop("text")}
                            elif "delta" in choice:
                                choice["delta"] = {"role": "assistant", "content": choice.pop("delta")}
                    content = json.dumps(resp_dict).encode()

            filtered_headers = {
                key: value for key, value in response.headers.items() 
                if key.lower() not in ["content-length", "content-encoding", "transfer-encoding"]
            }

            return Response(content=content, status_code=response.status_code, headers=filtered_headers)

        else: 

            async def try_gen():
                async with httpx.AsyncClient().stream("POST", **call_kws) as response:
                    yield response
                    agen = response.aiter_bytes()
                    async for cbytes in agen:
                        yield cbytes

            agen = try_gen()
            response = await agen.__anext__()

            if response.status_code != 200:
                response_bytes = await response.aread()
                content = response_bytes
                filtered_headers = {
                    key: value for key, value in response.headers.items() 
                    if key.lower() not in ["content-length", "content-encoding", "transfer-encoding"]
                }
                return Response(content=content, status_code=response.status_code, headers=filtered_headers)

            async def astream_fn():
                async for cbytes in agen:
                    if cbytes.startswith(b"data:"):
                        chunks = cbytes[5:].split(b"\n\ndata:")
                        chunks = [cell.strip() for cell in chunks]
                        chunks = [cell for cell in chunks if cell != b"[DONE]"]
                    else: 
                        chunks = [cbytes]
                    for chunk in chunks:
                        if is_chat_endpoint:
                            resp_dict = json.loads(chunk.decode())
                            assert chunk, resp_dict
                            if resp_dict.get("choices"):
                                for choice in resp_dict.get("choices"): 
                                    if "text" in choice:
                                        choice["delta"] = {"role": "assistant", "content": choice.pop("text")}
                            chunk = json.dumps(resp_dict).encode()
                            assert chunk, (chunk, chunk.startswith(b"data:"))
                        yield b"data: " + chunk + b"\n\n"

            return StreamingResponse(astream_fn(), media_type='text/event-stream')

    except Exception as e:
        # raise e
        print(f"Exception Raised: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def create_completion(request: Request):
    """Forwards completion requests to OpenAPI endpoint"""
    return await create_completion_base(request, is_chat_endpoint=False)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    """Forwards chat completion requests to OpenAPI endpoint"""
    return await create_completion_base(request, is_chat_endpoint=True)

@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    """Forwards chat completion requests to OpenAPI endpoint"""
    return await create_completion_base(request, is_chat_endpoint=False)

# @app.post("/set_keys/")
# async def set_key(data: Dict[str, str]):
#     global app_state 
#     app_state.keys.update(data)
#     os.environ.update(data)
#     return {"result" : "Keys set successfully"}

# @app.get("/get_keys/")
# async def get_key():
#     global app_state
#     return {"nvapi_key": app_state.get("keys")}