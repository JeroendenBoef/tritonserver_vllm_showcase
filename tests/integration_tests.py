# #!/usr/bin/env python3
import pytest
import requests
import json

# -------------------------
# Configuration
# -------------------------
TRITON_HOST = "localhost"
TRITON_HTTP_PORT = 8000

# Model endpoints
LLAMA_BLS_ENSEMBLE = f"http://{TRITON_HOST}:{TRITON_HTTP_PORT}/v2/models/llama_bls_ensemble/infer"
LLAMA_POSTPROCESS = f"http://{TRITON_HOST}:{TRITON_HTTP_PORT}/v2/models/llama_postprocess/infer"
LLAMA_MODEL = f"http://{TRITON_HOST}:{TRITON_HTTP_PORT}/v2/models/llama3-8b-instruct/generate"

# -------------------------
# Testcases
# -------------------------
ensemble_testcases = [
    {
        "description": "Simple request - no profanity",
        "payload": {
            "inputs": [
                {"name": "text_input", "shape": [1], "datatype": "BYTES", "data": ["How to fry an egg?"]},
                {"name": "temperature", "shape": [1], "datatype": "INT32", "data": [0]},
                {"name": "max_tokens", "shape": [1], "datatype": "INT32", "data": [150]},
                {"name": "stream", "shape": [1], "datatype": "BOOL", "data": [False]},
            ],
            "outputs": [{"name": "text_output"}],
        },
    },
    {
        "description": "Profanity in user request",
        "payload": {
            "inputs": [
                {"name": "text_input", "shape": [1], "datatype": "BYTES", "data": ["Fuck this!"]},
                {"name": "temperature", "shape": [1], "datatype": "INT32", "data": [0]},
                {"name": "max_tokens", "shape": [1], "datatype": "INT32", "data": [50]},
                {"name": "stream", "shape": [1], "datatype": "BOOL", "data": [False]},
            ],
            "outputs": [{"name": "text_output"}],
        },
    },
]

postprocess_testcases = [
    {
        "description": "Clean postprocess",
        "payload": {
            "inputs": [
                {
                    "name": "model_output",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["This is a nice output with no profanity."],
                }
            ],
            "outputs": [{"name": "postprocessed_output"}],
        },
    },
    {
        "description": "Profanity in model_output",
        "payload": {
            "inputs": [
                {
                    "name": "model_output",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["You are a bitch! This is some offensive text."],
                }
            ],
            "outputs": [{"name": "postprocessed_output"}],
        },
    },
]

llama_model_testcases = [
    {
        "description": "Direct Llama model call",
        "payload": {
            "text_input": "Hello",
            "parameters": {"stream": False, "temperature": 0, "max_tokens": 100},
        },
    }
]

# -------------------------
# Pytest Integration Tests
# -------------------------


@pytest.mark.parametrize("testcase", ensemble_testcases)
def test_llama_bls_ensemble(testcase):
    response = requests.post(LLAMA_BLS_ENSEMBLE, json=testcase["payload"])
    assert response.status_code == 200, f"Failed: {testcase['description']}"
    try:
        # Validate that the response is valid JSON.
        response.json()
    except json.JSONDecodeError:
        pytest.fail(f"Response is not valid JSON for: {testcase['description']}")


@pytest.mark.parametrize("testcase", postprocess_testcases)
def test_llama_postprocess(testcase):
    response = requests.post(LLAMA_POSTPROCESS, json=testcase["payload"])
    assert response.status_code == 200, f"Failed: {testcase['description']}"
    try:
        response.json()
    except json.JSONDecodeError:
        pytest.fail(f"Response is not valid JSON for: {testcase['description']}")


@pytest.mark.parametrize("testcase", llama_model_testcases)
def test_llama_model(testcase):
    response = requests.post(LLAMA_MODEL, json=testcase["payload"])
    assert response.status_code == 200, f"Failed: {testcase['description']}"
    try:
        response.json()
    except json.JSONDecodeError:
        pytest.fail(f"Response is not valid JSON for: {testcase['description']}")
