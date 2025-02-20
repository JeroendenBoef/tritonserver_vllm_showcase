#!/usr/bin/env python3

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
# Global counters
# -------------------------
TOTAL_TESTS = 0
PASSED_TESTS = 0


def send_inference_request(url, testcase):
    global TOTAL_TESTS, PASSED_TESTS

    desc = testcase["description"]
    payload = testcase["payload"]
    TOTAL_TESTS += 1

    print(f"\n=== Testcase: {desc} ===")
    print(f"Sending POST to: {url}")
    response = requests.post(url, json=payload)
    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        PASS_STATUS = True
    else:
        PASS_STATUS = False

    if response.ok:
        try:
            resp_json = response.json()
            print("Response JSON:")
            print(json.dumps(resp_json, indent=2))
        except json.JSONDecodeError:
            print("Received non-JSON response:", response.text)
    else:
        print("Error response text:", response.text)

    if PASS_STATUS:
        PASSED_TESTS += 1


def test_llama_bls_ensemble():
    print("\n[ Testing llama_bls_ensemble Endpoint ]")
    for tc in ensemble_testcases:
        send_inference_request(LLAMA_BLS_ENSEMBLE, tc)


def test_llama_postprocess():
    print("\n[ Testing llama_postprocess Endpoint ]")
    for tc in postprocess_testcases:
        send_inference_request(LLAMA_POSTPROCESS, tc)


def test_llama_model():
    print("\n[ Testing llama3-8b-instruct (LLM) Endpoint ]")
    for tc in llama_model_testcases:
        send_inference_request(LLAMA_MODEL, tc)


def main():
    test_llama_bls_ensemble()
    test_llama_postprocess()
    test_llama_model()

    print("\n====================")
    print("TEST RESULTS SUMMARY")
    print("====================")
    print(f"Total tests: {TOTAL_TESTS}")
    print(f"Passed tests: {PASSED_TESTS}")
    failed = TOTAL_TESTS - PASSED_TESTS
    print(f"Failed tests: {failed}")


if __name__ == "__main__":
    main()
