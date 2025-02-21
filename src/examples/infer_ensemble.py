import requests
import json

TRITON_URL = "http://localhost:8000/v2/models/llama_bls_ensemble/infer"

payload = {
    "inputs": [
        {"name": "text_input", "shape": [1], "datatype": "BYTES", "data": ["How to fry an egg?"]},
        {"name": "temperature", "shape": [1], "datatype": "INT32", "data": [0]},
        {"name": "max_tokens", "shape": [1], "datatype": "INT32", "data": [200]},
        {"name": "stream", "shape": [1], "datatype": "BOOL", "data": [False]},
    ],
    "outputs": [{"name": "text_output"}],
}


def main():
    # Send the POST request to Triton
    response = requests.post(TRITON_URL, json=payload)

    # Print status and the raw response text
    print("Status code:", response.status_code)
    if response.ok:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Error response text:", response.text)


if __name__ == "__main__":
    main()
