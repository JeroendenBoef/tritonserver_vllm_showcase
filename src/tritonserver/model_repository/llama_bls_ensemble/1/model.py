import time
import json
import logging

import numpy as np
import c_python_backend_utils as c_utils
import triton_python_backend_utils as pb_utils
from profanity_checker import check_for_profanity

logging.basicConfig(
    level=logging.ERROR,  # Set the log level (DEBUG, INFO, etc.)
    format="[%(asctime)s] - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class TritonPythonModel:
    def initialize(self, args: dict):
        self.logger = pb_utils.Logger

    def execute(self, requests: list[c_utils.InferenceRequest]):
        self.logger.log_info(f"Evaluating input for banned words")
        t0 = time.time()
        responses = []
        for request in requests:
            input_text = (
                pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()[0].decode("utf-8")
            )
            temperature = pb_utils.get_input_tensor_by_name(request, "temperature").as_numpy()[0]
            max_tokens = pb_utils.get_input_tensor_by_name(request, "max_tokens").as_numpy()[0]
            stream = bool(pb_utils.get_input_tensor_by_name(request, "stream").as_numpy()[0])

            profanity, processed_text = check_for_profanity(input_text)
            t1 = time.time()
            self.logger.log_info(f"Time elapsed for input validation: {np.around(t1 - t0, 3)} seconds")

            if profanity:
                out_tensor = pb_utils.Tensor(
                    "text_output", np.array([processed_text.encode("utf-8")], dtype="object")
                )
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(inference_response)
                continue

            else:
                # BLS call to Llama model
                sampling_params = self._construct_sampling_json(temperature, max_tokens)
                llama_inputs = [
                    pb_utils.Tensor("text_input", np.array([processed_text.encode("utf-8")], dtype="object")),
                    pb_utils.Tensor("stream", np.array([stream], dtype=bool)),
                    pb_utils.Tensor(
                        "sampling_parameters", np.array([sampling_params.encode("utf-8")], dtype="object")
                    ),
                ]
            t2 = time.time()
            llama_request = pb_utils.InferenceRequest(
                model_name="llama3-8b-instruct", requested_output_names=["text_output"], inputs=llama_inputs
            )

            llama_responses = llama_request.exec(decoupled=True)
            llama_output_str = ""
            for infer_response in llama_responses:
                if infer_response.has_error():
                    error_msg = infer_response.error().message()
                    self.logger.log_error(f"Error from Llama model: {error_msg}")
                    raise pb_utils.TritonModelException(error_msg)

                if len(infer_response.output_tensors()) > 0:
                    llama_output_tensor = pb_utils.get_output_tensor_by_name(infer_response, "text_output")
                    llama_output_str += llama_output_tensor.as_numpy()[0].decode("utf-8")
            t3 = time.time()
            self.logger.log_info(f"Time elapsed for Llama text generation: {np.around(t3 - t2, 3)} seconds")

            # BLS Call to postprocess model
            postprocess_inputs = [
                pb_utils.Tensor("model_output", np.array([llama_output_str.encode("utf-8")], dtype="object"))
            ]
            postprocess_request = pb_utils.InferenceRequest(
                model_name="llama_postprocess",
                requested_output_names=["postprocessed_output"],
                inputs=postprocess_inputs,
            )
            postprocess_response = postprocess_request.exec()
            if postprocess_response.has_error():
                error_msg = postprocess_response.error().message()
                self.logger.log_error(f"Error from postprocess model: {error_msg}")
                raise pb_utils.TritonModelException(error_msg)

            final_output_tensor = pb_utils.get_output_tensor_by_name(
                postprocess_response, "postprocessed_output"
            )
            final_output_str = final_output_tensor.as_numpy()[0].decode("utf-8")

            out_tensor = pb_utils.Tensor(
                "text_output", np.array([final_output_str.encode("utf-8")], dtype="object")
            )
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def _construct_sampling_json(self, temperature: int, max_tokens: int):
        return json.dumps({"temperature": int(temperature), "max_tokens": int(max_tokens)})
