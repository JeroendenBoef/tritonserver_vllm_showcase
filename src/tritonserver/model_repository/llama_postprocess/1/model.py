import time
import logging

import numpy as np
import c_python_backend_utils as c_utils
import triton_python_backend_utils as pb_utils
from model_gaurdrails import censor_profanity

logging.basicConfig(
    level=logging.ERROR,  # Set the log level (DEBUG, INFO, etc.)
    format="[%(asctime)s] - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class TritonPythonModel:
    def initialize(self, args: dict):
        self.logger = pb_utils.Logger

    def execute(self, requests: list[c_utils.InferenceRequest]):
        self.logger.log_info(f"Evaluating model output for banned words")
        t0 = time.time()
        responses = []
        for request in requests:
            model_output = (
                pb_utils.get_input_tensor_by_name(request, "model_output").as_numpy()[0].decode("utf-8")
            )
            postprocessed_output = censor_profanity(model_output)

            postprocessed_output_tensor = pb_utils.Tensor(
                "postprocessed_output", np.array([postprocessed_output.encode("utf-8")], dtype="object")
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[postprocessed_output_tensor]))

        t1 = time.time()
        self.logger.log_info(f"Time elapsed for model output validation: {np.around(t1 - t0, 3)} seconds")
        return responses
