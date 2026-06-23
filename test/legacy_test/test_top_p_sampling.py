#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import numpy as np
from op_test import get_device_place, is_custom_device

import paddle
from paddle.base import core


def TopPProcess(probs, top_p):
    sorted_probs = paddle.sort(probs, descending=True)
    sorted_indices = paddle.argsort(probs, descending=True)
    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

    # Remove tokens with cumulative probs above the top_p, But keep at
    # least min_tokens_to_keep tokens
    sorted_indices_to_remove = cumulative_probs > top_p

    # Keep the first token
    sorted_indices_to_remove = paddle.cast(
        sorted_indices_to_remove, dtype='int64'
    )

    sorted_indices_to_remove = paddle.static.setitem(
        sorted_indices_to_remove,
        (slice(None), slice(1, None)),
        sorted_indices_to_remove[:, :-1].clone(),
    )
    sorted_indices_to_remove = paddle.static.setitem(
        sorted_indices_to_remove, (slice(None), 0), 0
    )

    # Scatter sorted tensors to original indexing
    sorted_indices = (
        sorted_indices
        + paddle.arange(probs.shape[0]).unsqueeze(-1) * probs.shape[-1]
    )
    condition = paddle.scatter(
        sorted_indices_to_remove.flatten(),
        sorted_indices.flatten(),
        sorted_indices_to_remove.flatten(),
    )
    condition = paddle.cast(condition, 'bool').reshape(probs.shape)
    probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
    next_tokens = paddle.multinomial(probs)
    next_scores = paddle.index_sample(probs, next_tokens)
    return next_scores, next_tokens


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA ",
)
class TestTopPAPI(unittest.TestCase):
    def setUp(self):
        self.topp = 0.0
        self.seed = 6688
        self.batch_size = 3
        self.vocab_size = 10000
        self.dtype = "float32"
        self.input_data = np.random.rand(self.batch_size, self.vocab_size)

    def run_dygraph(self, place):
        with paddle.base.dygraph.guard(place):
            input_tensor = paddle.to_tensor(self.input_data, self.dtype)
            topp_tensor = paddle.to_tensor(
                [
                    self.topp,
                ]
                * self.batch_size,
                self.dtype,
            ).reshape((-1, 1))

            # test case for basic test case 1
            paddle_result = paddle.tensor.top_p_sampling(
                input_tensor, topp_tensor, seed=self.seed
            )
            ref_res = TopPProcess(input_tensor, self.topp)

            np.testing.assert_allclose(
                paddle_result[0].numpy(), ref_res[0].numpy(), rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[1].numpy().flatten(),
                ref_res[1].numpy().flatten(),
                rtol=0,
            )

            # test case for basic test case 1
            paddle_result = paddle.tensor.top_p_sampling(
                input_tensor,
                topp_tensor,
                seed=-1,
                k=5,
                mode="non-truncated",
                return_top=True,
            )
            ref_res = TopPProcess(input_tensor, self.topp)

            np.testing.assert_allclose(
                paddle_result[0].numpy(), ref_res[0].numpy(), rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[1].numpy().flatten(),
                ref_res[1].numpy().flatten(),
                rtol=0,
            )

    def run_static(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input_tensor = paddle.static.data(
                name="x", shape=[6, 1030], dtype=self.dtype
            )
            topp_tensor = paddle.static.data(
                name="topp", shape=[6, 1], dtype=self.dtype
            )
            result = paddle.tensor.top_p_sampling(
                input_tensor, topp_tensor, seed=self.seed
            )
            ref_res = TopPProcess(input_tensor, self.topp)
            exe = paddle.static.Executor(place)
            input_data = np.random.rand(6, 1030).astype(self.dtype)
            paddle_result = exe.run(
                feed={
                    "x": input_data,
                    "topp": np.array(
                        [
                            self.topp,
                        ]
                        * 6
                    ).astype(self.dtype),
                },
                fetch_list=[
                    result[0],
                    result[1],
                    ref_res[0],
                    ref_res[1],
                ],
            )
            np.testing.assert_allclose(
                paddle_result[0], paddle_result[2], rtol=1e-05
            )
            np.testing.assert_allclose(
                paddle_result[1], paddle_result[3], rtol=1e-05
            )

    def test_dygraph(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            places = [get_device_place()]
            for place in places:
                self.run_dygraph(place)

    def test_static(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            places = [get_device_place()]
            for place in places:
                self.run_static(place)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestTopPAPIBF16(unittest.TestCase):
    def test_dygraph_bfloat16(self):
        with paddle.base.dygraph.guard(get_device_place()):
            input_tensor = paddle.to_tensor(
                [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]], dtype="float32"
            ).astype("bfloat16")
            topp_tensor = paddle.to_tensor(
                [[0.8], [0.8]], dtype="float32"
            ).astype("bfloat16")

            out, ids = paddle.tensor.top_p_sampling(
                input_tensor, topp_tensor, seed=2023
            )

            self.assertEqual(out.dtype, paddle.bfloat16)
            self.assertEqual(ids.dtype, paddle.int64)
            self.assertEqual(out.shape, [2, 1])
            self.assertEqual(ids.shape, [2, 1])
            ids.numpy()


if __name__ == "__main__":
    unittest.main()
