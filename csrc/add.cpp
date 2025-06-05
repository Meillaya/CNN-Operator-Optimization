#include <torch/extension.h>
#include "add_kernel.h"

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Input a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Input b must be a CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "Input a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input b must be contiguous");
    TORCH_CHECK_EQ(a.numel(), b.numel());

    auto c = torch::empty_like(a);
    
    launch_add_kernel(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        a.numel()
    );

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "Vector Addition (CUDA)");
} 