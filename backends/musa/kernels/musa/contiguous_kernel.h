#include "paddle/phi/extension.h"
namespace custom_kernel{

    template <typename T, typename Context>
    void ContiguousKernel(const Context& dev_ctx,
                        const phi::DenseTensor& input,
                        phi::DenseTensor* out);
}