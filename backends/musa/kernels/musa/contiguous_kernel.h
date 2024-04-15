#include "paddle/phi/extension.h"
namespace musa {
template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      phi::DenseTensor* out);
}