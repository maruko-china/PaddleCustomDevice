#include "paddle/phi/common/type_traits.h"
#include <mudnn.h>



namespace musa_kernel{


    template <typename T, typename Context>
    void AddKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                DenseTensor* out) {
        dev_ctx.template Alloc<T>(out);

    }
}

PD_REGISTER_PLUGIN_KERNEL(add,
                          musa,
                          ALL_LAYOUT,
                          musa_kernel::AddKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}