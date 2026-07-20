#include "da3cpp/da3_c.h"

int da3cpp_c_header_check(void) {
    return da_capi_abi_version() >= 4 ? 0 : 1;
}
