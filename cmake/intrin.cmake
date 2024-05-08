function(check_if_avx_available)
    include(CheckCCompilerFlag)
    CHECK_C_COMPILER_FLAG("-mavx" COMPILER_SUPPORTS_AVX)
endfunction()

function(check_if_sse_available)
    include(CheckCCompilerFlag)
    CHECK_C_COMPILER_FLAG("-msse" COMPILER_SUPPORTS_SSE)
endfunction()
