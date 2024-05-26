function(check_for_mkl_available USE_MKL)
    if (USE_MKL)
        find_package(MKL CONFIG)
        if (NOT MKL_FOUND)
            message(STATUS "Can not find MKL")
            add_definitions(MKL)
        else()
            message(STATUS "Intel MKL found")
        endif()
    else()
        message(STATUS "You mention that do not want to use MKL")
    endif()
endfunction()