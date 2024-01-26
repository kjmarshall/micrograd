function(sdk_add_test TARGET)
    set(flags)
    set(args)
    set(listArgs SOURCES LIBRARIES)

    cmake_parse_arguments(arg "${flags}" "${args}" "${listArgs}" ${ARGN})

    if(NOT arg_SOURCES)
        message(FATAL_ERROR "[sdk_add_test]: SOURCES is a required argument")
    endif()

    if(SOURCES IN_LIST arg_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "[sdk_add_test]: SOURCES requires at least one value")
    endif()

    add_executable(${TARGET} ${arg_SOURCES})

    target_link_libraries(${TARGET} PRIVATE gtest gmock gtest_main ${arg_LIBRARIES})

    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TARGET}

        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    set_target_properties(${TARGET} PROPERTIES FOLDER tests)
    sdk_setup_project_bin(${TARGET})
endfunction()