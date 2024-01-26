project(network)
message(STATUS "PROCESSING ${PROJECT_NAME}")

include_directories(${CMAKE_SOURCE_DIR}/source)

sdk_list_header_files(HEADER_FILES)
sdk_list_source_files(SOURCE_FILES)
add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
sdk_setup_project_lib(${PROJECT_NAME})