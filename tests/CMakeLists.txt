CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG release-1.12.1
  VERSION 1.12.1
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)

# CPMAddPackage(
# NAME graphviz
# URL https://gitlab.com/graphviz/graphviz/-/archive/9.0.0/graphviz-9.0.0.zip
# )
include(add-tests)

include_directories(${CMAKE_SOURCE_DIR}/source)
sdk_add_test(valueTests SOURCES valueTests.cpp LIBRARIES engine)
sdk_add_test(networkTests SOURCES networkTests.cpp LIBRARIES engine network)
