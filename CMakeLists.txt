cmake_minimum_required(VERSION 3.18)

project(CubeCobraRecommenderGenerators VERSION "0.0.1")

include (CheckCCompilerFlag)
include (CheckCXXCompilerFlag)

macro (ADD_FLAG_IF_AVAIL TARGET_NAME ACCESS_MODIFIER FLAG)
  if (NOT "${FLAG}" STREQUAL "")
    # create a variable for checking the flag if supported, e.g.:
    #   -fp-model=precise --> cxx_fp_model_precise
    string (REGEX REPLACE "^-" "cxx_" FLAG_NAME "${FLAG}")
    string (REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
    string (REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
    string (REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")

    # runs check to see flag is supported by compiler
    check_cxx_compiler_flag ("${FLAG}" ${FLAG_NAME})
    if (${FLAG_NAME})
      target_compile_options (${TARGET_NAME} ${ACCESS_MODIFIER} $<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
    endif ()
  endif ()
endmacro ()

if(SKBUILD)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LBIRARY "${PYTHON_LIBRARY}")
  set(DUMMY "${PYTHON_VERSION_STRING}")  # Not needed, silences a warning
endif()

set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT
)
message("${_tmp_dir}")
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(extern/concurrentqueue)

# pybind11_add_module(generator MODULE cpp/generator.cpp)
pybind11_add_module(draftbot_generator MODULE cpp/draftbot_generator.cpp)

target_compile_definitions(draftbot_generator PRIVATE VERSION_INFO=${PROJECT_VERSION})

target_include_directories(draftbot_generator PRIVATE extern/pcg-cpp/include)
target_link_libraries(draftbot_generator PRIVATE concurrentqueue)

target_compile_features(draftbot_generator PRIVATE cxx_std_17)

add_flag_if_avail (draftbot_generator PRIVATE -Wall)
add_flag_if_avail (draftbot_generator PRIVATE -Wextra)
add_flag_if_avail (draftbot_generator PRIVATE /W4)
add_flag_if_avail (draftbot_generator PRIVATE -march=native)
add_flag_if_avail (draftbot_generator PRIVATE -ffast-math)
add_flag_if_avail (draftbot_generator PRIVATE /arch:AVX2)
add_flag_if_avail (draftbot_generator PRIVATE -fdiagnostics-color)
add_flag_if_avail (draftbot_generator PRIVATE -fcolor-diagnostics)

install(TARGETS draftbot_generator DESTINATION .)
