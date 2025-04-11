find_package(Doxygen REQUIRED)

get_target_property(ZEBRADM_PUBLIC_HEADER_DIR zebradm INTERFACE_INCLUDE_DIRECTORIES)
file(GLOB_RECURSE ZEBRADM_PUBLIC_HEADERS ${ZEBRADM_PUBLIC_HEADER_DIR}/*.hpp)

set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/include)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/xml/index.xml)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
add_custom_command(
    OUTPUT ${DOXYGEN_INDEX_FILE}
    DEPENDS ${ZEBRADM_PUBLIC_HEADERS}
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
    COMMENT "Generating docs"
    VERBATIM
)

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(Python3_FOUND)
    message(STATUS "Python found at: ${Python3_EXECUTABLE}")
else()
    message(FATAL_ERROR "Python3 not found, not installing requirements.")
endif()

find_package(Sphinx REQUIRED)

set(SPHINX_CONFIG_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SPHINX_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD_DIR}/index.html)

add_custom_command(
    OUTPUT ${SPHINX_INDEX_FILE}
    COMMAND
        ${SPHINX_EXECUTABLE}
        -b html
        -c ${SPHINX_CONFIG_DIR}
        -Dbreathe_projects.zest=${DOXYGEN_OUTPUT_DIR}/xml
        ${SPHINX_SOURCE_DIR} ${SPHINX_BUILD_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${SPHINX_SOURCE_DIR}/index.rst
        ${SPHINX_SOURCE_DIR}/getting_started.rst
        ${SPHINX_SOURCE_DIR}/introduction.rst
        ${SPHINX_SOURCE_DIR}/reference.rst
        ${DOXYGEN_INDEX_FILE}
    MAIN_DEPENDENCY ${SPHINX_CONFIG_DIR}/conf.py
    COMMENT "Generating documentation with Sphinx: ${SPHINX_EXECUTABLE}"
)

add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})
