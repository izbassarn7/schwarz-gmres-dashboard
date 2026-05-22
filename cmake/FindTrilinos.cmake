if(DEFINED ENV{Trilinos_DIR})
    set(Trilinos_DIR $ENV{Trilinos_DIR})
endif()

find_package(Trilinos QUIET HINTS ${Trilinos_DIR})

if(Trilinos_FOUND)
    message(STATUS "Found Trilinos: ${Trilinos_DIR}")
    message(STATUS "  Version: ${Trilinos_VERSION}")
    message(STATUS "  Packages: ${Trilinos_PACKAGE_LIST}")
else()
    message(FATAL_ERROR "Trilinos not found. Set Trilinos_DIR.")
endif()
