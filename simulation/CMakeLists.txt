cmake_minimum_required(VERSION 3.13)
project(muPos)

option(WITH_GEANT4_UIVIS "Build Geant4 UI and visualization drivers" OFF)
if(WITH_GEANT4_UIVIS)
    find_package(Geant4 REQUIRED ui_all vis_all)
else()
    find_package(Geant4 REQUIRED)
endif()
find_package(ROOT REQUIRED)
find_package(yaml-cpp REQUIRED)
set(CRY_ROOT "" CACHE PATH "CRY installation root")
find_path(CRY_INCLUDE_DIR CRYGenerator.h HINTS ${CRY_ROOT}/src REQUIRED)
find_library(CRY_LIBRARY NAMES CRY HINTS ${CRY_ROOT}/lib REQUIRED)
find_path(CRY_DATA_DIR cosmics_0.data HINTS ${CRY_ROOT}/data REQUIRED)

include(${Geant4_USE_FILE})
include(${ROOT_DIR}/ROOTUseFile.cmake)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(SYSTEM ${Geant4_INCLUDE_DIR} ${ROOT_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR} ${CRY_INCLUDE_DIR})
link_directories(${YAML_CPP_LIBRARY_DIR})
add_compile_definitions(CRY_DATA="${CRY_DATA_DIR}")
add_compile_options(-O3 -g -Wall -Wshadow -Wextra -Werror -Wno-error=cpp)

add_custom_command(
    OUTPUT Object.cxx Object_rdict.pcm
    COMMAND rootcint -f Object.cxx ${PROJECT_SOURCE_DIR}/include/Object.hh ${PROJECT_SOURCE_DIR}/src/ObjectLinkDef.h
    DEPENDS include/Object.hh src/ObjectLinkDef.h
    IMPLICIT_DEPENDS include/Object.hh
)
file(GLOB sources CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/[A-Za-z]*.cc)
file(GLOB headers CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/[A-Za-z]*.hh)
add_executable(muPos muPos.cc ${sources} ${PROJECT_BINARY_DIR}/Object.cxx ${headers})
target_link_libraries(muPos ${Geant4_LIBRARIES} ${ROOT_LIBRARIES} ${YAML_CPP_LIBRARIES} ${CRY_LIBRARY})
