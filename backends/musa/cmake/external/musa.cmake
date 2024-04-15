list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/external)
find_package(MUDNN)
set(MUSA_DEPENDENT_LIBRARIES CACHE STRING "musa library.")

if(MUDNN_FOUND)
  list(APPEND DEPENDENT_INCLUDE_DIRS ${MUDNN_INCLUDE_DIRS})
  list(APPEND MUSA_DEPENDENT_LIBRARIES ${MUDNN_LIBRARIES})
else()
  # set default mudnn library path
  message(
    WARNING
      " The environment variable MUSA_HOME may be not specified. Using default MUDNN PATH: /usr/local/musa"
  )
  list(APPEND DEPENDENT_INCLUDE_DIRS "/usr/local/musa/include")
  list(APPEND MUSA_DEPENDENT_LIBRARIES "/usr/local/musa/lib/libmudnn.so")
  set(MUDNN_PATH "/usr/local/musa")
  set(MUDNN_LIBRARIES "/usr/local/musa/lib/libmudnn.so")
endif()

if(USE_MCCL)
  find_package(MCCL)

  if(MCCL_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${MCCL_INCLUDE_DIRS})
    list(APPEND MUSA_DEPENDENT_LIBRARIES ${MCCL_LIBRARIES})
  else()
    message(WARNING " NO MCCL FOUND?")
    list(APPEND DEPENDENT_INCLUDE_DIRS "/usr/local/musa/include")
    list(APPEND MUSA_DEPENDENT_LIBRARIES "/usr/local/musa/lib/libmccl.so")
    set(MCCL_PATH "/usr/local/musa")
    set(MCCL_LIBRARIES "/usr/local/musa/lib/libmccl.so")
  endif()

  add_definitions(-DUSE_MCCL)
endif()

find_package(MUSAToolkits)

if(MUSAToolkits_FOUND)
  list(APPEND DEPENDENT_INCLUDE_DIRS ${MUSAToolkits_INCLUDE_DIRS})
  list(APPEND MUSA_DEPENDENT_LIBRARIES ${MUSAToolkits_LIBRARIES})
else()
  # set default musa_toolkits path
  message(
    WARNING
      " The environment variable MUSA_HOME may be not specified. Using default MUSATOOLKITS PATH: /usr/local/musa"
  )
  list(APPEND DEPENDENT_INCLUDE_DIRS "/usr/local/musa/include/")
  list(APPEND MUSA_DEPENDENT_LIBRARIES "/usr/local/musa/lib/libmusart.so")
  set(ENV{MUSA_HOME} "/usr/local/musa")
  set(MUSATOOLKITS_PATH "/usr/local/musa")
  set(MUSAToolkits_LIBRARIES "/usr/local/musa/lib/")
endif()


include_directories(${DEPENDENT_INCLUDE_DIRS})
