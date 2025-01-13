
add_library(molecular_integral SHARED 
    overlap.cpp
    nuclear_attraction.cpp
)

target_link_libraries(molecular_integral
    integral_utility
)
