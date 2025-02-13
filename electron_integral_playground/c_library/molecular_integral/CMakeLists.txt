
add_library(molecular_integral SHARED 
    overlap.cpp
    nuclear_attraction.cpp
    two_center.cpp
    three_center.cpp
)

target_link_libraries(molecular_integral
    integral_utility
)
