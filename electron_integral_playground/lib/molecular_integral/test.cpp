#include <stdio.h>

extern "C" {
    int debug_print() {
        printf("ABC\n");
        return 233;
    }
}
