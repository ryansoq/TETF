#ifndef F2UC_H
#define F2UC_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define COW 12

void float2uc(float *target, unsigned char **ptr, int size, bool dump, const char name[] = "weight")
{
    (*ptr) = (unsigned char *)malloc(size);
    memcpy(*ptr, target, size);

    if (dump)
    {
        for (int i = 0; i < size / sizeof(unsigned char); i++)
        {
            if (i)
            {
                printf(", ");
                if ((i % COW) == 0)
                    printf("\n");
            }
            else
            {
                printf("unsigned char ");
                printf("%s", name);
                printf("[] = {\n");
            }
            printf("0x%02x", (*ptr)[i]);
        }
        printf("};\n");
        printf("unsigned int ");
        printf("%s", name);
        printf("_size = %d;\n", size);
    }
}

void uc2float(unsigned char *target, float **ptr, int size, bool dump, const char name[] = "weight")
{
    (*ptr) = (float *)malloc(size);
    memcpy(*ptr, target, size);

    if (dump)
    {
        for (int i = 0; i < size / sizeof(float); i++)
        {
            if (i)
            {
                printf(", ");
                if ((i % COW) == 0)
                    printf("\n");
            }
            else
            {
                printf("float ");
                printf("%s", name);
                printf("[] = {\n");
            }
            printf("%08f", (*ptr)[i]);
        }
        printf("};\n");
        printf("unsigned int ");
        printf("%s", name);
        printf("_size = %d;\n", (size / (int)sizeof(float)));
    }
}

#endif
