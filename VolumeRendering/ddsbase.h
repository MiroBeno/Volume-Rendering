// (c) by Stefan Roettger // prevzane z http://code.google.com/p/vvv/

#ifndef DDSBASE_H
#define DDSBASE_H

#include <stdio.h>
#include <stdlib.h>

#include "data_utils.h"

#define BOOLINT char
#ifndef TRUE
#define TRUE (1)
#endif
#ifndef FALSE
#define FALSE (0)
#endif
#define ERRORMSG() errormsg(__FILE__,__LINE__)
#define ffloor(x) floor((double)(x))
#define ftrc(x) (int)ffloor(x)

inline void errormsg(const char *file,int line)
   {
   fprintf(stderr,"fatal error in <%s> at line %d!\n",file,line);
   exit(EXIT_FAILURE);
   }

unsigned char *readRAWfile(const char *filename,unsigned int *bytes);

unsigned char *readPVMvolume(const char *filename,
                             unsigned int *width,unsigned int *height,unsigned int *depth,unsigned int *components=NULL,
                             float *scalex=NULL,float *scaley=NULL,float *scalez=NULL,
                             unsigned char **description=NULL,
                             unsigned char **courtesy=NULL,
                             unsigned char **parameter=NULL,
                             unsigned char **comment=NULL);

int checkfile(const char *filename);

unsigned char *quantize(unsigned char *volume,
                        unsigned int width,unsigned int height,unsigned int depth,
                        BOOLINT linear=FALSE,BOOLINT nofree=FALSE);

#endif
