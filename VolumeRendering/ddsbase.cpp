// (c) by Stefan Roettger // prevzane z http://code.google.com/p/vvv/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ddsbase.h"

#define ERRORMSG() errormsg(__FILE__,__LINE__)

#define ffloor(x) floor((double)(x))
#define ftrc(x) (int)ffloor(x)

#define DDS_MAXSTR (256)

#define DDS_BLOCKSIZE (1<<20)
#define DDS_INTERLEAVE (1<<24)

#define DDS_RL (7)

#define DDS_ISINTEL (*((unsigned char *)(&DDS_INTEL)+1)==0)

char DDS_ID[]="DDS v3d\n";
char DDS_ID2[]="DDS v3e\n";

unsigned char *DDS_cache;
unsigned int DDS_cachepos,DDS_cachesize;

unsigned int DDS_buffer;
unsigned int DDS_bufsize;

unsigned short int DDS_INTEL=1;

inline void errormsg(const char *file,int line)
   {
   fprintf(stderr,"Fatal error in <%s> at line %d!\n",file,line);
   exit(EXIT_FAILURE);
   }

// helper functions for DDS:

inline unsigned int DDS_shiftl(const unsigned int value,const unsigned int bits)
   {return((bits>=32)?0:value<<bits);}

inline unsigned int DDS_shiftr(const unsigned int value,const unsigned int bits)
   {return((bits>=32)?0:value>>bits);}

inline void DDS_swapuint(unsigned int *x)
   {
   unsigned int tmp=*x;

   *x=((tmp&0xff)<<24)|
      ((tmp&0xff00)<<8)|
      ((tmp&0xff0000)>>8)|
      ((tmp&0xff000000)>>24);
   }

void DDS_initbuffer()
   {
   DDS_buffer=0;
   DDS_bufsize=0;
   }

inline void DDS_clearbits()
   {
   DDS_cache=NULL;
   DDS_cachepos=0;
   DDS_cachesize=0;
   }

inline void DDS_loadbits(unsigned char *data,unsigned int size)
   {
   DDS_cache=data;
   DDS_cachesize=size;

   if ((DDS_cache=(unsigned char *)realloc(DDS_cache,DDS_cachesize+4))==NULL) ERRORMSG();
   *((unsigned int *)&DDS_cache[DDS_cachesize])=0;

   DDS_cachesize=4*((DDS_cachesize+3)/4);
   if ((DDS_cache=(unsigned char *)realloc(DDS_cache,DDS_cachesize))==NULL) ERRORMSG();
   }

inline unsigned int DDS_readbits(unsigned int bits)
   {
   unsigned int value;

   if (bits<DDS_bufsize)
      {
      DDS_bufsize-=bits;
      value=DDS_shiftr(DDS_buffer,DDS_bufsize);
      }
   else
      {
      value=DDS_shiftl(DDS_buffer,bits-DDS_bufsize);

      if (DDS_cachepos>=DDS_cachesize) DDS_buffer=0;
      else
         {
         DDS_buffer=*((unsigned int *)&DDS_cache[DDS_cachepos]);
         if (DDS_ISINTEL) DDS_swapuint(&DDS_buffer);
         DDS_cachepos+=4;
         }

      DDS_bufsize+=32-bits;
      value|=DDS_shiftr(DDS_buffer,DDS_bufsize);
      }

   DDS_buffer&=DDS_shiftl(1,DDS_bufsize)-1;

   return(value);
   }

inline int DDS_decode(int bits)
   {return(bits>=1?bits+1:bits);}

// deinterleave a byte stream
void DDS_deinterleave(unsigned char *data,unsigned int bytes,unsigned int skip,unsigned int block=0,BOOLINT restore=FALSE)
   {
   unsigned int i,j,k;

   unsigned char *data2,*ptr;

   if (skip<=1) return;

   if (block==0)
      {
      if ((data2=(unsigned char *)malloc(bytes))==NULL) ERRORMSG();

      if (!restore)
         for (ptr=data2,i=0; i<skip; i++)
            for (j=i; j<bytes; j+=skip) *ptr++=data[j];
      else
         for (ptr=data,i=0; i<skip; i++)
            for (j=i; j<bytes; j+=skip) data2[j]=*ptr++;

      memcpy(data,data2,bytes);
      }
   else
      {
      if ((data2=(unsigned char *)malloc((bytes<skip*block)?bytes:skip*block))==NULL) ERRORMSG();

      if (!restore)
         {
         for (k=0; k<bytes/skip/block; k++)
            {
            for (ptr=data2,i=0; i<skip; i++)
               for (j=i; j<skip*block; j+=skip) *ptr++=data[k*skip*block+j];

            memcpy(data+k*skip*block,data2,skip*block);
            }

         for (ptr=data2,i=0; i<skip; i++)
            for (j=i; j<bytes-k*skip*block; j+=skip) *ptr++=data[k*skip*block+j];

         memcpy(data+k*skip*block,data2,bytes-k*skip*block);
         }
      else
         {
         for (k=0; k<bytes/skip/block; k++)
            {
            for (ptr=data+k*skip*block,i=0; i<skip; i++)
               for (j=i; j<skip*block; j+=skip) data2[j]=*ptr++;

            memcpy(data+k*skip*block,data2,skip*block);
            }

         for (ptr=data+k*skip*block,i=0; i<skip; i++)
            for (j=i; j<bytes-k*skip*block; j+=skip) data2[j]=*ptr++;

         memcpy(data+k*skip*block,data2,bytes-k*skip*block);
         }
      }

   free(data2);
   }

// interleave a byte stream
void DDS_interleave(unsigned char *data,unsigned int bytes,unsigned int skip,unsigned int block=0)
   {DDS_deinterleave(data,bytes,skip,block,TRUE);}

// decode a Differential Data Stream
void DDS_decode(unsigned char *chunk,unsigned int size,
                unsigned char **data,unsigned int *bytes,
                unsigned int block=0)
   {
   unsigned int skip,strip;

   unsigned char *ptr1,*ptr2;

   unsigned int cnt,cnt1,cnt2;
   int bits,act;

   DDS_initbuffer();

   DDS_clearbits();
   DDS_loadbits(chunk,size);

   skip=DDS_readbits(2)+1;
   strip=DDS_readbits(16)+1;

   ptr1=ptr2=NULL;
   cnt=act=0;

   while ((cnt1=DDS_readbits(DDS_RL))!=0)
      {
      bits=DDS_decode(DDS_readbits(3));

      for (cnt2=0; cnt2<cnt1; cnt2++)
         {
         if (strip==1 || cnt<=strip) act+=DDS_readbits(bits)-(1<<bits)/2;
         else act+=*(ptr2-strip)-*(ptr2-strip-1)+DDS_readbits(bits)-(1<<bits)/2;

         while (act<0) act+=256;
         while (act>255) act-=256;

         if ((cnt&(DDS_BLOCKSIZE-1))==0)
            if (ptr1==NULL)
               {
               if ((ptr1=(unsigned char *)malloc(DDS_BLOCKSIZE))==NULL) ERRORMSG();
               ptr2=ptr1;
               }
            else
               {
               if ((ptr1=(unsigned char *)realloc(ptr1,cnt+DDS_BLOCKSIZE))==NULL) ERRORMSG();
               ptr2=&ptr1[cnt];
               }

         *ptr2++=act;
         cnt++;
         }
      }

   if (ptr1!=NULL)
      if ((ptr1=(unsigned char *)realloc(ptr1,cnt))==NULL) ERRORMSG();

   DDS_interleave(ptr1,cnt,skip,block);

   *data=ptr1;
   *bytes=cnt;
   }

// read from a RAW file
unsigned char *readRAWfiled(FILE *file,unsigned int *bytes)
   {
   unsigned char *data;
   unsigned int cnt,blkcnt;

   data=NULL;
   cnt=0;

   do
      {
      if (data==NULL)
         {if ((data=(unsigned char *)malloc(DDS_BLOCKSIZE))==NULL) ERRORMSG();}
      else
         if ((data=(unsigned char *)realloc(data,cnt+DDS_BLOCKSIZE))==NULL) ERRORMSG();

      blkcnt=fread(&data[cnt],1,DDS_BLOCKSIZE,file);
      cnt+=blkcnt;
      }
   while (blkcnt==DDS_BLOCKSIZE);

   if (cnt==0)
      {
      free(data);
      return(NULL);
      }

   if ((data=(unsigned char *)realloc(data,cnt))==NULL) ERRORMSG();

   *bytes=cnt;

   return(data);
   }

// read a RAW file
unsigned char *readRAWfile(const char *filename,unsigned int *bytes)
   {
   FILE *file;

   unsigned char *data;

   if ((file=fopen(filename,"rb"))==NULL) return(NULL);

   data=readRAWfiled(file,bytes);

   fclose(file);

   return(data);
   }

// read a Differential Data Stream
unsigned char *readDDSfile(const char *filename,unsigned int *bytes)
   {
   int version=1;

   FILE *file;

   int cnt;

   unsigned char *chunk,*data;
   unsigned int size;

   if ((file=fopen(filename,"rb"))==NULL) return(NULL);

   for (cnt=0; DDS_ID[cnt]!='\0'; cnt++)
      if (fgetc(file)!=DDS_ID[cnt])
         {
         fclose(file);
         version=0;
         break;
         }

   if (version==0)
      {
      if ((file=fopen(filename,"rb"))==NULL) return(NULL);

      for (cnt=0; DDS_ID2[cnt]!='\0'; cnt++)
         if (fgetc(file)!=DDS_ID2[cnt])
            {
            fclose(file);
            return(NULL);
            }

      version=2;
      }

   if ((chunk=readRAWfiled(file,&size))==NULL) ERRORMSG();

   fclose(file);

   DDS_decode(chunk,size,&data,bytes,version==1?0:DDS_INTERLEAVE);

   free(chunk);

   return(data);
   }

// read a compressed PVM volume
unsigned char *readPVMvolume(const char *filename,
                             unsigned int *width,unsigned int *height,unsigned int *depth,unsigned int *components,
                             float *scalex,float *scaley,float *scalez,
                             unsigned char **description,
                             unsigned char **courtesy,
                             unsigned char **parameter,
                             unsigned char **comment)
   {
   unsigned char *data,*ptr;
   unsigned int bytes,numc;

   int version=1;

   unsigned char *volume;

   float sx=1.0f,sy=1.0f,sz=1.0f;

   unsigned int len1=0,len2=0,len3=0,len4=0;

   if ((data=readDDSfile(filename,&bytes))==NULL)
      if ((data=readRAWfile(filename,&bytes))==NULL) return(NULL);

   if (bytes<5) return(NULL);

   if ((data=(unsigned char *)realloc(data,bytes+1))==NULL) ERRORMSG();
   data[bytes]='\0';

   if (strncmp((char *)data,"PVM\n",4)!=0)
      {
      if (strncmp((char *)data,"PVM2\n",5)==0) version=2;
      else if (strncmp((char *)data,"PVM3\n",5)==0) version=3;
      else return(NULL);

      ptr=&data[5];
      if (sscanf((char *)ptr,"%d %d %d\n%g %g %g\n",width,height,depth,&sx,&sy,&sz)!=6) ERRORMSG();
      if (*width<1 || *height<1 || *depth<1 || sx<=0.0f || sy<=0.0f || sz<=0.0f) ERRORMSG();
      ptr=(unsigned char *)strchr((char *)ptr,'\n')+1;
      }
   else
      {
      ptr=&data[4];
      while (*ptr=='#')
         while (*ptr++!='\n');

      if (sscanf((char *)ptr,"%d %d %d\n",width,height,depth)!=3) ERRORMSG();
      if (*width<1 || *height<1 || *depth<1) ERRORMSG();
      }

   if (scalex!=NULL && scaley!=NULL && scalez!=NULL)
      {
      *scalex=sx;
      *scaley=sy;
      *scalez=sz;
      }

   ptr=(unsigned char *)strchr((char *)ptr,'\n')+1;
   if (sscanf((char *)ptr,"%d\n",&numc)!=1) ERRORMSG();
   if (numc<1) ERRORMSG();

   if (components!=NULL) *components=numc;
   else if (numc!=1) ERRORMSG();

   ptr=(unsigned char *)strchr((char *)ptr,'\n')+1;
   if (version==3) len1=strlen((char *)(ptr+(*width)*(*height)*(*depth)*numc))+1;
   if (version==3) len2=strlen((char *)(ptr+(*width)*(*height)*(*depth)*numc+len1))+1;
   if (version==3) len3=strlen((char *)(ptr+(*width)*(*height)*(*depth)*numc+len1+len2))+1;
   if (version==3) len4=strlen((char *)(ptr+(*width)*(*height)*(*depth)*numc+len1+len2+len3))+1;
   if ((volume=(unsigned char *)malloc((*width)*(*height)*(*depth)*numc+len1+len2+len3+len4))==NULL) ERRORMSG();
   if (data+bytes!=ptr+(*width)*(*height)*(*depth)*numc+len1+len2+len3+len4) ERRORMSG();

   memcpy(volume,ptr,(*width)*(*height)*(*depth)*numc+len1+len2+len3+len4);
   free(data);

   if (description!=NULL)
      if (len1>1) *description=volume+(*width)*(*height)*(*depth)*numc;
      else *description=NULL;

   if (courtesy!=NULL)
      if (len2>1) *courtesy=volume+(*width)*(*height)*(*depth)*numc+len1;
      else *courtesy=NULL;

   if (parameter!=NULL)
      if (len3>1) *parameter=volume+(*width)*(*height)*(*depth)*numc+len1+len2;
      else *parameter=NULL;

   if (comment!=NULL)
      if (len4>1) *comment=volume+(*width)*(*height)*(*depth)*numc+len1+len2+len3;
      else *comment=NULL;

   return(volume);
   }

// helper functions for quantize:

inline int DDS_get(unsigned short int *data,
                   unsigned int width,unsigned int height,unsigned int depth,
                   unsigned int i,unsigned int j,unsigned int k)
   {return(data[i+(j+k*height)*width]);}

inline double DDS_getgrad(unsigned short int *data,
                          unsigned int width,unsigned int height,unsigned int depth,
                          unsigned int i,unsigned int j,unsigned int k)
   {
   double gx,gy,gz;

   if (i>0)
      if (i<width-1) gx=(DDS_get(data,width,height,depth,i+1,j,k)-DDS_get(data,width,height,depth,i-1,j,k))/2.0;
      else gx=DDS_get(data,width,height,depth,i,j,k)-DDS_get(data,width,height,depth,i-1,j,k);
   else
      if (i<width-1) gx=DDS_get(data,width,height,depth,i+1,j,k)-DDS_get(data,width,height,depth,i,j,k);
      else gx=0.0;

   if (j>0)
      if (j<height-1) gy=(DDS_get(data,width,height,depth,i,j+1,k)-DDS_get(data,width,height,depth,i,j-1,k))/2.0;
      else gy=DDS_get(data,width,height,depth,i,j,k)-DDS_get(data,width,height,depth,i,j-1,k);
   else
      if (j<height-1) gy=DDS_get(data,width,height,depth,i,j+1,k)-DDS_get(data,width,height,depth,i,j,k);
      else gy=0.0;

   if (k>0)
      if (k<depth-1) gz=(DDS_get(data,width,height,depth,i,j,k+1)-DDS_get(data,width,height,depth,i,j,k-1))/2.0;
      else gz=DDS_get(data,width,height,depth,i,j,k)-DDS_get(data,width,height,depth,i,j,k-1);
   else
      if (k<depth-1) gz=DDS_get(data,width,height,depth,i,j,k+1)-DDS_get(data,width,height,depth,i,j,k);
      else gz=0.0;

   return(sqrt(gx*gx+gy*gy+gz*gz));
   }

// quantize 16 bit data to 8 bit using a non-linear mapping
unsigned char *quantize(unsigned char *data,
                        unsigned int width,unsigned int height,unsigned int depth,
                        BOOLINT linear,BOOLINT nofree)
   {
   unsigned int i,j,k;

   unsigned char *data2;
   unsigned short int *data3;

   int v,vmin,vmax;

   double *err,eint;

   BOOLINT done;

   if ((data3=(unsigned short int*)malloc(width*height*depth*sizeof(unsigned short int)))==NULL) ERRORMSG();

   vmin=65535;
   vmax=0;

   for (k=0; k<depth; k++)
      for (j=0; j<height; j++)
         for (i=0; i<width; i++)
            {
            v=256*data[2*(i+(j+k*height)*width)]+data[2*(i+(j+k*height)*width)+1];
            data3[i+(j+k*height)*width]=v;

            if (v<vmin) vmin=v;
            if (v>vmax) vmax=v;
            }

   if (!nofree) free(data);

   err=new double[65536];

   if (linear)
      for (i=0; i<65536; i++) err[i]=255*(double)i/vmax;
   else
      {
      for (i=0; i<65536; i++) err[i]=0.0;

      for (k=0; k<depth; k++)
         for (j=0; j<height; j++)
            for (i=0; i<width; i++)
               err[DDS_get(data3,width,height,depth,i,j,k)]+=sqrt(DDS_getgrad(data3,width,height,depth,i,j,k));

      for (i=0; i<65536; i++) err[i]=pow(err[i],1.0/3);

      err[vmin]=err[vmax]=0.0;

      for (k=0; k<256; k++)
         {
         for (eint=0.0,i=0; i<65536; i++) eint+=err[i];

         done=TRUE;

         for (i=0; i<65536; i++)
            if (err[i]>eint/256)
               {
               err[i]=eint/256;
               done=FALSE;
               }

         if (done) break;
         }

      for (i=1; i<65536; i++) err[i]+=err[i-1];

      if (err[65535]>0.0f)
         for (i=0; i<65536; i++) err[i]*=255.0f/err[65535];
      }

   if ((data2=(unsigned char *)malloc(width*height*depth))==NULL) ERRORMSG();

   for (k=0; k<depth; k++)
      for (j=0; j<height; j++)
         for (i=0; i<width; i++)
            data2[i+(j+k*height)*width]=(int)(err[DDS_get(data3,width,height,depth,i,j,k)]+0.5);

   delete err;
   free(data3);

   return(data2);
   }
