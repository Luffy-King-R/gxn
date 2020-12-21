/* 
nvcc cudaMFLOPS1.cu -I ~/NVIDIA_GPU_Computing_SDK/C/common/inc -I ~/NVIDIA_GPU_Computing_SDK/shared/inc cpuida64.o cpuidc64.o -o cudamflops

nvcc cudaMFLOPS1.cu -I ~/NVIDIA_GPU_Computing_SDK/C/common/inc -I ~/NVIDIA_GPU_Computing_SDK/shared/inc cpuida32.o cpuidc32.o -o cudamflops

*/


 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <ctype.h>
 #include "cpuidhn.h"
 #include <cuda.h>
 #include <cutil.h>
 #include <cutil_inline.h>
 #include <shrUtils.h>


   #define SPDP double
   #define Version "x86 64 Bits DP MFLOPS Benchmark 1.4"
//   #define Version "x64 64 Bits DP MFLOPS Benchmark 1.4"

//   #define SPDP float
//   #define Version "x86 32 Bits SP MFLOPS Benchmark 1.4"
//    #define Version "x64 32 Bits SP MFLOPS Benchmark 1.4"

   #define CudaV "Linux CUDA 3.2"

typedef int         BOOL;
#define TRUE            1
#define FALSE           0

 
 FILE    *outfile;
 int     endit;
 int     test;
 int     part;
 int     opwd;

 SPDP   *x_cpu;                  // Pointer to CPU arrays
 SPDP   *x_gpu;                  // Pointer to GPU  array
 size_t  size_x;
 int     words     = 100000;      // E Number of words in arrays
 int     repeats   = 2500;        // R Number of repeat passes 
 int     threads = 256;           // threads per block
 int     sharedMemSize;
 int     threadsPerBlock = 256;
 int     numBlocks;
 int     minutes = 0;
 int     reportSecs = 15;
 int     relFunc = 7;
 size_t freeVRAM;
 size_t totalVRAM;
 unsigned int maxVRAM = 0;
 unsigned int minVRAM = 4294967295;

 BOOL    relTest = FALSE;
 SPDP   xval = 0.999950f;
 SPDP   aval = 0.000020f;
 SPDP   bval = 0.999980f;
 SPDP   cval = 0.000011f;
 SPDP   dval = 1.000011f;
 SPDP   eval = 0.000012f;
 SPDP   fval = 0.999992f;
 SPDP   gval = 0.000013f;
 SPDP   hval = 1.000013f;
 SPDP   jval = 0.000014f;
 SPDP   kval = 0.999994f;
 SPDP   lval = 0.000015f;
 SPDP   mval = 1.000015f;
 SPDP   oval = 0.000016f;
 SPDP   pval = 0.999996f;
 SPDP   qval = 0.000017f;
 SPDP   rval = 1.000017f;
 SPDP   sval = 0.000018f;
 SPDP   tval = 1.000018f;
 SPDP   uval = 0.000019f;
 SPDP   vval = 1.000019f;
 SPDP   wval = 0.000021f;
 SPDP   yval = 1.000021f;


 // Kernels that executes on the CUDA device
 __global__ void calc32(int n, SPDP a, SPDP b, SPDP c, SPDP d, SPDP e, SPDP f, SPDP g, SPDP h, SPDP j, SPDP k, SPDP l, SPDP m, SPDP o, SPDP p, SPDP q, SPDP r, SPDP s, SPDP t, SPDP u, SPDP v, SPDP w, SPDP y, SPDP *x)
 {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if( i<n ) x[i] = (x[i]+a)*b-(x[i]+c)*d+(x[i]+e)*f-(x[i]+g)*h+(x[i]+j)*k-(x[i]+l)*m+(x[i]+o)*p-(x[i]+q)*r+(x[i]+s)*t-(x[i]+u)*v+(x[i]+w)*y;
 } 

 __global__ void calc8(int n, SPDP a, SPDP b, SPDP c, SPDP d, SPDP e, SPDP f, SPDP *x)
 {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if( i<n ) x[i] = (x[i]+a)*b-(x[i]+c)*d+(x[i]+e)*f;
 }

  __global__ void calc2(int n, SPDP a, SPDP b, SPDP *x)
 {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if( i<n ) x[i] = (x[i]+a)*b;
 }

 __global__ void calc32r(int n, int rep, SPDP a, SPDP b, SPDP c, SPDP d, SPDP e, SPDP f, SPDP g, SPDP h, SPDP j, SPDP k, SPDP l, SPDP m, SPDP o, SPDP p, SPDP q, SPDP r, SPDP s, SPDP t, SPDP u, SPDP v, SPDP w, SPDP y, SPDP *x)
 {
     for (int rr=0; rr<rep; rr++)
     {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if( i<n ) x[i] = (x[i]+a)*b-(x[i]+c)*d+(x[i]+e)*f-(x[i]+g)*h+(x[i]+j)*k-(x[i]+l)*m+(x[i]+o)*p-(x[i]+q)*r+(x[i]+s)*t-(x[i]+u)*v+(x[i]+w)*y;
     }
 } 
 __global__ void calc8r(int n, int rep, SPDP a, SPDP b, SPDP c, SPDP d, SPDP e, SPDP f, SPDP *x)
 {
     for (int rr=0; rr<rep; rr++)
     {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if( i<n ) x[i] = (x[i]+a)*b-(x[i]+c)*d+(x[i]+e)*f;
     }
 }

  __global__ void calc2r(int n, int rep, SPDP a, SPDP b, SPDP *x)
 {
     for (int rr=0; rr<rep; rr++)
     {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if( i<n ) x[i] = (x[i]+a)*b;
     }
 }

__global__ void sharecalc(SPDP *xi, int rep, SPDP a, SPDP b, SPDP c, SPDP d, SPDP e, SPDP f, SPDP g, SPDP h, SPDP j, SPDP k, SPDP l, SPDP m, SPDP o, SPDP p, SPDP q, SPDP r, SPDP s, SPDP t, SPDP u, SPDP v, SPDP w, SPDP y)
{
    extern __shared__ SPDP x[];

    int inOffset  = blockDim.x * blockIdx.x;
    int in  = inOffset + threadIdx.x;

    // Load one element per thread from device memory to temporary shared memory
    x[threadIdx.x] = xi[in];

    // Block until all threads in the block have written data to shared memory
    __syncthreads();

  int i = threadIdx.x;
  for (int rr=0; rr<rep; rr++)
  {
    x[i] = (x[i]+a)*b-(x[i]+c)*d+(x[i]+e)*f-(x[i]+g)*h+(x[i]+j)*k-(x[i]+l)*m+(x[i]+o)*p-(x[i]+q)*r+(x[i]+s)*t-(x[i]+u)*v+(x[i]+w)*y;
  }

  // write the data from shared memory to device memory 
  xi[in] = x[threadIdx.x];
}

__global__ void sharecalc3(SPDP *xi, int rep, SPDP a, SPDP b, SPDP c, SPDP d, SPDP e, SPDP f)
{
    extern __shared__ SPDP x[];

    int inOffset  = blockDim.x * blockIdx.x;
    int in  = inOffset + threadIdx.x;

    // Load one element per thread from device memory to temporary shared memory
    x[threadIdx.x] = xi[in];

    // Block until all threads in the block have written data to shared memory
    __syncthreads();

  int i = threadIdx.x;
  for (int rr=0; rr<rep; rr++)
  {
    x[i] = (x[i]+a)*b-(x[i]+c)*d+(x[i]+e)*f;
  }

  // write the data from shared memory to device memory 
  xi[in] = x[threadIdx.x];
}

__global__ void sharecalc2(SPDP *xi, int rep, SPDP a, SPDP b)
{
    extern __shared__ SPDP x[];

    int inOffset  = blockDim.x * blockIdx.x;
    int in  = inOffset + threadIdx.x;

    // Load one element per thread from device memory to temporary shared memory
    x[threadIdx.x] = xi[in];

    // Block until all threads in the block have written data to shared memory
    __syncthreads();

  int i = threadIdx.x;
  for (int rr=0; rr<rep; rr++)
  {
    x[i] = (x[i]+a)*b;
  }

  // write the data from shared memory to device memory 
  xi[in] = x[threadIdx.x];
  
}

 void runTests()
 {
    int  i;
    unsigned blocks = (words + threads-1) / threads;       // blocks needed

    if (test == 3)
    {
       calc2r<<<blocks, threads>>>(words, repeats, aval, xval, x_gpu);
       opwd = 2;

       cutilSafeCall( cudaThreadSynchronize() );
    
       // check if kernel execution generated an error
       cutilCheckMsg("Kernel execution failed");
    }
    else if (test == 4)
    {
       sharecalc2<<< numBlocks, threadsPerBlock, sharedMemSize >>>(x_gpu, repeats,  aval, xval);
       opwd = 2;

       cutilSafeCall( cudaThreadSynchronize() );
    
       // check if kernel execution generated an error
       cutilCheckMsg("Kernel execution failed");
    }
    else if (test == 5)
    {
       calc8r<<<blocks, threads>>>(words, repeats, aval, bval, cval, dval, eval, fval,  x_gpu);
       opwd = 8;

       cutilSafeCall( cudaThreadSynchronize() );
    
       // check if kernel execution generated an error
       cutilCheckMsg("Kernel execution failed");
    }
    else if (test == 6)
    {
       sharecalc3<<< numBlocks, threadsPerBlock, sharedMemSize >>>(x_gpu, repeats,  aval, bval, cval, dval, eval, fval);
       opwd = 8;

       cutilSafeCall( cudaThreadSynchronize() );
    
       // check if kernel execution generated an error
       cutilCheckMsg("Kernel execution failed");
    }
    else if (test == 7)
    {
       calc32r<<<blocks, threads>>>(words, repeats, aval, bval, cval, dval, eval, fval, gval, hval, jval, kval, lval, mval, oval, pval, qval, rval, sval, tval, uval, vval, wval, yval,  x_gpu);
       opwd = 32;

       cutilSafeCall( cudaThreadSynchronize() );
    
       // check if kernel execution generated an error
       cutilCheckMsg("Kernel execution failed");
    }
    else if (test == 8)
    {
       sharecalc<<< numBlocks, threadsPerBlock, sharedMemSize >>>(x_gpu, repeats,  aval, bval, cval, dval, eval, fval, gval, hval, jval, kval, lval, mval, oval, pval, qval, rval, sval, tval, uval, vval, wval, yval);
       opwd = 32;

       cutilSafeCall( cudaThreadSynchronize() );
    
       // check if kernel execution generated an error
       cutilCheckMsg("Kernel execution failed");
    }
    else
    {
        for (i=0; i<repeats; i++)
        {
           // Copy data to GPU
           if (test == 0)
           {
              cudaMemcpy(x_gpu, x_cpu, size_x, cudaMemcpyHostToDevice);
           }
           // calculations in GPU
           if (part == 0)
           {
              calc2<<<blocks, threads>>>(words, aval, xval, x_gpu);
              opwd = 2;
           }
           if (part == 1)
           {
              calc8<<<blocks, threads>>>(words, aval, bval, cval, dval, eval, fval, x_gpu);
              opwd = 8;
           }
           if (part == 2)
           {
              calc32<<<blocks, threads>>>(words, aval, bval, cval, dval, eval, fval, gval, hval, jval, kval, lval, mval, oval, pval, qval, rval, sval, tval, uval, vval, wval, yval,  x_gpu);
              opwd = 32;
           }   
           cutilSafeCall( cudaThreadSynchronize() );
    
           // check if kernel execution generated an error
           cutilCheckMsg("Kernel execution failed");
    
           if (test < 2)
           {
              // Copy results to CPU
              cudaMemcpy(x_cpu, x_gpu, size_x, cudaMemcpyDeviceToHost);
           }
        }
    }    
 }


 // main program that executes in the CPU
 int main(int argc, char *argv[])
 {
         
    int     i, g, p;
    int     param1;
    SPDP   fpmops;
    SPDP   mflops;
    char    title[9][15];
    char    relErrors[40];
    SPDP   errors[2][10];
    int     erdata[5][10];
    int     count1 = 0;
    int     count2 = 0;
    int     isok0 = 0;
    int     isok1 = 0;
    int     isok2 = 0;
    SPDP   newdata = 0.999999f;
    SPDP   answer = 0;
    int     partStart = 0;
    int     partEnd   = 3;
    int     pStart    = 0;
    int     pEnd      = 3;
    int     testStart = 0;
    int     testEnd   = 3;
    int     wordsIn = 0;
    int     wordsmult = 100;
    int     repeatsIn = 0;
    int     wsize = sizeof(SPDP);
        
    strcpy(title[0], "Data in & out ");
    strcpy(title[1], "Data out only ");
    strcpy(title[2], "Calculate only");
    strcpy(title[3], "Calculate     ");
    strcpy(title[4], "Shared Memory ");
    strcpy(title[5], "Calculate     ");
    strcpy(title[6], "Shared Memory ");
    strcpy(title[7], "Calculate     ");
    strcpy(title[8], "Shared Memory ");


    for (i=1; i<9; i=i+2)
    {
       if (argc > i)
       {
          switch (toupper(argv[i][0]))
          {
                case 'T':
                param1 = 0;
                if (argc > i+1)
                {
                   sscanf(argv[i+1],"%d", &param1);
                   if (param1 > 0)
                   {
                      threads = param1;
                   }
                }
                break;

               case 'W':
                param1 = 0;
                if (argc > i+1)
                {
                   sscanf(argv[i+1],"%d", &param1);
                   if (param1 > 0) wordsIn = param1;
                }
                break;

                case 'R':
                param1 = 0;
                if (argc > i+1)
                {
                   sscanf(argv[i+1],"%d", &param1);
                   if (param1 > 0) repeatsIn = param1;
                   if (repeatsIn < 100) repeatsIn = 100;
                }
                break;

               case 'M':
                param1 = 0;
                relTest = TRUE;
                minutes = 1;
                if (argc > i+1)
                {
                   sscanf(argv[i+1],"%d", &param1);
                   if (param1 > 0) minutes = param1;
                }
                break;

               case 'S':
                param1 = 0;
                if (argc > i+1)
                {
                   sscanf(argv[i+1],"%d", &param1);
                   if (param1 > 0) reportSecs = param1;
                }
                break;
          }
       }
    }
    for (i=1; i<argc; i++)
    {
       if (strcmp(argv[i],  "FC") == 0)
       {
          relFunc = 8;
       }
    }
    getDetails();    
    local_time();
    outfile = fopen("CudaLog.txt","a+");
    if (outfile == NULL)
    {
        printf (" Cannot open results file \n\n");
        printf(" Press Enter\n\n");
        g = getchar();
        exit(0);
    }
    printf("\n");
    for (i=1; i<10; i++)
    {
        printf("%s\n", configdata[i]);
    }
    printf("\n");
    fprintf (outfile, " #####################################################\n\n");                     
    for (i=1; i<10; i++)
    {
        fprintf(outfile, "%s \n", configdata[i]);
    }
    fprintf (outfile, "\n");
 
    printf(" ##########################################\n"); 
    fprintf (outfile, " #####################################################\n\n");                     
    printf("\n  %s %s %s\n\n", CudaV, Version, timeday);
    fprintf(outfile, "  %s %s %s\n", CudaV, Version, timeday);
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if( deviceCount == 0 )
    {
        printf("  No CUDA devices found \n\n");
    }
    else
    {
        if (threads > 512)
        {
           printf("  Error %d threads specified, reduced to 512\n", threads);
           fprintf(outfile, "  Error %d threads specified, reduced to 512\n", threads);
           threads = 512;
        }
        threadsPerBlock = threads;
        if (relTest)
        {
           wordsmult = 1;
           repeats = 2;
           if (wordsIn == 0) words = words * 100;
        }
        else
        {
           if (repeatsIn > 0) repeats = repeatsIn;
        }
        if (wordsIn > 0)
        {
           words = wordsIn;
        }
        else
        {
           wordsIn = words;
        }
        if (words * wordsmult / threads > 65535)
        {
           words = threads * 65535 / wordsmult;
           printf("  Error %d words too high for 65535 blocksize, reduced to %d\n", wordsIn, words);
           fprintf(outfile, "  Error %d words too high for 65535 blocksize, reduced to %d\n", wordsIn, words);
           
        } 

        int  startWords = words;
        int  startRepeats = repeats;
    
        printf("  CUDA devices found \n");
        fprintf(outfile, "  CUDA devices found \n");
//        for (i=0; i<deviceCount; i++)
        i =0;
        {
           cudaDeviceProp deviceProp;
           cudaGetDeviceProperties(&deviceProp, i);
           int processors = deviceProp.multiProcessorCount;

           printf ("  Device %d: %s  with %d Processors %d cores \n", i, deviceProp.name, processors, processors*8);
           fprintf(outfile, "  Device %d: %s  with %d Processors %d cores \n", i, deviceProp.name, processors, processors*8);

           int tgm = (int)((SPDP)deviceProp.totalGlobalMem/1048576/1.024);
           int smpb = deviceProp.sharedMemPerBlock;
           printf ("  Global Memory %d MB, Shared Memory/Block %d B, Max Threads/Block %d\n", tgm, smpb, deviceProp.maxThreadsPerBlock);
           fprintf (outfile, "  Global Memory %d MB, Shared Memory/Block %d B, Max Threads/Block %d\n", tgm, smpb, deviceProp.maxThreadsPerBlock);
        }
        printf("\n  Using %d Threads\n", threads);
        fprintf(outfile, "\n  Using %d Threads\n", threads);

        if (relTest)
        {
           if ((SPDP)reportSecs / 60.0 > (SPDP)minutes) reportSecs = minutes * 60;
           testStart = relFunc;
           testEnd   = testStart + 1;
           pStart = 2;
           pEnd   = 3;
           partStart = 2;
           partEnd   = 3;
           printf("\n  %s Reliability Test %d minutes, report every %d seconds\n", title[testStart], minutes, reportSecs);

           fprintf(outfile, "\n  %s Reliability Test %d minutes, report every %d seconds\n", title[testStart], minutes, reportSecs);
        }
        else
        {
           fprintf(outfile, "\n  Test            %d Byte  Ops  Repeat   Seconds   MFLOPS             First  All\n", wsize);
           fprintf(outfile,   "                   Words  /Wd  Passes                              Results Same\n\n");
                     printf("\n  Test            %d Byte  Ops  Repeat   Seconds   MFLOPS             First  All\n", wsize);
                       printf("                   Words  /Wd  Passes                              Results Same\n\n");
        }
        for (part=partStart; part<partEnd; part++)
        {           
            words = startWords;
            repeats = startRepeats;
            int  mult;
            int  words2;

            for (p=pStart; p<pEnd; p++)
            {
               mult   = (words + threads - 1) / threads;
               words2 = mult * threads;
               size_x = words2 * sizeof(SPDP);
               numBlocks = words2 / threadsPerBlock;
               sharedMemSize = threadsPerBlock * sizeof(SPDP);
               
    
               // Allocate arrays for host CPU
               x_cpu = (SPDP *)malloc(size_x);
    
  
               // Allocate array for GPU
               cudaMalloc((void **) &x_gpu, size_x);
  
               if (part == 2 && p == 2 && partStart == 0) testEnd = 9;  

               for (test=testStart; test<testEnd; test++)
               {
                  // Data for array
                  for (i=0; i<words; i++)
                  {
                     x_cpu[i] = newdata;
                  }

                  // Copy data to GPU for clean start
                  cudaMemcpy(x_gpu, x_cpu, size_x, cudaMemcpyHostToDevice);

                  if (!relTest)
                  {
                     start_time();
                     runTests();
                     end_time();
                     fpmops = (SPDP)words * (SPDP)opwd;
                     mflops = (SPDP)repeats * fpmops / 1000000.0f / (SPDP)secs;
    
                     // Copy results to CPU after timing for calculate only
                     cudaMemcpy(x_cpu, x_gpu, size_x, cudaMemcpyDeviceToHost);

                     // Print results
                     fprintf(outfile, "%15s %8d %4d %7d %9.6f %8.0f", title[test], words, opwd, repeats, secs, mflops);
                               printf("%15s %8d %4d %7d %9.6f %8.0f", title[test], words, opwd, repeats, secs, mflops);

   // Test error output
   // if (test == 0 && p == 0 && part == 0) x_cpu[99] = 0.99999880790799;
   // if (test == 1 && p == 1 && part == 1) x_cpu[77] = 1.99999880791234;
   // if (test == 8 && p == 2 && part == 2) x_cpu[55] = 2.11111;
   // if (test == 0 && p == 0 && part == 0) x_cpu[0] = newdata;
                     isok1  = 0;
                     SPDP one = x_cpu[0];
                     if (one == newdata)
                     {
                        isok0 = 1;
                        isok1 = 1;
                     }
                     for (i=1; i<words; i++)
                     {
                        if (one != x_cpu[i])
                        {
                           isok1 = 1;
                           if (count1 < 10)
                           {
                              errors[0][count1] = x_cpu[i];
                              errors[1][count1] = one;
                              erdata[0][count1] = i;
                              erdata[1][count1] = words;                          
                              erdata[2][count1] = opwd;
                              erdata[3][count1] = repeats;
                              erdata[4][count1] = test;
                              count1 = count1 + 1;
                           }
                        }
                     }
                     if (isok1 == 0)
                     {
                        fprintf(outfile, " %17.13f  Yes\n", x_cpu[0]);
                                  printf(" %17.13f  Yes\n", x_cpu[0]);
                     }
                     else
                     {
                        fprintf(outfile, "         See later   No\n");
                                  printf("           See log   No\n");
                     }
                     if (test == 2 && p == 2 && part == 2 && partStart == 0)
                     {
                        if (secs > 1.5)
                        {
                           fprintf(outfile, "\n Extra tests not run as 2 second timeout is possible\n\n");
                           printf("\n Extra tests not run as 2 second timeout is possible\n\n");
                           test = 9;          
                        }
                        else
                        {
                           fprintf(outfile, "\n Extra tests - loop in main CUDA Function\n\n");
                           printf("\n Extra tests - loop in main CUDA Function\n\n");
                        }
                     }
                     if (test == 4 || test == 6)
                     {
                        fprintf(outfile, "\n");                     
                        printf("\n");
                     }
                  }
                  else
                  {
                     for (int g=0; g<16; g++)
                     {
                        // Data for array
                        for (i=0; i<words; i++)
                        {
                           x_cpu[i] = newdata;
                        }
                        // Copy data to GPU for clean start
                        cudaMemcpy(x_gpu, x_cpu, size_x, cudaMemcpyHostToDevice);

                        start_time();
                        runTests();
                        end_time();
                        if (secs < 0.001)
                        {
                            repeats = repeats * 2;
                        }
                        else
                        {
                            g = 16;
                        }  
                     }
                     repeats = (int)((SPDP)repeats * 1.75f / (SPDP)secs);
                     if (repeats < 1)repeats = 1;
                     start_time();
                     runTests();
                     end_time();
           
                     int repeat2 = (int)((SPDP)reportSecs / (SPDP)secs);
                     printf("\n  Repeat CUDA %d times at %5.2f seconds. Repeat former %d times\n", repeats, secs, repeat2);
                     printf("  Tests - %d %d Byte Words, %d Operations Per Word, %d Repeat Passes\n", words, wsize, opwd, repeats*repeat2);          
                     printf("  Test Running ");
                     fprintf(outfile, "\n  Repeat CUDA %d times at %5.2f seconds. Repeat former %d times\n", repeats, secs, repeat2);
                     fprintf(outfile, "  Tests - %d %d Byte Words, %d Operations Per Word, %d Repeat Passes\n", words, wsize, opwd, repeats*repeat2);
                     int relPasses = (int)((SPDP)minutes * 60.0/ (SPDP)reportSecs);
                     if (relPasses < 1) relPasses = 1;
                     for (int rp=0; rp<relPasses; rp++)
                     { 
                        isok2  = 0;
                        count2 =0;
                        // Data for array
                        for (i=0; i<words; i++)
                        {
                           x_cpu[i] = newdata;
                        }

                        // Copy data to GPU for clean start
                        cudaMemcpy(x_gpu, x_cpu, size_x, cudaMemcpyHostToDevice);

                        start_time();
                        for (int rp2=0; rp2<repeat2; rp2++)
                        {
                           runTests();
                           printf(".");
                           fflush( stdout );
                        }
                        end_time();
                        fpmops = (SPDP)words * (SPDP)opwd;
                        mflops = (SPDP)repeats * (SPDP)repeat2 * fpmops / 1000000.0f / (SPDP)secs;
    
                        // Copy results to CPU after timing for calculate only
                        cudaMemcpy(x_cpu, x_gpu, size_x, cudaMemcpyDeviceToHost);

                        if (rp==0)
                        {
                           answer = x_cpu[0];
                           fprintf(outfile, "\n  Results of all calculations should be -    %18.16f\n", answer);
                           printf("\n\n  Results of all calculations should be -    %18.16f\n", answer);
                           fprintf(outfile, "\n  Test Seconds   MFLOPS    Errors     First               Value\n");
                           fprintf(outfile,   "                                       Word \n\n");
                                     printf("\n  Test Seconds   MFLOPS    Errors     First               Value\n");
                                     printf(  "                                       Word \n\n");
                        }
                        else
                        {
                             printf("\n");
                        }
                            
      // Error
      //  x_cpu[1] = x_cpu[1] + 1.0;
                    
                        for (i=0; i<words; i++)
                        {
                           if (answer != x_cpu[i])
                           {
                             if (isok2 == 0) isok2 = i;
                             count2 = count2 + 1;
                           }
                        }
                        if (isok2 == 0)
                        {
                            sprintf(relErrors, "  None found");
                        }
                        else
                        {
                           sprintf(relErrors, "%9d %9d %19.16f",  count2, isok2, x_cpu[isok2]);
                        }

                        // Print results
                        fprintf(outfile, "%4d %9.3f %8.0f %s", rp+1, secs, mflops, relErrors);
                                  printf("%4d %9.3f %8.0f %s", rp+1, secs, mflops, relErrors);
                        fprintf(outfile, "\n");
                        if (rp == relPasses - 1)
                        {
                           printf("\n");
                        }
                        else
                        {
                            printf("\n  Test Running ");
                        }
                     }
                  }
               }
/*
               cuInit(0);
               CUdevice dev;        
               CUcontext ctx;
               cuDeviceGet(&dev,0);       // use 1st CUDA device
               cuCtxCreate(&ctx, 0, dev); // create context for it
               CUresult memres = cuMemGetInfo(&freeVRAM, &totalVRAM);
               cuCtxDetach(ctx);
*/
  
               if (freeVRAM > maxVRAM) maxVRAM = freeVRAM;
               if (freeVRAM < minVRAM) minVRAM = freeVRAM;

               fprintf(outfile, "\n");
               printf("\n");
              // Cleanup
               free(x_cpu); 
               cudaFree(x_gpu); 
               words = words * 10;
               repeats = repeats / 10;
               if (repeats < 1) repeats = 1; 
           }
        }
    }

    fprintf(outfile,"\n");

    if( deviceCount == 0 )
    {
                fprintf(outfile, "  No CUDA devices found \n\n");
    }
    else
    {
/*
        totalVRAM = (int)((SPDP)totalVRAM/1048576/1.024);
        minVRAM = (int)((SPDP)minVRAM/1048576/1.024);
        maxVRAM = (int)((SPDP)maxVRAM/1048576/1.024);
        fprintf(outfile, "  %d MB Graphics RAM, Used %d Minimum, %d Maximum\n\n",
                  (int)totalVRAM, (int)(totalVRAM - maxVRAM), (int)(totalVRAM - minVRAM));
        printf("  %d MB Graphics RAM, Used %d Minimum, %d Maximum\n\n",
                  (int)totalVRAM, (int)(totalVRAM - maxVRAM), (int)(totalVRAM - minVRAM));
*/
     }
    if (isok0 > 0)
    {
       fprintf(outfile," ERROR - At least one first result of 0.999999 (DP %18.16f) - no calculations?\n\n", newdata);
       printf(" ERROR - At least one first result of 0.999999 - no calculations?\n\n");
    }
    if (count1 > 0)
    {
       fprintf(outfile," First Unexpected Results - not same as word 0\n");
       for (i=0; i<count1; i++)
       {
         fprintf(outfile,"%15s %8d %4d %7d word %9d was %18.16f not %18.16f\n",
           title[erdata[4][i]], erdata[1][i], erdata[2][i], erdata[3][i], erdata[0][i], errors[0][i], errors[1][i]);
       }
       fprintf(outfile,"\n");
    }
    fclose (outfile);
    printf(" Press Enter\n\n");
    g = getchar();
    if (g == 99) printf("%d\n\n", g);
    return 0;
 }


