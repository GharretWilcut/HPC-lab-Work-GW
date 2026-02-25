// app.cpp - Main application file for VA-CPP DPU Host
#include <cstdio> 
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <unistd.h>
#include <getopt.h>
#include <cassert>


extern "C" {
    #include <dpu.h>
    #include <dpu_log.h>
    #if ENERGY
    #include <dpu_probe.h>
    #endif
}

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

static T*  A;
static T*  B;
static T* C;
static T* C2;

//create vectors
static void read_input (T* A, T* B, unsigned int size){
    srand(0);
    printf("size\t%u\t",size);
    for(unsigned int i = 0; i < size; i++){
        A[i] = (T)(rand());
        B[i] = (T)(rand());
    }
}

static void vector_addition_host(T* C, T* A, T* B, unsigned int size){
    for(unsigned int i = 0; i < size; i++){
        C[i] = A[i] + B[i];
    }
}

int main(int argc,char **argv){

    Params params = input_params(argc, argv);

    dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;

    #if ENERGY
        dpu_probe_t probe;
        DPU_ASSERT(dpu_probe_init("eng probe", &probe));
    #endif

    DPU_ASSERT(dpu_alloc(NR_DPUS,NULL,&dpu_set));
    DPU_ASSERT(dpu_load(dpu_set,DPU_BINARY,NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set,&nr_of_dpus));
    printf("Number of DPUs: %u\n",nr_of_dpus);

    const unsigned int input_size = (params.exp == 0) ? params.input_size * nr_of_dpus : params.input_size;
    const unsigned int input_size_8bytes =
        ((input_size * sizeof(T)) % 8 != 0)
            ? roundup(input_size, 8)
            : input_size;
    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus);
    const unsigned int input_size_dpu_8bytes = ((input_size_dpu * sizeof(T)) % 8 != 0)
        ? roundup(input_size_dpu, 8)
        : input_size_dpu;
    
    A = static_cast<T*>(malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T)));
    B = static_cast<T*>(malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T)));
    C = static_cast<T*>(malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T)));
    C2 = static_cast<T*>(malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T)));

    T* bufferA = A;
    T* bufferB = B;
    T* bufferC = C;

    read_input(A,B,input_size);

    Timer timer;

    printf("NR_TASKLETS\t%u\t",NR_TASKLETS);

    for (int rep = 0; rep < params.n_warmup + params.n_reps; rep++) {

        const bool do_timing = (rep >= params.n_warmup);

        if (do_timing)
            start(&timer, 0, rep - params.n_warmup);

        vector_addition_host(C, A, B, input_size);

        if (do_timing)
            stop(&timer, 0);

        if (do_timing)
            start(&timer, 1, rep - params.n_warmup);

        auto kernel = dpu_arguments_t::kernel1;
        dpu_arguments_t input_arguments[NR_DPUS];

        for (unsigned int i = 0; i < nr_of_dpus; i++){
            input_arguments[i].size = input_size_dpu_8bytes * sizeof(T);
            input_arguments[i].transfer_size = input_size_dpu_8bytes * sizeof(T);
            input_arguments[i].kernel = kernel;
        }

        input_arguments[nr_of_dpus - 1].size =
            (input_size_8bytes - input_size_dpu_8bytes * (NR_DPUS - 1)) * sizeof(T);
        input_arguments[nr_of_dpus - 1].transfer_size =
            (input_size_8bytes * sizeof(T));
        input_arguments[nr_of_dpus - 1].kernel = kernel;

        unsigned int i = 0;
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,sizeof(input_arguments[0]),DPU_XFER_DEFAULT));
        i=0;
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferA + input_size_dpu_8bytes * i));
        }

        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));

        i=0;
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferB + input_size_dpu_8bytes * i));
        }

        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T), input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));

        if (rep >= params.n_warmup)
            stop(&timer, 1);

        if (rep >= params.n_warmup) {
            start(&timer, 2, rep - params.n_warmup);
        #if ENERGY
                    DPU_ASSERT(dpu_probe_start(&probe));
        #endif
        }

        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

        if (rep >= params.n_warmup) {
            stop(&timer, 2);
        #if ENERGY
                    DPU_ASSERT(dpu_probe_stop(&probe));
        #endif
        }
        
        if (rep >= params.n_warmup) 
            start (&timer,3,rep - params.n_warmup);
        
        i = 0;
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferC + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T), input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));

        if(rep >= params.n_warmup)
            stop(&timer,3);
    } 

    printf("CPU ");
    print(&timer, 0, params.n_reps);
    printf("CPU-DPU ");
    print(&timer, 1, params.n_reps);
    printf("DPU Kernel ");
    print(&timer, 2, params.n_reps);
    printf("DPU-CPU ");
    print(&timer, 3, params.n_reps);

    #if ENERGY
        double energy;
        DPU_ASSERT(dpu_probe_get(&probe,DPU_ENERGY,DPU_AVERAGE, &energy));
        printf("DPU Energy (J): %f\t", energy);
    #endif

    bool status = true;
    for (unsigned int i = 0; i < input_size; i++){
        if (C[i] != bufferC[i]){
            status = false;
        }
    }

    printf("[%s] Outputs %s\n",
           status ? "OK" : "ERROR",
           status ? "are equal" : "differ");

    free(A);
    free(B);
    free(C);
    free(C2);
    DPU_ASSERT(dpu_free(dpu_set));
    return status ? 0 : 1;
}

