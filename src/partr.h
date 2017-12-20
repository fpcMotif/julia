// This file is a part of Julia. License is MIT: https://julialang.org/license

/*  partr -- parallel tasks runtime options
 */

#ifndef PARTR_H
#define PARTR_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef JULIA_ENABLE_PARTR

#include "julia.h"


/* multiq */
#define MULTIQ_HEAP_C                   4
    /* number of heaps = MULTIQ_HEAP_C * nthreads */
#define MULTIQ_TASKS_PER_HEAP           129
    /* how many in each heap */

/* parfor */
#define GRAIN_K                         4
    /* tasks = niters / (GRAIN_K * nthreads) */

/* synchronization */
#define ARRIVERS_P                      2
    /* narrivers = ((GRAIN_K * nthreads) ^ ARRIVERS_P) + 1
       limit for number of recursive parfors */
#define REDUCERS_FRAC                   1
    /* nreducers = narrivers * REDUCERS_FRAC */


/* interface */
int  partr_spawn(jl_task_t **t, jl_value_t *_args, int8_t sticky, int8_t detach);
int  partr_sync(jl_value_t *r, jl_task_t *t);
int  partr_parfor(jl_task_t **t, jl_value_t *_args, int64_t count, jl_value_t *_rargs);
int  partr_yield();
int  partr_wait(jl_condition_t *c);
int  partr_notify(jl_condition_t *c);

#endif /* JULIA_ENABLE_PARTR */

#ifdef __cplusplus
}
#endif

#endif /* PARTR_H */

