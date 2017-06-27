// This file is a part of Julia. License is MIT: https://julialang.org/license

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "julia.h"
#include "julia_internal.h"
#include "threading.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef JULIA_ENABLE_THREADING
#ifdef JULIA_ENABLE_PARTR

// task states
extern jl_sym_t *done_sym;
extern jl_sym_t *failed_sym;
extern jl_sym_t *runnable_sym;

// task/stack switch functions used
extern void init_task_entry(void (*task_entry)(void), jl_task_t *t, char *stack);

// multiq
// ---

/* a task heap */
typedef struct taskheap_tag {
    jl_mutex_t lock;
    jl_task_t **tasks;
    int16_t ntasks, prio;
} taskheap_t;

/* multiqueue parameters */
static const int16_t heap_d = 8;
static const int heap_c = 4;

/* size of each heap */
static const int tasks_per_heap = 129;

/* the multiqueue's heaps */
static taskheap_t *heaps;
static int16_t heap_p;

/* unbias state for the RNG */
static uint64_t cong_unbias;


/*  multiq_init()
 */
static inline void multiq_init()
{
    heap_p = heap_c * jl_n_threads;
    heaps = (taskheap_t *)calloc(heap_p, sizeof(taskheap_t));
    for (int16_t i = 0;  i < heap_p;  ++i) {
        jl_mutex_init(&heaps[i].lock);
        heaps[i].tasks = (jl_task_t **)calloc(tasks_per_heap, sizeof(jl_task_t *));
        heaps[i].ntasks = 0;
        heaps[i].prio = INT16_MAX;
    }
    unbias_cong(heap_p, &cong_unbias);
}


/*  sift_up()
 */
static inline void sift_up(taskheap_t *heap, int16_t idx)
{
    if (idx > 0) {
        int16_t parent = (idx-1)/heap_d;
        if (heap->tasks[idx]->prio <= heap->tasks[parent]->prio) {
            jl_task_t *t = heap->tasks[parent];
            heap->tasks[parent] = heap->tasks[idx];
            heap->tasks[idx] = t;
            sift_up(heap, parent);
        }
    }
}


/*  sift_down()
 */
static inline void sift_down(taskheap_t *heap, int16_t idx)
{
    if (idx < heap->ntasks) {
        for (int16_t child = heap_d*idx + 1;
                child < tasks_per_heap && child <= heap_d*idx + heap_d;
                ++child) {
            if (heap->tasks[child]
                    &&  heap->tasks[child]->prio <= heap->tasks[idx]->prio) {
                jl_task_t *t = heap->tasks[idx];
                heap->tasks[idx] = heap->tasks[child];
                heap->tasks[child] = t;
                sift_down(heap, child);
            }
        }
    }
}


/*  multiq_insert()
 */
static inline int multiq_insert(jl_task_t *task, int16_t priority)
{
    jl_ptls_t ptls = jl_get_ptls_states();
    uint64_t rn;

    task->prio = priority;
    do {
        rn = cong(heap_p, cong_unbias, &ptls->rngseed);
    } while (!jl_mutex_trylock_nogc(&heaps[rn].lock));

    if (heaps[rn].ntasks >= tasks_per_heap) {
        jl_mutex_unlock_nogc(&heaps[rn].lock);
        return -1;
    }

    heaps[rn].tasks[heaps[rn].ntasks++] = task;
    sift_up(&heaps[rn], heaps[rn].ntasks-1);
    jl_mutex_unlock_nogc(&heaps[rn].lock);
    int16_t prio = jl_atomic_load(&heaps[rn].prio);
    if (task->prio < prio)
        jl_atomic_compare_exchange(&heaps[rn].prio, prio, task->prio);

    return 0;
}


/*  multiq_deletemin()
 */
static inline jl_task_t *multiq_deletemin()
{
    jl_ptls_t ptls = jl_get_ptls_states();
    uint64_t rn1, rn2;
    int16_t i, prio1, prio2;
    jl_task_t *task;

    for (i = 0;  i < jl_n_threads;  ++i) {
        rn1 = cong(heap_p, cong_unbias, &ptls->rngseed);
        rn2 = cong(heap_p, cong_unbias, &ptls->rngseed);
        prio1 = jl_atomic_load(&heaps[rn1].prio);
        prio2 = jl_atomic_load(&heaps[rn2].prio);
        if (prio1 > prio2) {
            prio1 = prio2;
            rn1 = rn2;
        }
        else if (prio1 == prio2 && prio1 == INT16_MAX)
            continue;
        if (jl_mutex_trylock_nogc(&heaps[rn1].lock)) {
            if (prio1 == heaps[rn1].prio)
                break;
            jl_mutex_unlock_nogc(&heaps[rn1].lock);
        }
    }
    if (i == jl_n_threads)
        return NULL;

    task = heaps[rn1].tasks[0];
    heaps[rn1].tasks[0] = heaps[rn1].tasks[--heaps[rn1].ntasks];
    heaps[rn1].tasks[heaps[rn1].ntasks] = NULL;
    prio1 = INT16_MAX;
    if (heaps[rn1].ntasks > 0) {
        sift_down(&heaps[rn1], 0);
        prio1 = heaps[rn1].tasks[0]->prio;
    }
    jl_atomic_store(&heaps[rn1].prio, prio1);
    jl_mutex_unlock_nogc(&heaps[rn1].lock);

    return task;
}


// sync trees
// ---

/* arrival tree */
struct _arriver_t {
    int16_t index, next_avail;
    int16_t **tree;
};

/* reduction tree */
struct _reducer_t {
    int16_t index, next_avail;
    jl_value_t ***tree;
};


/* pool of arrival trees */
static arriver_t *arriverpool;
static int16_t num_arrivers, num_arriver_tree_nodes, next_arriver;

/* pool of reduction trees */
static reducer_t *reducerpool;
static int16_t num_reducers, num_reducer_tree_nodes, next_reducer;


/*  synctreepool_init()
 */
static inline void synctreepool_init()
{
    num_arriver_tree_nodes = (GRAIN_K * jl_n_threads) - 1;
    num_reducer_tree_nodes = (2 * GRAIN_K * jl_n_threads) - 1;

    /* num_arrivers = ((GRAIN_K * jl_n_threads) ^ ARRIVERS_P) + 1 */
    num_arrivers = GRAIN_K * jl_n_threads;
    for (int i = 1;  i < ARRIVERS_P;  ++i)
        num_arrivers = num_arrivers * num_arrivers;
    ++num_arrivers;

    num_reducers = num_arrivers * REDUCERS_FRAC;

    /* allocate */
    arriverpool = (arriver_t *)calloc(num_arrivers, sizeof (arriver_t));
    next_arriver = 0;
    for (int i = 0;  i < num_arrivers;  ++i) {
        arriverpool[i].index = i;
        arriverpool[i].next_avail = i + 1;
        arriverpool[i].tree = (int16_t **)
                jl_malloc_aligned(num_arriver_tree_nodes * sizeof (int16_t *), 64);
        for (int j = 0;  j < num_arriver_tree_nodes;  ++j)
            arriverpool[i].tree[j] = (int16_t *)jl_malloc_aligned(sizeof (int16_t), 64);
    }
    arriverpool[num_arrivers - 1].next_avail = -1;

    reducerpool = (reducer_t *)calloc(num_reducers, sizeof (reducer_t));
    next_reducer = 0;
    for (int i = 0;  i < num_reducers;  ++i) {
        reducerpool[i].index = i;
        reducerpool[i].next_avail = i + 1;
        reducerpool[i].tree = (jl_value_t ***)
                jl_malloc_aligned(num_reducer_tree_nodes * sizeof (jl_value_t **), 64);
        for (int j = 0;  j < num_reducer_tree_nodes;  ++j)
            reducerpool[i].tree[j] = (jl_value_t **)jl_malloc_aligned(sizeof (jl_value_t *), 64);
    }
    if (num_reducers > 0)
        reducerpool[num_reducers - 1].next_avail = -1;
    else
        next_reducer = -1;
}


/*  arriver_alloc()
 */
static inline arriver_t *arriver_alloc()
{
    int16_t candidate;
    arriver_t *arr;

    do {
        candidate = jl_atomic_load(&next_arriver);
        if (candidate == -1)
            return NULL;
        arr = &arriverpool[candidate];
    } while (!jl_atomic_bool_compare_exchange(&next_arriver,
                candidate, arr->next_avail));
    return arr;
}


/*  arriver_free()
 */
static inline void arriver_free(arriver_t *arr)
{
    for (int i = 0;  i < num_arriver_tree_nodes;  ++i)
        *arr->tree[i] = 0;

    jl_atomic_exchange_generic(&next_arriver, &arr->index, &arr->next_avail);
}


/*  reducer_alloc()
 */
static inline reducer_t *reducer_alloc()
{
    int16_t candidate;
    reducer_t *red;

    do {
        candidate = jl_atomic_load(&next_reducer);
        if (candidate == -1)
            return NULL;
        red = &reducerpool[candidate];
    } while (!jl_atomic_bool_compare_exchange(&next_reducer,
                     candidate, red->next_avail));
    return red;
}


/*  reducer_free()
 */
static inline void reducer_free(reducer_t *red)
{
    for (int i = 0;  i < num_reducer_tree_nodes;  ++i)
        *red->tree[i] = 0;

    jl_atomic_exchange_generic(&next_reducer, &red->index, &red->next_avail);
}


/*  last_arriver()
 */
static inline int last_arriver(arriver_t *arr, int idx)
{
    int arrived, aidx = idx + (GRAIN_K * jl_n_threads) - 1;

    while (aidx > 0) {
        --aidx;
        aidx >>= 1;
        arrived = jl_atomic_fetch_add(arr->tree[aidx], 1);
        if (!arrived) return 0;
    }

    return 1;
}


/*  reduce()
 */
static inline jl_value_t *reduce(arriver_t *arr, reducer_t *red, jl_generic_fptr_t *fptr,
                                 jl_method_instance_t *mfunc, jl_value_t **args, uint32_t nargs,
                                 jl_value_t *val, int idx)
{
    int arrived, aidx = idx + (GRAIN_K * jl_n_threads) - 1, ridx = aidx, nidx;

    *red->tree[ridx] = val;
    while (aidx > 0) {
        --aidx;
        aidx >>= 1;
        arrived = jl_atomic_fetch_add(arr->tree[aidx], 1);
        if (!arrived) return NULL;

        /* neighbor has already arrived, get its value and reduce it */
        nidx = ridx & 0x1 ? ridx + 1 : ridx - 1;
        val = jl_thread_run_fun(fptr, mfunc, args, nargs);

        /* move up the tree */
        --ridx;
        ridx >>= 1;
        *red->tree[ridx] = val;
    }

    return val;
}


// parallel task runtime
// ---

// sticky task queues need to be visible to all threads
jl_taskq_t *sticky_taskqs;


// initialize the threading infrastructure
void jl_init_threadinginfra(void)
{
    /* initialize the synchronization trees pool and the multiqueue */
    synctreepool_init();
    multiq_init();

    /* allocate sticky task queues */
    sticky_taskqs = (jl_taskq_t *)jl_malloc_aligned(jl_n_threads * sizeof(jl_taskq_t), 64);
}


// initialize the thread function argument
void jl_init_threadarg(jl_threadarg_t *targ) { }


// helper for final thread initialization
static void init_started_thread()
{
    jl_ptls_t ptls = jl_get_ptls_states();

    /* allocate this thread's sticky task queue pointer and initialize the lock */
    seed_cong(&ptls->rngseed);
    ptls->sticky_taskq = &sticky_taskqs[ptls->tid];
    ptls->sticky_taskq->head = NULL;
}


// once the threads are started, perform any final initializations
void jl_init_started_threads(jl_threadarg_t **targs)
{
    // master thread final initialization
    init_started_thread();
}


static void run_next();


// thread function: used by all except the main thread
void jl_threadfun(void *arg)
{
    jl_threadarg_t *targ = (jl_threadarg_t *)arg;

    // initialize this thread (set tid, create heap, set up root task)
    jl_init_threadtls(targ->tid);
    jl_init_stack_limits(0);
    init_started_thread();
    jl_ptls_t ptls = jl_get_ptls_states();
    jl_init_root_task(ptls->stack_lo, ptls->stack_hi - ptls->stack_lo);

    // Assuming the functions called below don't contain unprotected GC
    // critical region. In general, the following part of this function
    // shouldn't call any managed code without calling `jl_gc_unsafe_enter`
    // first.
    jl_gc_state_set(ptls, JL_GC_STATE_SAFE, 0);
    uv_barrier_wait(targ->barrier);

    // free the thread argument here
    free(targ);

    // set a jump context for this root task
    jl_setjmp(ptls->current_task->ctx, 0);

    /* get the highest priority task and run it */
    run_next();
}


// add the specified task to the task queue
static void add_to_taskq(jl_task_t *task, jl_taskq_t *taskq)
{
    JL_LOCK(&taskq->lock);
    if (taskq->head == NULL)
        taskq->head = task;
    else {
        jl_task_t *pt = taskq->head;
        while (pt->next)
            pt = pt->next;
        pt->next = task;
    }
    JL_UNLOCK(&taskq->lock);
}


// pop the first task off the task queue
static jl_task_t *get_from_taskq(jl_taskq_t *taskq)
{
    /* racy check for quick path */
    if (taskq->head == NULL)
        return NULL;

    JL_LOCK(&taskq->lock);
    jl_task_t *task = taskq->head;
    if (task) {
        taskq->head = task->next;
        task->next = NULL;
    }
    JL_UNLOCK(&taskq->lock);

    return task;
}


// enqueue the specified task for execution
static void enqueue_task(jl_task_t *task)
{
    /* sticky tasks go to the thread's sticky queue */
    if (task->settings & TASK_IS_STICKY) {
        assert(task->sticky_tid != -1);
        add_to_taskq(task, &sticky_taskqs[task->sticky_tid]);
    }

    /* all others go back into the multiq */
    else
        multiq_insert(task, task->prio);
}


// add back all the tasks in the specified task's completion queue
static void add_back_task_cq(jl_task_t *task)
{
    JL_LOCK(&task->cq.lock);
    jl_task_t *cqtask = task->cq.head;
    task->cq.head = NULL;
    JL_UNLOCK(&task->cq.lock);

    jl_task_t *cqnext;
    while (cqtask) {
        cqnext = cqtask->next;
        cqtask->next = NULL;
        enqueue_task(cqtask);
        cqtask = cqnext;
    }
}


// parfor grains must synchronize/reduce as they end
static void sync_grains(jl_task_t *task)
{
    int was_last = 0;

    /* reduce... */
    if (task->red) {
        task->result = reduce(task->arr, task->red, task->rfptr, task->mredfunc,
                              task->rargs, task->nrargs, task->result, task->grain_num);

        /*  if this task is last, set the result in the parent task */
        if (task->result) {
            task->parent->red_result = task->result;
            was_last = 1;
        }
    }
    /* ... or just sync */
    else {
        if (last_arriver(task->arr, task->grain_num))
            was_last = 1;
    }

    /* the last task to finish needs to finish up the loop */
    if (was_last) {
        /* a non-parent task must wake up the parent */
        if (task->grain_num > 0)
            multiq_insert(task->parent, 0);

        /* this is the parent task which was last; it can just end */
        if (task->red)
            reducer_free(task->red);
        arriver_free(task->arr);
    }
    else {
        /* the parent task needs to wait */
        if (task->grain_num == 0)
            partr_yield(0);
    }
}


// all tasks except the root task start and exit here
static void NOINLINE JL_NORETURN task_wrapper()
{
    jl_ptls_t ptls = jl_get_ptls_states();
    jl_task_t *task = ptls->current_task;

    // GC safe
    uint32_t nargs;
    jl_value_t **args;
    if (!jl_is_svec(task->args)) {
        nargs = 1;
        args = &task->args;
    }
    else {
        nargs = jl_svec_len(task->args);
        args = jl_svec_data(task->args);
    }

    task->started = 1;
    JL_TRY {
        if (ptls->defer_signal) {
            ptls->defer_signal = 0;
            jl_sigint_safepoint(ptls);
        }
        JL_TIMING(ROOT);
        ptls->world_age = jl_world_counter;
        (void)jl_assume(fptr->jlcall_api != JL_API_CONST);
        task->result = jl_call_fptr_internal(&task->fptr, task->mfunc, args, nargs);
        jl_gc_wb(task, task->result);
        task->state = done_sym;
    }
    JL_CATCH {
        task->exception = ptls->exception_in_transit;
        jl_gc_wb(task, task->exception);
        task->state = failed_sym;
    }

    /* grain tasks must synchronize */
    if (task->grain_num >= 0)
        sync_grains(task);

    /* add back any tasks in this one's completion queue */
    if (!(task->settings & TASK_IS_DETACHED))
        add_back_task_cq(task);

    /* clear thread state */
    ptls->in_finalizer = 0;
    ptls->in_pure_callback = 0;
    ptls->world_age = jl_world_counter;

    /* next task */
    run_next();

    /* unreachable */
    gc_debug_critical_error();
    abort();
}


// get the next available task and run it
static void JL_NORETURN run_next()
{
    jl_ptls_t ptls = jl_get_ptls_states();
    jl_task_t *task = NULL;

    /* TODO: threads should sleep after spinning for some time */
    do {
        /* first check for sticky tasks */
        task = get_from_taskq(ptls->sticky_taskq);

        /* no sticky tasks, go to the multiq */
        if (!task) {
            task = multiq_deletemin();

            if (task) {
                /* a sticky task will only come out of the multiq if it has not been run */
                if (task->settings & TASK_IS_STICKY) {
                    assert(task->sticky_tid == -1);
                    task->sticky_tid = ptls->tid;
                }
            }
            else jl_cpu_pause();
        }
    } while (task == NULL);

    /* run/resume the task */
    ptls->pgcstack = task->gcstack;
    ptls->world_age = task->world_age;
    ptls->current_task = task;
    // TODO: set ptls->current_module correctly
    task->current_tid = ptls->tid;
    jl_longjmp(task->ctx, 1);

    /* unreachable */
    gc_debug_critical_error();
    abort();
}


// specialize and compile the user function
static int setup_task_fun(jl_value_t *_args, jl_value_t ***args, uint32_t *nargs,
                          jl_method_instance_t **mfunc, jl_generic_fptr_t *fptr)
{
    jl_ptls_t ptls = jl_get_ptls_states();

    if (!jl_is_svec(_args)) {
        *nargs = 1;
        *args = &_args;
    }
    else {
        *nargs = jl_svec_len(_args);
        *args = jl_svec_data(_args);
    }

    *mfunc = jl_lookup_generic(*args, *nargs,
                               jl_int32hash_fast(jl_return_address()), ptls->world_age);

    // Ignore constant return value for now.
    if (jl_compile_method_internal(fptr, *mfunc))
        return -1;

    return 0;
}


// allocate and initialize a task
static jl_task_t *new_task(jl_value_t *_args)
{
    jl_ptls_t ptls = jl_get_ptls_states();

    jl_task_t *task = (jl_task_t *)jl_gc_alloc(ptls, sizeof (jl_task_t),
                                               jl_task_type);
    if (setup_task_fun(_args, &task->args, &task->nargs, &task->mfunc, &task->fptr) != 0)
        return NULL;
    task->result = jl_nothing;

    // initialize elements
    task->next = NULL;
    task->storage = jl_nothing;
    task->state = runnable_sym;
    task->started = 0;
    task->consumers = jl_nothing;
    task->donenotify = jl_nothing;
    task->exception = jl_nothing;
    task->backtrace = jl_nothing;
    task->eh = NULL;
    arraylist_new(&task->locks, 0);
    task->gcstack = NULL;
    task->current_module = NULL;
    task->world_age = ptls->world_age;
    task->current_tid = -1;
    task->sticky_tid = -1;
    task->parent = NULL;
    task->arr = NULL;
    task->red = NULL;
    task->red_result = jl_nothing;
    task->grain_num = -1;
#ifdef ENABLE_TIMINGS
    task->timing_stack = NULL;
#endif

    // set up stack with guard page
    JL_GC_PUSH1(&task);
    task->ssize = LLT_ALIGN(1*1024*1024, jl_page_size);
    size_t stkbufsize = ssize + jl_page_size + (jl_page_size - 1);
    task->stkbuf = (void *)jl_gc_alloc_buf(ptls, stkbufsize);
    jl_gc_wb_buf(task, task->stkbuf, stkbufsize);
    char *stk = (char *)LLT_ALIGN((uintptr_t)task->stkbuf, jl_page_size);
    if (mprotect(stk, jl_page_size - 1, PROT_NONE) == -1)
        jl_errorf("mprotect: %s", strerror(errno));
    stk += jl_page_size;

    // set up entry point for this task
    init_task_entry(task_wrapper, task, stk);

    // for task cleanup
    jl_gc_add_finalizer((jl_value_t *)task, jl_unprotect_stack_func);

    JL_GC_POP();
    return task;
}


// allocate a task and copy the specified task's contents into it
static jl_task_t *copy_task(jl_task_t *ft)
{
    jl_ptls_t ptls = jl_get_ptls_states();

    jl_task_t *task = (jl_task_t *)jl_gc_alloc(ptls, sizeof (jl_task_t),
                                                 jl_task_type);
    memcpy(task, ft, sizeof (jl_task_t));
    return task;
}


/*  partr_spawn() -- create a task for `f(arg)` and enqueue it for execution

    Implicitly asserts that `f(arg)` can run concurrently with everything
    else that's currently running. If `detach` is set, the spawned task
    will not be returned (and cannot be synced). Yields.
 */
int partr_spawn(jl_task_t **t, jl_value_t *_args, int8_t sticky, int8_t detach)
{
    jl_ptls_t ptls = jl_get_ptls_states();

    jl_task_t *task = new_task(_args);
    if (task == NULL)
        return -1;
    if (sticky)
        task->settings |= TASK_IS_STICKY;
    if (detach)
        task->settings |= TASK_IS_DETACHED;

    if (multiq_insert(task, ptls->tid) != 0) {
        return -2;
    }

    *t = detach ? NULL : task;

    /* only yield if we're running a non-sticky task */
    if (!(ptls->current_task->settings & TASK_IS_STICKY))
        partr_yield(1);

    return 0;
}


/*  partr_sync() -- get the return value of task `t`

    Returns only when task `t` has completed.
 */
int partr_sync(jl_value_t *r, jl_task_t *task)
{
    jl_ptls_t ptls = jl_get_ptls_states();

    /* if the target task has not finished, add the current task to its
       completion queue; the thread that runs the target task will add
       this task back to the ready queue
     */
    if (task->state != done_sym  &&  task->state != failed_sym) {
        ptls->current_task->next = NULL;
        JL_LOCK(&task->cq.lock);

        /* ensure the task didn't finish before we got the lock */
        if (task->state != done_sym  &&  task->state != failed_sym) {
            /* add the current task to the CQ */
            if (task->cq.head == NULL)
                task->cq.head = ptls->current_task;
            else {
                jl_task_t *pt = task->cq.head;
                while (pt->next)
                    pt = pt->next;
                pt->next = ptls->current_task;
            }

            JL_UNLOCK(&task->cq.lock);

            partr_yield(0);
        }

        /* the task finished before we could add to its CQ */
        else
            JL_UNLOCK(&task->cq.lock);
    }

    if (r)
        *r = task->grain_num >= 0 && task->red ?
                task->red_result : task->result;
    return 0;
}


/*  partr_parfor() -- spawn multiple tasks for a parallel loop

    Spawn tasks that invoke `f(arg, start, end)` such that the sum of `end-start`
    for all tasks is `count`. Uses `rf()`, if provided, to reduce the return
    values from the tasks, and returns the result. Yields.
 */
int partr_parfor(jl_task_t **t, jl_value_t *_args, int64_t count, jl_value_t *_rargs)
{
    jl_ptls_t ptls = jl_get_ptls_states();

    int64_t n = GRAIN_K * jl_n_threads;
    lldiv_t each = lldiv(count, n);

    /* allocate synchronization tree(s) */
    arriver_t *arr = arriver_alloc();
    if (arr == NULL)
        return -1;
    reducer_t *red = NULL;
    jl_value_t **rargs = NULL;
    uint32_t nrargs = 0;
    jl_method_instance_t *mredfunc = NULL;
    jl_generic_fptr_t rfptr;
    if (_rargs != NULL) {
        red = reducer_alloc();
        if (red == NULL) {
            arriver_free(arr);
            return -2;
        }
        if (setup_task_fun(_rargs, &rargs, &nrargs, &mredfunc, &rfptr) != 0) {
            reducer_free(red);
            arriver_free(arr);
            return -3;
        }
    }

    /* allocate and enqueue (GRAIN_K * nthreads) tasks */
    *t = NULL;
    int64_t start = 0, end;
    for (int64_t i = 0;  i < n;  ++i) {
        end = start + each.quot + (i < each.rem ? 1 : 0);
        jl_task_t *task;
        if (*t == NULL)
            *t = task = new_task(_args);
        else
            task = copy_task(*t);
        if (task == NULL)
            return -4;

        task->start = start;
        task->end = end;
        task->parent = *t;
        task->grain_num = i;
        task->arr = arr;
        if (_rargs != NULL) {
            task->rargs = rargs;
            task->nrargs = nrargs;
            task->mredfunc = mredfunc;
            task->rfptr = rfptr;
            task->red = red;
        }

        if (multiq_insert(task, ptls->tid) != 0) {
            return -5;
        }

        start = end;
    }

    /* only yield if we're running a non-sticky task */
    if (!(ptls->current_task->settings & TASK_IS_STICKY))
        partr_yield(1);

    return 0;
}


/*  partr_yield() -- cause the invoking task to yield

    If `requeue` is set, the task is inserted into the relevant queue
    (sticky or multiqueue), otherwise it is assumed it will be re-queued
    in some other way (e.g. from another task's completion queue).
 */
int partr_yield(int requeue)
{
    jl_ptls_t ptls = jl_get_ptls_states();
    jl_task_t *ytask = ptls->current_task;

#ifdef ENABLE_TIMINGS
    jl_timing_block_t *blk = ytask->timing_stack;
    if (blk)
        jl_timing_block_stop(blk);
#endif

    if (!jl_setjmp(ytask->ctx, 0)) {
        ytask->current_tid = -1;
        ptls->current_task = NULL;

        // backtraces don't survive task switches, see issue #12485
        ptls->bt_size = 0;

        // save state into yielding task
        ytask->gcstack = ptls->pgcstack;
        ytask->world_age = ptls->world_age;

        // If the current task is not holding any locks, free the locks list
        // so that it can be GC'd without leaking memory.
        // TODO: this will be too slow!
        arraylist_t *locks = &ytask->locks;
        if (locks->len == 0  &&  locks->items != locks->_space) {
            arraylist_free(locks);
            arraylist_new(locks, 0);
        }

        // re-enqueue the task
        if (requeue)
            enqueue_task(ytask);

        // run the next available task
        run_next();

        // unreachable
        gc_debug_critical_error();
        abort();
    }

#ifdef ENABLE_TIMINGS
    assert(blk == jl_current_task->timing_stack);
    if (blk)
        jl_timing_block_start(blk);
#endif

    return 0;
}


/*  partr_wait() -- deschedules the task until the specified condition is
        triggered
 */
int partr_wait(jl_condition_t *c)
{
}


/*  partr_notify() -- triggers the specified condition, causing all tasks
        waiting on it to become schedulable
 */
int partr_notify(jl_condition_t *c)
{
}

#endif // JULIA_ENABLE_PARTR
#endif // JULIA_ENABLE_THREADING

#ifdef __cplusplus
}
#endif
