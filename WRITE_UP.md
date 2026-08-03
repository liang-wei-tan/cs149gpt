# Warm-Up: Accessing Tensors (3 Points)

Tensor Shape: torch.Size([1, 2, 4, 4])

4D Tensor Contents:
 tensor([[[[0.0000e+00, 1.0000e-04, 2.0000e-04, 3.0000e-04],
          [2.0000e-04, 3.0000e-04, 4.0000e-04, 5.0000e-04],
          [4.0000e-04, 5.0000e-04, 6.0000e-04, 7.0000e-04],
          [6.0000e-04, 7.0000e-04, 8.0000e-04, 9.0000e-04]],

         [[0.0000e+00, 1.0000e-04, 2.0000e-04, 3.0000e-04],
          [2.0000e-04, 3.0000e-04, 4.0000e-04, 5.0000e-04],
          [4.0000e-04, 5.0000e-04, 6.0000e-04, 7.0000e-04],
          [6.0000e-04, 7.0000e-04, 8.0000e-04, 9.0000e-04]]]])

Indexing Value When: x = 0, y = 0, z = 3, b = 1
Expected: 0.0007
Result: 0.0007

> Briefly describe how a 4D tensor/array is laid out in memory. Why do you think this convention was chosen and how does it leverage hardware?

4D tensor are contiguous laid out im memory, preventing us from having to do random access let's say if we were to initialize x new nested vector for each additonal dimension. Additionally, elements of the inner most loop is packed next to one another for spatial locality. Hence if we were using a loop structure, with the lowest dimension in outermost loop and highest dimension in innermost loop, when we load a cache line, it will contain elements in the next iteration, leading to cache hits. Another example would be parallelizing the outermost loop across processor, each processor will read and process and continuos row of eleemtns, leading to better cache performance. 

# Part 1: A Simple (But Not So Efficient) Implementation of Attention 
Just nested loops
python3 gpt149.py part1

Compiling code into a PyTorch module...


# Running Part 1 Test: Naive Unfused Attention

-----RUNNING REFERENCE IMPLEMENTATION-----

WARNING:2026-07-31 13:43:58 7535:7535 init.cpp:155] function cbapi->getCuptiStatus() failed with error CUPTI_ERROR_INVALID_DEVICE (2)
WARNING:2026-07-31 13:43:58 7535:7535 init.cpp:156] CUPTI initialization failed - CUDA profiler activities will be missing
INFO:2026-07-31 13:43:58 7535:7535 init.cpp:158] If you see CUPTI_ERROR_INSUFFICIENT_PRIVILEGES, refer to https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti
STAGE:2026-07-31 13:43:59 7535:7535 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2026-07-31 13:43:59 7535:7535 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2026-07-31 13:43:59 7535:7535 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
manual attention == pytorch attention True
Manual Execution Time:  0.20553326606750488 

-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                    aten::empty         0.02%      42.000us         0.02%      42.000us      14.000us       5.00 Mb       5.00 Mb             3  
    REFERENCE - NAIVE ATTENTION        99.00%     203.501ms        99.98%     205.515ms     205.515ms       4.50 Mb      -1.00 Mb             1  
                    aten::zeros         0.02%      40.000us         0.61%       1.251ms     625.500us       4.50 Mb           0 b             2  
                    aten::clone         0.02%      31.000us         0.34%     701.000us     350.500us       1.00 Mb           0 b             2  
                model_inference         0.02%      42.000us       100.00%     205.557ms     205.557ms     512.00 Kb      -4.00 Mb             1  
                  aten::flatten         0.02%      46.000us         0.22%     457.000us      91.400us     512.00 Kb           0 b             5  
               aten::empty_like         0.00%       8.000us         0.01%      17.000us      17.000us     512.00 Kb           0 b             1  
            aten::empty_strided         0.01%      16.000us         0.01%      16.000us      16.000us     512.00 Kb     512.00 Kb             1  
                    aten::zero_         0.02%      49.000us         0.57%       1.178ms     589.000us           0 b           0 b             2  
                    aten::fill_         0.55%       1.129ms         0.55%       1.129ms     564.500us           0 b           0 b             2  
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 205.557ms

REFERENCE - NAIVE ATTENTION statistics
cpu time:  205.515ms
mem usage:  4718592 bytes
-----RUNNING STUDENT IMPLEMENTATION-----

STAGE:2026-07-31 13:44:05 7535:7535 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2026-07-31 13:44:06 7535:7535 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2026-07-31 13:44:06 7535:7535 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
manual attention == pytorch attention True
Manual Execution Time:  0.1925826072692871 

-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                  aten::empty         0.00%       9.000us         0.00%       9.000us       3.000us       5.00 Mb       5.00 Mb             3  
    STUDENT - NAIVE ATTENTION        99.47%     191.579ms        99.98%     192.563ms     192.563ms       4.50 Mb      -1.00 Mb             1  
                  aten::zeros         0.01%      16.000us         0.15%     283.000us     141.500us       4.50 Mb           0 b             2  
                  aten::clone         0.01%      17.000us         0.35%     676.000us     338.000us       1.00 Mb           0 b             2  
              model_inference         0.02%      42.000us       100.00%     192.605ms     192.605ms     512.00 Kb      -4.00 Mb             1  
                aten::flatten         0.01%      19.000us         0.15%     287.000us      57.400us     512.00 Kb           0 b             5  
             aten::empty_like         0.00%       4.000us         0.00%       5.000us       5.000us     512.00 Kb           0 b             1  
          aten::empty_strided         0.00%       6.000us         0.00%       6.000us       6.000us     512.00 Kb     512.00 Kb             1  
                  aten::zero_         0.00%       8.000us         0.13%     259.000us     129.500us           0 b           0 b             2  
                  aten::fill_         0.13%     251.000us         0.13%     251.000us     125.500us           0 b           0 b             2  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 192.605ms

STUDENT - NAIVE ATTENTION statistics
cpu time:  192.563ms
mem usage:  4718592 bytes

# Running Part 2: Blocked Matrix Multiply and Unfused Softmax (20 Points)

## Reference
REFERENCE - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  179.29ms
mem usage:  4718592 bytes

## Naive version 
Just to benchmark since i'm using a different machine from Part 1
STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  194.881ms
mem usage:  4718592 bytes

## Tiling of QK_t computation
### BLOCKSIZE = 16

Q = 16 * d = 16 * 64 = 1KB, K =  16 * d = 16 * 64 = 1KB
= 2-3KB used 

STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  193.046ms
mem usage:  4718592 bytes

### BLOCKSIZE = 32

 Q = 32 * d = 32 * 64 = 2KB, K =  16 * d = 16 * 64 = 2KB

STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  193.488ms
mem usage:  4718592 bytes

### BLOCKSIZE = 64

Q = 64 * d = 64 * 64 = 4KB, K =  64 * d = 64 * 64 = 4KB

STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  192.459ms
mem usage:  4718592 bytes



### BLOCKSIZE = 128

Q = 128 * d = 128 * 64 = 8KB, K =  128 * d = 128 * 64 = 8KB

STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  192.254ms
mem usage:  4718592 bytes


## Tiling to both computation of QK transpose and O 
REFERENCE - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  179.204ms
mem usage:  4718592 bytes
-----RUNNING STUDENT IMPLEMENTATION-------

### N = 8
same as reference pretty much

STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  175.535ms
mem usage:  4718592 bytes


### N = 16
Beats reference by quite a bit.

STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  137.249ms
mem usage:  4718592 bytes

### N = 32

STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  154.235ms
mem usage:  4718592 bytes

### N = 64

STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  159.746ms
mem usage:  4718592 bytes

### N = 128
STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  176.119ms
mem usage:  4718592 bytes

### N = 256
STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  194.285ms
mem usage:  4718592 bytes

## Questions
1. Share us some data about what tile sizes you tried when N=1024, and what the performance times were for each. What was the optimal tile size for your matrix multiplications? Explain why you think this tile size worked best for your implementation. There really isn't a wrong answer here, we just want to see that you experimented and tried to form conclusions.

> Optimal tile size is 16. there are 2 cases where tiling are used. Probably, 16 is the amount where both fit completely into memory for both. Increasing to 32 made one of the tiling loops experience cache trashing etc and hence, reducing the speedup. Increasing to256 on the other hand cause trashing in both cases and hence, becomes just like the naive case. Reducing to 8 on the other hand made the factor of reduction in memory read 8 and did not help much. Hence 16 is ideal. 

2. For a matrix multiply of Q(Nxd) and Ktranspose (dxN), what is the ratio of DRAM accesses in Part 2 versus DRAM acceses in Part 1? (assume 4 byte float primitives, 64 byte cache lines, as well as N and d are very large).
Ratio of DRAM for part 1. For first iteration in Loop for Q, we have to do a DRAM access for every row. And because cache isn't big enough to hold dXN, cache lines get evicted when iterating to higher N rows for K transpose. Hence, for every row, DRAM access is needed for all elements of K. 

Hence. N x d/cacheline for Q, and for N times, dxN is laoded fully, hence N x (Nxd/cacheline). Total = Nxd/cacheline + d x N x N / cacheline.

For part 2. we process in batch of 16 where all 16 lines of Q and K transpose fit in cache. For each batch of 16 rows of Q, all dxN of K has to be loaded too. Hence, the number of dram access is = loading of entire Q + N/16 batches x loading of entire K transpose. hence total = Nxd/cacheline + N / 32 * d x N / cacheline.

Hence ratio is ((1/cacheline)  x ( Nxd + NxNxd)) / ((1/cacheline) * (Nxd + NxNxd/16) ) = (Nxd + NxNxd) / (Nxd + NxNxd/16) ~~ 1/16x lesser DRAM access. 

# Part 3: Fused Attention (25 Points)
## Reference
Self CPU time total: 55.094ms

REFERENCE - FUSED ATTENTION statistics
cpu time:  55.057ms
mem usage:  557056 bytes

## Initial implementation naive fusing. 
It's odd that fusing does worse than fully naive. 
Naive attention does 192ms which is better. 

STUDENT - FUSED ATTENTION statistics
cpu time:  228.728ms
mem usage:  557056 bytes

For N=3000, Naive does 1703.56ms while naive fusing does 1893.063ms;

Based on research the reason is that while cache performance improve when fusing, it sacrificed parallel execution performance due to compiler not parallelizing it well. It is much easier in the naive example for compiler to identify parallelizable loops for SIMD execution.

## Fused implementation with openMP multi core procesisng on outermost loop 
STUDENT - FUSED ATTENTION statistics
cpu time:  239.625ms
mem usage:  557056 bytes

## Fused implementation with openMP multi core procesisng on 2 outermost loop 
Parallelizing on b and h
Self CPU time total: 87.780ms

STUDENT - FUSED ATTENTION statistics
cpu time:  87.743ms
mem usage:  557056 bytes

## Fused implementation with openMP multi core procesisng on 3 outermost loop 
Parallelizing on b and h, and row level
Self CPU time total: 58.832ms

STUDENT - FUSED ATTENTION statistics
cpu time:  58.791ms
mem usage:  557056 bytes

## FUsed implementation with multi core procesisng on 3 outermost loop , restructure loop such that innermost loop iterates on d which is contiguous
Significantly beats reference of 55ms
Self CPU time total: 34.331ms

STUDENT - FUSED ATTENTION statistics
cpu time:  34.291ms
mem usage:  557056 bytes

## Write up questions
1. Why do we use a drastically smaller amount of memory in Part 3 when compared to Parts 1 & 2?
> We used around ~557056 or 500 KB in part 3 vs 4718592 or 5MB in part 1 and 2, a 10x reduction. This is because we don't need a big NxN temporary array which for 1024 would have taken up 1024 * 1024 * 4 bytes = 4MB. Each thread though would have initialized 1024 x 4 = 4K bytes, which can safely fit into the L1 cache of each thread. 

2. Comment out your #pragma omp ... statement, what happens to your cpu time? Record the cpu time in your writeup. Why does fused attention make it easier for us utilize multithreading to a much fuller extent when compared to Part 1?
> without the pragma command, it takes 4x longer 123ms vs 34ms. It increases drastically because we run on a single core. With openMP, we can distribute workload across multi core, utilize more compute per unit time and hence, the whole programe finishes much faster. 

> With fused attention, computation for each row is consolidated into a nice independent block making it easier to parallelize. On the other hand, when it is not fused, we have several blocks within the b and h loop iterating though different dimension which makes it harder to parallelize. To parallelize the innermost loop which has different dimension, we would require locks to coordinate and generate correct output which leads to contention and other problems which slows thing down. Without locks, We would only be able to parallize across the b and h loops instead of 3 loops. This leads to lower CPU utilization and slower completion. 

# Part 4 : Putting it all Together - Flash Attention (35 Points)
Self CPU time total: 185.116ms

STUDENT - FLASH ATTENTION statistics
cpu time:  184.912ms
mem usage:  524288 bytes

## Questions
1. How does the memory usage of Part 4 compare to that of the previous parts? Why is this the case?
> 524288 bytes used or ~ 500KB. About same as what's used in Part 3 but less than what's used in part 1 & 2 which is about 4718592 bytes or 4MB. Flash attention like part 3 which fuses the operation does not require a big intermediate NxN array which is 1024 x 1024 x 4 bytes = 4MB. It requires only a few intermediate arrays based on the tiling size. Namely Kj (Bc x d), Vj (Bc x d), Qi (Br x d), Oi(Br x d), Li(Br) , given br and bc = 256, => 16384 x 4 + 256= 65792 = 65KB only. 

2. Notice that the performance of Part 4 is slower than that of the previous parts. Have we fully optimized Part 4? What other performance improvements can be done? Please list them and describe why they would increase performance.
- No multiprocessing involved
- we should be able to parallelize based on blocks.
- no explicit use of SIMD unit


# SIMD parallelization & multi core processingw with open mp
## Parallelizing across QKt only, speed up from 180ms to 60ms.

Self CPU time total: 65.192ms

STUDENT - FLASH ATTENTION statistics
cpu time:  64.887ms
mem usage:  524288 bytes

## Parallelizing across QKT, and exponential compute. 
reduce further from 64ms to 59ms
-----  ------------  
Self CPU time total: 60.104ms

STUDENT - FLASH ATTENTION statistics
cpu time:  59.886ms
mem usage:  524288 bytes