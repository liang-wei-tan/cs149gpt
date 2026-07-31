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


Running Part 1 Test: Naive Unfused Attention

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