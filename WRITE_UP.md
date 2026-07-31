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

