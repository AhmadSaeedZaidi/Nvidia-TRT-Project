### 27/04
- started analyzing repository
- lots of new concepts here 
- the actual code we need to check seems to be in cpp/tensortllm/ not tensorrtllm

### 29/04
- ok so from what it seems like, no one actually writes "regular cuda" or more specifically the stuff we've learned in class
- everything actually works on this other library called "CUTLASS" and its already on version 3.
- CUTLASS, which is a template library, along with its sub library CuTe, are basically designed to reduce the manual pointer arithmetic we have to do in regular cuda. 
- Instead in CuTe we use something called a **Layout**. A Layout is a tuple combining a Shape and a Stride. The Shape defines the logical dimensions of a matrix. The Stride defines the physical memory spacing between elements. This mechanism maps any multi-dimensional coordinate to a flat one-dimensional memory index at compile time.
- For example,transposing a matrix or changing tile sizes only requires passing a different Stride parameter to the same kernel logic.
- CuTe also uses tensors (probably will utilize tensor cores). A Tensor in CuTe pairs a raw memory pointer with a Layout. This decouples the physical location of the data from the indexing logic.
- Tensors explicitly track their memory space, such as Global Memory, Shared Memory, or Registers. You index the Tensor directly. The underlying Layout resolves the memory offset automatically based on the assigned memory space.
- CuTe abstracts specific GPU instructions into discrete components. An MMA_Atom represents Matrix-Multiply-Accumulate instructions tailored specifically for Tensor Cores. A Copy_Atom represents data movement instructions. Modern architectures like Hopper use the Tensor Memory Accelerator (TMA) to move data directly from Global Memory to Shared Memory, completely bypassing the register file. You configure a Copy_Atom to use TMA and map it to an MMA_Atom to execute the math.
- Operations involving TMA are asynchronous. CuTe provides cute::mbarrier primitives to handle execution synchronization. A warp issues a TMA copy instruction via a Copy_Atom. The warp can then execute independent instructions. It subsequently waits at the mbarrier until the memory transfer is complete before feeding the populated Shared Memory tensor into the MMA_Atom.
- All in all, these new tools essentially take care of a lot of the hassle we had to deal with while writing regular cuda.
- Example code:
```cpp
#include <cute/tensor.hpp>

__global__ void copy_kernel(float const* S, float* D) {
    using namespace cute;

    // Define layouts
    auto m = Int<128>{};
    auto n = Int<128>{};
    
    // Source: Global Memory
    Tensor s_g = make_tensor(make_gmem_ptr(S), make_layout(make_shape(m, n)));
    // Destination: Shared Memory
    __shared__ float smem[128*128];
    Tensor s_s = make_tensor(make_smem_ptr(smem), make_layout(make_shape(m, n)));

    // Define a tiling strategy (e.g., 16 threads in M, 8 in N)
    auto t_layout = make_layout(make_shape(Int<16>{}, Int<8>{}));
    // TiledCopy allows using optimized instructions (like LDGSTS)
    auto tiled_copy = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{}, 
                                      t_layout, 
                                      Layout<Shape<Int<8>, Int<16>>>{}); // Thread layout

    // Partition the tensors based on thread ID
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    Tensor tS = thr_copy.partition_S(s_g);
    Tensor tD = thr_copy.partition_D(s_s);

    // Perform the copy
    copy(tiled_copy, tS, tD);
}
```
### 30/04
- Going to look into triton today since tensorrt makes heavy use of it along with CUTLASS and CuTe.
- "Triton is an open-source programming language and compiler developed by OpenAI for writing highly efficient custom GPU kernels using Python syntax."
- Regular CUDA requires you to manage execution at the thread level by explicitly assigning individual threads to scalar memory addresses. Triton operates at the block level. You write operations that apply to entire tiles of data simultaneously. So basically, no more thread level coding.
- This block-level abstraction allows the Triton compiler to automate the most difficult parts of GPU programming. It automatically manages shared memory allocation. It automatically handles memory coalescing for global memory reads and writes. The compiler analyzes the block-level operations and generates the highly optimized PTX instructions required for tensor cores. wow
- Some other compiler related stuff we don't really know about but some optimizations in the compilation pipeline as well to make it even faster.
- No wonder this stuff is industry standard.
----

- Cloning the repo today finally, but GIKI net is so slow we probably won't even be able to do anything right now. 
- Its over 1 whole GB wow.

### 01/05
- ok so looking into the repository the problem is that there are missing kernels in TensorRT-LLM\cpp\tensorrt_llm\kernels\trtllmGenKernels\fmha\cubin/
- this just has thousands of weird fiels these aren't even cuda?
- its compiled stuff
- ok so i need to find what code compiles those kernels and why seq lengths 32 and above are not compiled
----
- ok its been a few hours now and i still haven't found where these are compiled
- lots of templates and such and setup.py inside cpp\kernels\fmha_v2\setup.py looks promising since this file is definitely being used to compile kernels

### 02/05
- found something that makes this project impossible
- cpp\tensorrt_llm\kernels\trtllmGenKernels\fmha\fmhaRunner.cpp, line 44: 
```cpp
    TLLM_CHECK_WITH_INFO(mSM == kSM_100 || mSM == kSM_103, "Unsupported architecture");
```
- there are actually two variants of blackwell architecture
- consumer grade gpus are sm120, and this is sm100 aka a datacenter gpu, so even if we write the code we can't even run it
----
- new update: secured temp access to a b200! now we can even reproduce the issue and profile it on full size
- project doable!

### 03/05
- double nail in the coffin today:
    --- 
    1. - those kernels are missing not because of some missing code or whatever
        - those binaries actually come precompiled from nvidia
        - they probaby use their own toolchain to generate these files and they've probably omitted these by mistake
        -  even if we compile these and force them into cubin/ it will be quite a mess
    ---
    2. - the original dev who was assigned this issue actually was working on it and the kernels have most likely already been merged last week on upstream i.e flash-infer
        - the changes just haven't been propagated back down
        - our whole "project" is redundant and basically over
    
- :(
