# Creating CoreML Apple Neural Engine Models
- Currently all of these models would have been converted from openelm-coreml.py
- Review ops, layers, precision for a model
- Review [apple/ml-recurrent-drafter](https://github.com/apple/ml-recurrent-drafter)
  - [modeling_llama.py](https://github.com/apple/ml-recurrent-drafter/blob/main/recurrent_drafting/modeling_llama.py)
  - this model also seems to be an ANE optimized Llama with the ANE Principles being implemented
    - lines 161 and 162 deal with key_states and value_states
    - Class LlamaAttention
      - key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.n_kv_groups)
      - value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.n_kv_groups)
- Review [chunk_mlprogram.py](https://github.com/antmikinka/swift-transformers-test/blob/main/chunk_mlprogram.py) (changed from [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion/blob/main/python_coreml_stable_diffusion/chunk_mlprogram.py))
	- Optimize for chunking text LLMs
	- needs to check PSNR 
		- random_gen_input_feature_type func is not working due to the model being converted, not properly displaying a value type to let the func know how to generate those input features (this seems to be the issue)
	- program does work




## Ways to View Layers, OPs, & Precision
- The differences: how they get info, how they display it, and environment packages
- [smpanaro/CoreMLInspect](https://github.com/smpanaro/CoreMLInspect)
	- this would work basically all around in any env
- [layer-iteration.py](https://github.com/antmikinka/swift-transformers-test/blob/main/layer-iteration.py)
	- this requires something similar to [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples) env 
	- due to missing PIL package, I had issues using my python venv


  
## CoreMLInspect Models - Layers, OPs, & Precision
- [CoreMLInspect-OpenELM-1B-Instruct](https://github.com/antmikinka/swift-transformers-test/blob/main/CoreMLInspect-OpenELM-1B-Instruct-Compiled-Model-CPU-NE.txt)
- [CoreMLInspect-OpenELM-270M-Instruct](https://github.com/antmikinka/swift-transformers-test/blob/main/CoreMLInspect-OpenELM-270M-Instruct-Compiled-Model-CPU-NE.txt)



  
## layer-iteration.py Model - Layers, OPs, & Precision
- [OpenELM-270M-Instruct](https://github.com/antmikinka/swift-transformers-test/blob/main/OpenELM-270M-Instruct-128-FP16ComputePrecisoinv2.txt)
- OpenELM-1B-Instruct (may not come, have to determine if RAM or Storage Issue)




## Performance Tests
- [Chunked OpenELM-270M-Instruct](https://github.com/antmikinka/swift-transformers-test/blob/main/Model%20Performance%20Tests/OpenELM-270M-Instruct-128-FP16ComputePrecisionv2_chunked_pipeline%20PERFORMANCE%20TEST.png)
- [Not Chunked OpenELM-270M-Instruct](https://github.com/antmikinka/swift-transformers-test/blob/main/Model%20Performance%20Tests/OpenELM-270M-Instruct-128-FP16ComputePrecisionv2%20PERFORMANCE%20TEST.png)




## OpenELM-270M-Instruct
#### Chunked
- [anthonymikinka/OpenELM-270M-Instruct-128-FP16ComputePrecisionv2_chunked_pipeline](https://huggingface.co/anthonymikinka/OpenELM-270M-Instruct-128-FP16ComputePrecisionv2_chunked_pipeline/tree/main)


## OpenELM-1B-Instruct
- [anthonymikinka/OpenELM-1_1B-Instruct-128-FP16ComputePrecision_v2](https://huggingface.co/anthonymikinka/OpenELM-1_1B-Instruct-128-FP16ComputePrecision_v2)
  
#### Palttized
- [anthonymikinka/OpenELM-1_1B-Instruct-128-FP16ComputePrecision_v2-Palettized-Kmeans-4bits](https://huggingface.co/anthonymikinka/OpenELM-1_1B-Instruct-128-FP16ComputePrecision_v2-Palettized-Kmeans-4bits)
- [anthonymikinka/OpenELM-1_1B-Instruct-128-FP16ComputePrecision_v2-Palettized-Kmeans-6Bits](https://huggingface.co/anthonymikinka/OpenELM-1_1B-Instruct-128-FP16ComputePrecision_v2-Palettized-Kmeans-6Bits)
