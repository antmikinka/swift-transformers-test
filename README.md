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
- Review chunk_mlprogram.py (changed from [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion/blob/main/python_coreml_stable_diffusion/chunk_mlprogram.py))
	- Optimize for chunking text LLMs
	- needs to check PSNR 
		- random_gen_input_feature_type func is not working due to the model being converted, not properly displaying a value type to let the func know how to generate those input features (this seems to be the issue)
	- program does work




## Ways to View Layers, OPs, & Precision
- The difference is the way they display information and how they get it
- [CoreMLInspect](https://github.com/smpanaro/CoreMLInspect)
- layer-iteration.py



  
## CoreMLInspect Models - Layers, OPs, & Precision
- [CoreMLInspect-OpenELM-1B-Instruct](https://github.com/antmikinka/swift-transformers-test/blob/main/CoreMLInspect-OpenELM-1B-Instruct-Compiled-Model-CPU-NE.txt)
- [CoreMLInspect-OpenELM-270M-Instruct](https://github.com/antmikinka/swift-transformers-test/blob/main/CoreMLInspect-OpenELM-270M-Instruct-Compiled-Model-CPU-NE.txt)



  
## layer-iteration.py Model - Layers, OPs, & Precision
- [OpenELM-270M-Instruct](https://github.com/antmikinka/swift-transformers-test/blob/main/OpenELM-270M-Instruct-128-FP16ComputePrecisoinv2.txt)
- OpenELM-1B-Instruct (may not come, have to determine if RAM or Storage Issue)


## Models
- 



![newgithub-anthonymikinka-swift-transformers-test](https://github.com/antmikinka/swift-transformers-test/assets/67480807/05282faf-88f7-450b-a10b-0d87a261e894)
