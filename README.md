# MW-MAS
Code for MW-MAS: A Multi-Agent System for Multimodal Watermarking with Agent Orchestration

```
cd mw_mas
python multimodal_infer.py
```

## 🧪 Experimental Setup

MW-MAS was evaluated on **250 image–text pairs** from the [WIT dataset](https://github.com/google-research-datasets/wit).  
Each image was resized to **512×512**, and its caption was used as the **text prompt**.



### 📝 Text Watermark Agent
- **Backbone:** Guanaco-7B (PEFT, no fine-tuning)  
- **Prompt:** Guanaco dialogue template  
- **Generation:**
`temperature = 1.0`
`top_p = 0.95`
`max_length = 120`

- **Watermarking:** Three-Bricks
`n_gram = 1`
`γ = 0.25`
`δ = 2.0`


### 🖼️ Image Watermark Agent
- **Framework:** [WAM](https://github.com/facebookresearch/watermark-anything) ([frozen checkpoint](https://huggingface.co/facebook/watermark-anything/blob/main/checkpoint.pth))  
- **Payload:** 1-bit watermark masked over 50% of pixels  



### 🧩 Orchestration Agent
Samples are divided into three difficulty regions:

| Region | Condition |
|:--------|:------------|
| **Easy** | textLen < 30 or entropy < 5.0 |
| **Medium** | otherwise |
| **Hard** | textLen < 85 or edge density < 0.2 |

_These thresholds are empirically tuned from the feature distributions._

- **Reward factors:** detection score (AUC / bit accuracy), distortion (SBERT / SSIM), and computational cost (model size, inference time).  
- **Hyperparameters:**  `λ = 0.5` `μ = 0.1`

## 📚 Citation

If you find **MW-MAS** useful for your research, please cite our paper:

```
TBD
