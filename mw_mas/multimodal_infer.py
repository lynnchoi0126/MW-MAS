import os
import sys
import json
from glob import glob
from PIL import Image
from datetime import datetime
import re

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'WAM')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'three_bricks')))

from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    load_model_from_checkpoint, default_transform, unnormalize_img,
    create_random_mask, plot_outputs, msg2str
)

from main_watermark import get_args_parser, WmGenerator, OpenaiGenerator, OpenaiDetector, OpenaiDetectorZ
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
import numpy as np
def natural_key(filename):
    # Extract numbers as integers from filename for sorting
    nums = re.findall(r'\d+', filename)
    return [int(num) for num in nums]
# ---------- Configs ----------
tb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'three_bricks'))
wam_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'WAM'))

IMG_DIR = tb_path + "/WIT_dataset/WIT_data_organized_resized_combined"  # Directory containing the original images
PROMPT_PATH = tb_path + "/data/WIT_data_0623_combined.json"
OUTPUT_DIR = "paired_output"
CHECKPOINT_DIR = wam_path + "/checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load Models ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image model
wam = load_model_from_checkpoint(
    os.path.join(CHECKPOINT_DIR, "params.json"),
    os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
).to(device).eval()

# Text model
auth_token = None
# ### guanaco-7b:
# model_name = "huggyllama/llama-7b"
# adapters_name = 'timdettmers/guanaco-7b'

# ### GPT:  
# model_name = "gpt2" #"openai-community/gpt2"
# adapters_name = None

# ### Mistral-7B:
# auth_token = 'XXX'  # Replace with your Hugging Face token if needed
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# adapters_name = None

# ### llama-7b":
# model_name = "huggyllama/llama-7b"
# adapters_name = None

### "guanaco-7b":
model_name = "huggyllama/llama-7b"
adapters_name = 'timdettmers/guanaco-7b'
    
#### "guanaco-13b":
    #     model_name = "huggyllama/llama-13b"

# ### gemma 2
# auth_token = 'XXX'  # Replace with your Hugging Face token if needed
# model_name = "google/gemma-2-9b-it"
# adapters_name = None
# print(f"SELECTED LANGUAGE MODEL: Gemma 2 9B Instruct")

print(f"Using model: {model_name}, adapters: {adapters_name}")    #     adapters_name = 'timdettmers/guanaco-13b'
if auth_token is not None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        use_auth_token=auth_token,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
    )
from peft import PeftModel
if adapters_name is not None:
    model = PeftModel.from_pretrained(model, adapters_name)
model.eval()

# Generator
generator = OpenaiGenerator(
    model, tokenizer,
    1, 42,
    "hash", 35317, 0
)

# generator = OpenaiGenerator(
#     model, tokenizer,
#     ngram=1, seed=42,
#     seeding="hash", 35317, payload=0
# )
# Detector
# detector = OpenaiDetector(tokenizer, ngram=1, seed=42, seeding="hash", hash_key=35317)
detector = OpenaiDetector(tokenizer, 1, 42, "hash", 35317)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

# ---------- Load Data ----------
with open(PROMPT_PATH, "r") as f:
    text_data = json.load(f)

image_paths = glob(os.path.join(IMG_DIR, "*.jpg"))
image_paths = sorted(image_paths, key=natural_key)

assert len(image_paths) == len(text_data), "Mismatch between image and text data lengths"

# ---------- Data Slicing ----------
ind_st = 0
ind_end = 250
text_data = text_data[ind_st:ind_end]
image_paths = image_paths[ind_st:ind_end]

start = datetime.now()
# ---------- Process Each Pair ----------
log_stats = []
all_pvalues = []  # Store all p-values for AUC calculation
all_labels = []    # Store corresponding labels (1 for watermarked, 0 for non-watermarked

for idx, (img_path, text_obj) in enumerate(zip(image_paths, text_data)):
    print(f"Processing pair {idx+1}/{len(image_paths)}")
    idx += ind_st  # Adjust index for logging

    ## Image inference
    img = Image.open(img_path).convert("RGB")
    img_tensor = default_transform(img).unsqueeze(0).to(device)
    wm_msg = wam.get_random_msg(1)
    outputs = wam.embed(img_tensor, wm_msg)
    mask = create_random_mask(img_tensor, num_masks=1, mask_percentage=0.5)
    img_w = outputs['imgs_w'] * mask + img_tensor * (1 - mask)

    preds = wam.detect(img_w)["preds"]
    mask_preds = torch.sigmoid(preds[:, 0, :, :])
    bit_preds = preds[:, 1:, :, :]
    pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()
    bit_acc = (pred_message == wm_msg).float().mean().item()

    ## Text generation - watermarked
    ins = text_obj["instruction"]
    input = text_obj["input"]
    prompt = f"A chat between a human and an artificial intelligence assistant.\nThe assistant gives helpful answers to the user's questions.\n\n### Human: {ins}\n\n### Input:\n{input}\n\n### Assistant:"
    result = generator.generate([prompt],max_gen_len=60, temperature=1.0,top_p=0.95)[0]
    # result = result.replace(ins, "").strip()  # Remove instruction part
    result = result[len(prompt):].strip()

    # For watermarked text
    scores_no_aggreg_wm = detector.get_scores_by_t([result], scoring_method="v2", payload_max=0)
    scores_wm = detector.aggregate_scores(scores_no_aggreg_wm)
    pvalues_wm = detector.get_pvalues(scores_no_aggreg_wm)
    pvalue_wm = float(pvalues_wm[0][0])
    
    # For original/non-watermarked text
    original_text = text_obj["output"]
    scores_no_aggreg_orig = detector.get_scores_by_t([original_text], scoring_method="v2", payload_max=0)
    scores_orig = detector.aggregate_scores(scores_no_aggreg_orig)
    pvalues_orig = detector.get_pvalues(scores_no_aggreg_orig)
    pvalue_orig = float(pvalues_orig[0][0])
    
    # Collect for AUC
    all_pvalues.extend([pvalue_wm, pvalue_orig])
    all_labels.extend([1, 0])  # 1=watermarked, 0=non-watermarked
    
    # SBERT similarity
    xs = sbert_model.encode([result, text_obj["output"]], convert_to_tensor=True)
    score_sbert = cossim(xs[0], xs[1]).item()
    log_stat = {
        'text_index': idx,
        'num_token': len(scores_no_aggreg_wm[0]),
        'score': float(scores_wm[0][0]),
        'pvalue': pvalue_wm,
        'score_sbert': score_sbert,
        'bit_accuracy': bit_acc,
    }
    log_stats.append(log_stat)

    ## Save outputs
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    sample_dir = os.path.join(OUTPUT_DIR, f"sample_{idx:04d}")
    os.makedirs(sample_dir, exist_ok=True)

    save_image(unnormalize_img(img_w), os.path.join(sample_dir, "watermarked.png"))
    save_image(mask_preds.unsqueeze(1), os.path.join(sample_dir, "pred_mask.png"))
    save_image(mask, os.path.join(sample_dir, "target_mask.png"))

    with open(os.path.join(sample_dir, "text_output.txt"), "w") as f:
        f.write("Prompt:\n" + prompt + "\n\n")
        f.write("Generated (Watermarked):\n" + result + "\n")
        f.write("Original (Non-watermarked):\n" + original_text + "\n\n")
        f.write(f"Bit accuracy: {bit_acc:.4f}\n")
        f.write(f"Predicted message: {msg2str(pred_message[0])}\n")
        f.write(f"P-value (watermarked): {pvalue_wm:.6f}\n")
        f.write(f"P-value (non-watermarked): {pvalue_orig:.6f}\n")
        f.write(f"S-BERT Similarity: {score_sbert:.4f}\n")

    plot_outputs(img_tensor.detach(), img_w.detach(), mask.detach(), mask_preds.unsqueeze(1).detach())
# ---------- AUC Calculation ----------
# Use 1-pvalue as score since lower p-values indicate higher watermark likelihood
scores_auc = [1 - p for p in all_pvalues]
auc_score = roc_auc_score(all_labels, scores_auc)
print(f"AUC Score: {auc_score:.4f}")
print(f"Mean S-BERT Score: {np.mean([stat['score_sbert'] for stat in log_stats]):.4f}")
print(f"Mean Bit Accuracy: {np.mean([stat['bit_accuracy'] for stat in log_stats]):.4f}")

# pvalues = [stat['pvalue'] for stat in log_stats]
# pvalue_threshold = 1e-3
# predicted_labels = [1 if p <= pvalue_threshold else 0 for p in pvalues]
# true_labels = [1] * len(pvalues)  # all are watermarked in this run
# auc_score = roc_auc_score(true_labels, predicted_labels)
# print(f"AUC Score (based on p-value <= {pvalue_threshold}): {auc_score:.4f}")

print("✅ All pairs processed!")
end = datetime.now()

diff = end - start
print(f"총 소요 시간: {diff.total_seconds()/(ind_end-ind_st)}초")