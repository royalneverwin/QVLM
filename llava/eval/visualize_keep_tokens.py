import argparse
import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image

def fast_map_dpp(kernel, k):
    """
    Fast MAP inference for DPP.
    kernel: (B, N, N)
    k: int, number of items to select
    Returns: select_idx (B, k)
    """
    B, N, _ = kernel.shape
    device = kernel.device
    cis = torch.zeros((k, B, N), device=device) # (T, B, N)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone() # (B, N)
    select_idx = torch.empty((k, B), dtype=torch.long, device=device) # (T, B)
    
    for i in range(k):
        j = torch.argmax(di2s, dim=-1)
        select_idx[i] = j

        eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
            / (torch.sqrt(di2s[torch.arange(B), j] + 1e-8)).unsqueeze(-1)
        cis[i, :, :] = eis
        di2s -= torch.square(eis)
        di2s[torch.arange(B), j] = -float('inf')
    
    select_idx = torch.sort(select_idx.t()).values # (B, k)
    return select_idx

def create_mask_image(select_idx, num_patches_side=24, patch_size=14, target_size=336):
    """
    Creates a heatmap where selected tokens are highlighted.
    select_idx: (k,) tensor of selected indices
    """
    mask = torch.zeros((num_patches_side * num_patches_side,))
    mask[select_idx] = 1.0
    mask = mask.reshape(1, 1, num_patches_side, num_patches_side)
    mask = F.interpolate(mask, size=(target_size, target_size), mode='nearest')
    mask = mask.squeeze().numpy()
    return mask

def visualize(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name, 
        load_8bit=False, 
        load_4bit=False,
        visual_token_num=args.visual_token_num
    )

    # Load ScienceQA questions
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))

    # Filter out questions without images and take the first 10
    image_samples = [q for q in questions if 'image' in q][:10]

    for sample in image_samples:
        image_file = sample["image"]
        question_id = sample["id"]
        
        # In ScienceQA, the query is the first conversation's value

        qs = sample['conversations'][0]['value'].replace('<image>', '').strip()
        cur_prompt = qs
        
        # Output directory for this specific question
        output_dir = os.path.join(args.output_folder, str(question_id))
        os.makedirs(output_dir, exist_ok=True)

        image_path = os.path.join(args.image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping...")
            continue
            
        image = Image.open(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        images = image_tensor.unsqueeze(0).half().cuda()

        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        cur_prompt = '<image>' + '\n' + cur_prompt

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.no_grad():
            image_features, image_embeds, text_embeds = self.get_model().get_vision_tower()(
                images, 
                texts=sample['conversations'][0]['value']
            )

            # --- 1. Compute Base Relevance (CDPruner style) ---
            # Normalize
            image_embeds_norm = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds_norm = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # Relevance (negative cosine similarity as in CDPruner code)
            relevance = torch.matmul(image_embeds_norm, text_embeds_norm.t()) # (N, 1)
            relevance = -relevance.squeeze(-1) # (N,)
            
            # Normalize relevance to [0, 1]
            relevance_min, relevance_max = relevance.min(), relevance.max()
            relevance_norm = (relevance - relevance_min + 1e-6) / (relevance_max - relevance_min + 1e-6)
            
            # --- 2. Compute Quantization Sensitivity ---
            quant_sensitivity = image_features.norm(dim=-1) # (N,)
            quant_min, quant_max = quant_sensitivity.min(), quant_sensitivity.max()
            quant_norm = (quant_sensitivity - quant_min + 1e-6) / (quant_max - quant_min + 1e-6)
            
            # --- 3. Compute Similarity Matrix ---
            image_features_norm = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            similarity = torch.matmul(image_features_norm, image_features_norm.t()) # (N, N)
            
            # Expand dims for batch processing format (B=1)
            relevance_norm_b = relevance_norm.unsqueeze(0)
            quant_norm_b = quant_norm.unsqueeze(0)
            similarity_b = similarity.unsqueeze(0)
            
            # --- Selection 1: Standard CDPruner ---
            kernel_base = relevance_norm_b.unsqueeze(2) * similarity_b * relevance_norm_b.unsqueeze(1)
            select_base = fast_map_dpp(kernel_base, args.visual_token_num)[0]
            mask_base = create_mask_image(select_base)
            
            # --- Selection 2: Quant-Aware CDPruner ---
            alpha = args.alpha
            fused_relevance = alpha * relevance_norm_b + (1 - alpha) * quant_norm_b
            kernel_quant = fused_relevance.unsqueeze(2) * similarity_b * fused_relevance.unsqueeze(1)
            select_quant = fast_map_dpp(kernel_quant, args.visual_token_num)[0]
            mask_quant = create_mask_image(select_quant)
            
            # --- Selection 3: Top Quantization Outliers (Magnitude only) ---
            # Select top-K highest magnitude tokens directly
            _, select_outliers = torch.topk(quant_sensitivity, args.visual_token_num)
            mask_outliers = create_mask_image(select_outliers)
            
            # --- Plotting ---
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Create a wrapped title for long questions
            import textwrap
            wrapped_qs = textwrap.fill(qs, width=100)
            fig.suptitle(f'QID: {question_id} | Keep: {args.visual_token_num}\n{wrapped_qs}', fontsize=12)
            
            # Original
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Standard CDPruner
            overlay_base = np.uint8(original_image * 0.5 + plt.cm.jet(mask_base)[:, :, :3] * 255 * 0.5)
            axes[1].imshow(overlay_base)
            axes[1].set_title("Standard CDPruner")
            axes[1].axis('off')
            
            # Quant-Aware CDPruner
            overlay_quant = np.uint8(original_image * 0.5 + plt.cm.jet(mask_quant)[:, :, :3] * 255 * 0.5)
            axes[2].imshow(overlay_quant)
            axes[2].set_title(f"Quant-Aware CDPruner (alpha={alpha})")
            axes[2].axis('off')
            
            # Outliers Only
            overlay_outliers = np.uint8(original_image * 0.5 + plt.cm.jet(mask_outliers)[:, :, :3] * 255 * 0.5)
            axes[3].imshow(overlay_outliers)
            axes[3].set_title("Top Outliers (Magnitude)")
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{question_id}_comparison.png"))
            plt.close()
            
            print(f"Processed QID: {question_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--question-file", type=str, required=True, help="Path to ScienceQA question JSON")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to ScienceQA images folder")
    parser.add_argument("--output-folder", type=str, default="playground/data/visualize")
    parser.add_argument("--visual_token_num", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.7, help="Fusion weight for Quant-Aware method")
    args = parser.parse_args()

    visualize(args)