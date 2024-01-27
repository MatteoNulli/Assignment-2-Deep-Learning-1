"""Defines the VisualPrompting model (based on CLIP)"""
from pprint import pprint
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import warnings


def load_clip_to_cpu(cfg):
    """Loads CLIP model to CPU."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class DeepPromptCLIP(nn.Module):
    """Modified CLIP module to support prompting."""
    def __init__(self, args, dataset, template="This is a photo of {}"):
        super(DeepPromptCLIP, self).__init__()
        classnames = dataset.classes

        print(f"Loading CLIP (backbone: {args.arch})")
        clip_model = self.load_clip_to_cpu(args)
        clip_model.to(args.device)

        # hack to make model as float() (This is a CLIP hack)
        if args.device == "cpu":
            clip_model = clip_model.float()


        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        print("List of prompts:")
        pprint(prompts)

        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(args.device)

        # TODO: Write code to compute text features.
        # Hint: You can use the code from clipzs.py here!

        # Instructions:
        # - Given a list of prompts, compute the text features for each prompt.
        # - Return a tensor of shape (num_prompts, 512).
        input_text = prompts
        with torch.no_grad():
            # Calculate features
            # - Compute the text features (encodings) for each prompt.
            text_features = clip_model.encode_text(input_text)
        
        # - Normalize the text features.
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # remove this line once you implement the function
        # raise NotImplementedError("Write the code to compute text features.")

        self.text_features = text_features
        self.clip_model = clip_model
        self.logit_scale = self.clip_model.logit_scale.exp().detach()

        self.injection_layer = args.injection_layer

        # TODO: Initialize the learnable deep prompt.
        # Hint: consider the shape required for the deep prompt to be compatible with the CLIP model 
        # Hint: CLIP uses different datatypes for CPU (float32) and GPU (float16)
        # Hint: use args.prompt_num to specify the number of deep prompts to use

        # self.deep_prompt = nn.Parameter(torch.randn((1, 3, args.prompt_size, args.prompt_size)))
        self.deep_prompt = nn.Parameter(torch.randn(args.prompt_num, 1, 768), requires_grad=True)
        
        # remove this line once you implement the function
        # raise NotImplementedError("Write the code to compute text features.")


    def forward(self, image):
        """Forward pass of the model."""

        # TODO: Implement the forward function. This is not exactly the same as
        # the model_inferece function in clipzs.py! Please see the steps below.

        # Steps:
        # - Compute the image features using the CLIP model (be sure use the custom_encode_image function).
        image_features = self.custom_encode_image(image)
        # - Normalize the image features.
        image_features = torch.nn.functional.normalize(image_features)
        # - Compute similarity logits between the image features and the text features.
        # - You need to multiply the similarity logits with the logit scale (clip_model.logit_scale).
        similarity_logits = (image_features @ self.text_features.T) * self.clip_model.logit_scale.exp()
        
        # - Return logits of shape (batch size, number of classes).
        return similarity_logits

    def custom_encode_image(self, x):
        """Encode image using CLIP model and add deep prompts."""
        # cf. https://github.com/openai/CLIP/blob/main/clip/model.py#L223

        x = x.type(self.clip_model.dtype)
        image_encoder = self.clip_model.visual
        
        x = image_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + image_encoder.positional_embedding.to(x.dtype)
        x = image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        # TODO: Implement the part of the code where the deep prompt is injected into the CLIP model.
        # The custom_encode_image function largely follows the code from the CLIP repository.
        # You only need to modify the code responsible for running the transformer blocks.

        # Steps:
       
       
        # - Iterate over the transformer blocks (image_encoder.transformer.resblocks).
        
        # deep_prompt_expanded = self.deep_prompt.expand(-1, x.shape[1], -1)
        # print(self.deep_prompt.shape)
        # x = torch.cat([deep_prompt_expanded, x], dim=1)  
        # print(x.shape)
        
        deep_prompt = self.deep_prompt.type(self.clip_model.dtype)
        
        deep_prompt = deep_prompt.expand(-1, x.shape[1], -1)

        for i, block in enumerate(image_encoder.transformer.resblocks):
            if i == self.injection_layer:
                x = torch.cat([x, deep_prompt], axis=0)
                
            x = block(x)

        # Hint: Beware of the batch size (the deep prompt is the same for all images in the batch).

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = image_encoder.ln_post(x[:, 0, :])

        if image_encoder.proj is not None:
            x = x @ image_encoder.proj

        return x

    def load_clip_to_cpu(self, args):
        """Loads CLIP model to CPU."""
        backbone_name = args.arch
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url, args.root)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip.build_model(state_dict or model.state_dict())
        return model

    @torch.no_grad()
    def visualize_prompt(self, method):
        """Visualizes the prompt."""
        warnings.warn("Deep prompts are not supported for visualization.")
