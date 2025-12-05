import torch
import torch.nn as nn
from transformers import PreTrainedModel, T5ForConditionalGeneration, T5Config, AutoTokenizer
from diffusers import AutoencoderKL
from einops.layers.torch import Rearrange
from einops import repeat
from torchvision.transforms import functional as F
from typing import Optional, Tuple, List, Any
from PIL import Image
from transformers import PretrainedConfig
from torchvision.utils import make_grid

class EmuruConfig(PretrainedConfig):
    model_type = "emuru"

    def __init__(self, 
                 t5_name_or_path='google-t5/t5-large', 
                 vae_name_or_path='blowing-up-groundhogs/emuru_vae',
                 tokenizer_name_or_path='google/byt5-small',
                 slices_per_query=1,
                 vae_channels=1,
                 style_enc="mean",
                 use_start_latent="True",
                 **kwargs):
        super().__init__(**kwargs)
        self.t5_name_or_path = t5_name_or_path
        self.vae_name_or_path = vae_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.slices_per_query = slices_per_query
        self.vae_channels = vae_channels
        self.style_enc = style_enc
        self.use_start_latent = use_start_latent

class Emuru(PreTrainedModel):
    """
    Emuru is a conditional generative model that integrates a T5-based decoder with a VAE
    for image generation conditioned on text and style images.
    Attributes:
        config_class (Type): Configuration class for the model.
        tokenizer (AutoTokenizer): Tokenizer loaded from the provided tokenizer configuration.
        T5 (T5ForConditionalGeneration): T5 model adapted for conditional generation.
        sos (nn.Embedding): Start-of-sequence embedding.
        vae_to_t5 (nn.Linear): Linear projection from VAE latent space to T5 hidden space.
        t5_to_vae (nn.Linear): Linear projection from T5 hidden space back to VAE latent space.
        padding_token (nn.Parameter): Non-trainable parameter for padding tokens.
        padding_token_threshold (nn.Parameter): Non-trainable parameter for padding token threshold.
        vae (AutoencoderKL): Pre-trained Variational Autoencoder.
        query_rearrange (Rearrange): Layer to rearrange VAE latent representations for queries.
        z_rearrange (Rearrange): Layer to rearrange T5 outputs back to VAE latent dimensions.
        mse_criterion (nn.MSELoss): Mean squared error loss function.
    """
    config_class = EmuruConfig

    def __init__(self, config: EmuruConfig) -> None:
        """
        Initialize the Emuru model.
        Args:
            config (EmuruConfig): Configuration object containing model hyperparameters and paths.
        """
        super().__init__(config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)

        t5_config = T5Config.from_pretrained(config.t5_name_or_path)
        t5_config.vocab_size = len(self.tokenizer)
        self.T5 = T5ForConditionalGeneration(t5_config)
        self.T5.lm_head = nn.Identity()
        self.sos = nn.Embedding(1, t5_config.d_model)

        vae_latent_size = 8 * config.vae_channels * config.slices_per_query
        self.vae_to_t5 = nn.Linear(vae_latent_size, t5_config.d_model)
        self.t5_to_vae = nn.Linear(t5_config.d_model, vae_latent_size, bias=False)

        self.padding_token = nn.Parameter( torch.tensor([[-0.4951,  0.8021,  0.3429,  0.5622,  0.5271,  0.5756,  0.7194,  0.6150]]), requires_grad=False)
        self.padding_token_threshold = nn.Parameter(torch.tensor(0.484982096850872), requires_grad=False)

        self.query_rearrange = Rearrange('b c h (w q) -> b w (q c h)', q=config.slices_per_query)
        self.z_rearrange = Rearrange('b w (q c h) -> b c h (w q)', c=config.vae_channels, q=config.slices_per_query)

        self.style_enc = config.style_enc if hasattr(config, 'style_enc') else "mean"
        if self.style_enc == "MLP": # w -> 1
            self.style_encoder = nn.Linear(vae_latent_size, 1)
        elif self.style_enc == "MLP2":
            self.style_encoder = nn.Sequential(
                nn.Linear(vae_latent_size, vae_latent_size),
                nn.SiLU(),
                nn.Linear(vae_latent_size, 1)
            )

        self.use_start_latent = config.use_start_latent if hasattr(config, 'use_start_latent') else "True"
        if self.use_start_latent.lower() == "true":
            self.use_start_latent = True
        else:
            self.use_start_latent = False

        self.mse_criterion = nn.MSELoss()
        self.init_weights()

        self.vae = AutoencoderKL.from_pretrained(config.vae_name_or_path)
        self.set_training(self.vae, False)

    def set_training(self, model: nn.Module, training: bool) -> None:
        """
        Set the training mode for a given model and freeze/unfreeze parameters accordingly.
        Args:
            model (nn.Module): The model to set the training mode for.
            training (bool): If True, set the model to training mode; otherwise, evaluation mode.
        """
        model.train() if training else model.eval()
        for param in model.parameters():
            param.requires_grad = training

    def forward_nonAR(
        self,
        img: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        noise: float = 0,
        label_img: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        Args:
            img (Optional[torch.Tensor]): Input Style image tensor.
            input_ids (Optional[torch.Tensor]): Tokenized input IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask for the inputs.
            noise (float): Amount of noise to add in image encoding.
            **kwargs: Additional arguments.
        Returns:
            Tuple containing:
                - mse_loss (torch.Tensor): Mean squared error loss.
                - pred_latent (torch.Tensor): Predicted latent representations.
                - z (torch.Tensor): Sampled latent vector from VAE.
        """
        decoder_inputs_embeds, z_sequence, z = self._img_encode(img, noise)

        output = self.T5(input_ids, attention_mask=attention_mask, decoder_inputs_embeds=decoder_inputs_embeds)
        vae_latent = self.t5_to_vae(output.logits[:, :-1])
        pred_latent = self.z_rearrange(vae_latent)

        if self.training:
            assert label_img is not None, 'label_img must be provided during training'
            posterior_label = self.vae.encode(label_img.float())
            z_label = posterior_label.latent_dist.sample()
            z_label_sequence = self.query_rearrange(z_label)

            # Fix: Ensure sequence lengths match for loss computation
            min_seq_len = min(vae_latent.size(1), z_label_sequence.size(1))
            vae_latent_trimmed = vae_latent[:, :min_seq_len]
            z_label_sequence_trimmed = z_label_sequence[:, :min_seq_len]    

            mse_loss = self.mse_criterion(vae_latent_trimmed, z_label_sequence_trimmed)
        else:
            mse_loss = torch.tensor(0.0, device=self.device)
        return mse_loss, pred_latent, z
    
    def forward(
        self,
        img: Optional[torch.Tensor] = None,
        label_img: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        style_noise: float = 0,
        label_noise: float = 0,
        teacher_p: float = 1.0,
        teacher_w: float = 1.0,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        Args:
            img (Optional[torch.Tensor]): Input Style image tensor.
            input_ids (Optional[torch.Tensor]): Tokenized input IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask for the inputs.
            noise (float): Amount of noise to add in image encoding.
            **kwargs: Additional arguments.
        Returns:
            Tuple containing:
                - mse_loss (torch.Tensor): Mean squared error loss.
                - pred_latent (torch.Tensor): Predicted latent representations.
                - z (torch.Tensor): Sampled latent vector from VAE.
        """
        assert label_img is not None, 'label_img must be provided during training'

        posterior_style = self.vae.encode(img.float())
        z_style = posterior_style.latent_dist.sample()
        z_style_sequence = self.query_rearrange(z_style)
        if style_noise > 0:
            z_style_sequence = z_style_sequence + torch.randn_like(z_style_sequence) * style_noise

        if self.style_enc == "mean":
            style_global = z_style_sequence.mean(dim=1, keepdim=True)  # (b, 1, d)
        elif self.style_enc in ["MLP", "MLP2"]:
            style_scores = self.style_encoder(z_style_sequence)  # (b, w, 1)
            style_weights = torch.softmax(style_scores, dim=1)  # (b, w, 1)
            style_global = (z_style_sequence * style_weights).sum(dim=1, keepdim=True)  # (b, 1, d)
        else:
            raise ValueError(f"Unknown style_enc type: {self.style_enc}")
        
        style_token_embed = self.vae_to_t5(style_global)  # (b, 1, t5_d_model)

        posterior_label = self.vae.encode(label_img.float())
        z_label = posterior_label.latent_dist.sample()
        z_label_sequence = self.query_rearrange(z_label) # (b, w, d)

        if label_noise > 0:
            z_label_sequence_noisy = z_label_sequence + torch.randn_like(z_label_sequence) * label_noise
        else:
            z_label_sequence_noisy = z_label_sequence

        sos = repeat(self.sos.weight, '1 d -> b 1 d', b=input_ids.size(0))

        if teacher_p >= 1.0 and teacher_w >= 1.0:        
            label_embeds = self.vae_to_t5(z_label_sequence_noisy)  # (b, w, t5_d_model)
            decoder_inputs_embeds = torch.cat(
                [sos, style_token_embed, label_embeds[:, :-1]], dim=1
            ) # (b, 1 + 1 + w - 1, t5_d_model)

            output = self.T5(input_ids, attention_mask=attention_mask, decoder_inputs_embeds=decoder_inputs_embeds) # (b, 2+l_pred -1, t5_d_model)
            all_vae_latent = self.t5_to_vae(output.logits) # (b, 2 + l_pred -1, vae_latent_size)

            if self.use_start_latent:
                vae_latent = all_vae_latent[:, 1:, :] # [z_0, ..., z_{l_pred -1}] (b, l_pred, vae_latent_size)
            else:
                vae_latent = all_vae_latent[:, 2:, :]  # Remove sos and style token (b, l_pred -1, vae_latent_size) [z_1, ..., z_{l_pred -1}]
                z_label_sequence = z_label_sequence[:, 1:, :] # [z_1, ..., z_{l_pred -1}]
        else:
            seq_len = z_label_sequence_noisy.size(1)
            decoder_inputs_embeds = torch.cat(
                [sos, style_token_embed], dim=1
            ) # (b, 1 + 1, t5_d_model)

            vae_latent_list = []

            for t in range(seq_len):
                output = self.T5(input_ids, attention_mask=attention_mask, decoder_inputs_embeds=decoder_inputs_embeds) # (b, 2+i, t5_d_model)
                last_hidden = output.logits[:, -1:, :]  # (b, 1, t5_d_model)
                vae_latent_t = self.t5_to_vae(last_hidden) # (b, 1, vae_latent_size)
                vae_latent_list.append(vae_latent_t)

                gt_t = z_label_sequence_noisy[:, t:t+1, :]  # (b, 1, vae_latent_size)

                if teacher_w >= 1.0:
                    if teacher_p <= 0.0:
                        next_input = vae_latent_t.detach()
                    else:
                        mask = (torch.rand_like(gt_t[..., :1]) < teacher_p).to(vae_latent_t.dtype)  # (b, 1, 1)
                        next_input = mask * gt_t + (1 - mask) * vae_latent_t.detach()
                
                elif teacher_w <= 0.0:
                    next_input = vae_latent_t.detach()
                else:
                    next_input = teacher_w * gt_t + (1 - teacher_w) * vae_latent_t.detach() # (b, 1, vae_latent_size)

                next_input_embeds = self.vae_to_t5(next_input)  # (b, 1, t5_d_model)
                decoder_inputs_embeds = torch.cat(
                    [decoder_inputs_embeds, next_input_embeds], dim=1
                ) # (b, 2 + t + 1, t5_d_model)

            vae_latent = torch.cat(vae_latent_list, dim=1)  # (b, l_pred, vae_latent_size)

        # Fix: Ensure sequence lengths match for loss computation
        min_seq_len = min(vae_latent.size(1), z_label_sequence.size(1))
        vae_latent_trimmed = vae_latent[:, :min_seq_len]
        z_label_sequence_trimmed = z_label_sequence[:, :min_seq_len]    

        mse_loss = self.mse_criterion(vae_latent_trimmed, z_label_sequence_trimmed)
        pred_latent = self.z_rearrange(vae_latent_trimmed)

        return mse_loss, pred_latent, z_label

    @torch.inference_mode()
    def generate(
        self,
        gen_text: str,
        style_img: torch.Tensor,
        **kwargs: Any
    ) -> Image.Image:
        """
        Generate an image by combining style and generation texts with a style image.
        Args:
            style_text (str): Style-related text prompt.
            gen_text (str): Generation-related text prompt.
            style_img (torch.Tensor): Style image tensor. Expected shape is either 3D or 4D.
            **kwargs: Additional keyword arguments.
        Returns:
            Image.Image: Generated image as a PIL image.
        """
        if style_img.ndim == 3:
            style_img = style_img.unsqueeze(0)
        elif style_img.ndim == 4: # (b, c, h, w)
            pass
        else:
            raise ValueError('style_img must be 3D or 4D')
        
        imgs, _ = self._generate(texts=[gen_text], imgs=style_img, **kwargs)
        imgs = (imgs + 1) / 2
        return F.to_pil_image(imgs[0].detach().cpu())
    
    @torch.inference_mode()
    def generate_batch(
        self,
        gen_texts: List[str],
        style_imgs: torch.Tensor,
        **kwargs: Any
    ) -> List[Image.Image]:
        """
        Generate a batch of images from lists of style texts, generation texts, and style images.
        Args:
            style_texts (List[str]): List of style-related text prompts.
            gen_texts (List[str]): List of generation-related text prompts.
            style_imgs (torch.Tensor): Batch of style images (4D tensor).
            lengths (List[int]): List of lengths corresponding to each image.
            **kwargs: Additional keyword arguments.
        Returns:
            List[Image.Image]: List of generated images as PIL images.
        """
        assert style_imgs.ndim == 4, 'style_imgs must be 4D'
        assert len(gen_texts) == len(style_imgs), 'gen_texts and style_imgs must have the same length'
        texts = [gen_text for gen_text in gen_texts]
        
        imgs, _ = self._generate(texts=texts, imgs=style_imgs, **kwargs)
        imgs = (imgs + 1) / 2

        out_imgs = []
        for i in range(imgs.size(0)):
            out_imgs.append(F.to_pil_image(imgs[i].detach().cpu()))
        return out_imgs

    def _generate(
        self,
        texts: Optional[List[str]] = None,
        imgs: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        z_sequence: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        stopping_criteria: str = 'latent',
        stopping_after: int = 10,
        stopping_patience: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal generation routine that combines textual and visual inputs to iteratively generate
        latent representations and decode them into images.
        Args:
            texts (Optional[List[str]]): List of text prompts.
            imgs (Optional[torch.Tensor]): Input image tensor.
            lengths (Optional[List[int]]): Desired lengths for each image in latent space.
            input_ids (Optional[torch.Tensor]): Tokenized input IDs.
            z_sequence (Optional[torch.Tensor]): Precomputed latent sequence.
            max_new_tokens (int): Maximum tokens to generate.
            stopping_criteria (str): Criteria for stopping ('latent' or 'none').
            stopping_after (int): Number of tokens to check for stopping condition.
            stopping_patience (int): Patience parameter for stopping condition.
        Returns:
            Tuple containing:
                - imgs (torch.Tensor): Generated images.
                - canvas_sequence (torch.Tensor): Generated latent canvas sequence.
                - img_ends (torch.Tensor): End indices for each generated image.
        """
        assert texts is not None or input_ids is not None, 'Either texts or input_ids must be provided'
        assert imgs is not None or z_sequence is not None, 'Either imgs or z_sequence must be provided'

        if input_ids is None:
            input_ids = self.tokenizer(texts, return_tensors='pt', padding=True).input_ids
            input_ids = input_ids.to(self.device)

        if z_sequence is None:
            posterior_style = self.vae.encode(imgs.float())
            z_style = posterior_style.latent_dist.sample()
            z_style_sequence = self.query_rearrange(z_style)
            z_sequence = z_style_sequence
        
        if self.style_enc == "mean":
            style_global = z_sequence.mean(dim=1, keepdim=True)  # (b, 1, d)
        elif self.style_enc == "MLP" or self.style_enc == "MLP2":
            style_scores = self.style_encoder(z_sequence)  # (b, w, 1)
            style_weights = torch.softmax(style_scores, dim=1)  # (b, w, 1)
            style_global = (z_sequence * style_weights).sum(dim=1, keepdim=True)  # (b, 1, d)
        else:
            raise ValueError(f"Unknown style_enc type: {self.style_enc}")
        
        style_token_embed = self.vae_to_t5(style_global)  # (b, 1, t5_d_model)

        # prepare for decoder input
        sos = repeat(self.sos.weight, '1 d -> b 1 d', b=input_ids.size(0))
        pad_token = repeat(self.padding_token, '1 d -> b 1 d', b=input_ids.size(0))

        generated_latents: List[torch.Tensor] = []
        active = torch.ones(input_ids.size(0), dtype=torch.bool, device=self.device) # for batch processing

        for step in range(max_new_tokens):
            if len(generated_latents) == 0:
                decoder_inputs_embeds = torch.cat([sos, style_token_embed], dim=1) # (b, 2, t5_d_model)
            else:
                lat_seq = torch.stack(generated_latents, dim=1)  # (b, t, vae_latent_size)
                lat_embeds = self.vae_to_t5(lat_seq)  # (b, t, t5_d_model)
                decoder_inputs_embeds = torch.cat([sos, style_token_embed, lat_embeds], dim=1) # (b, 2 + t, t5_d_model)
        
            output = self.T5(input_ids, decoder_inputs_embeds=decoder_inputs_embeds)
            last_hidden = output.logits[:, -1:, :]  # (b, 1, t5_d_model)
            vae_latent = self.t5_to_vae(last_hidden)[:, 0, :]  # (b, vae_latent_size)

            if stopping_criteria == 'latent' and (~active).any():
                vae_latent = torch.where(
                    active.unsqueeze(-1),
                    vae_latent,
                    pad_token.squeeze(1)
                )
            
            generated_latents.append(vae_latent) 
            canvas_sequence = torch.stack(generated_latents, dim=1)  # (b, t+1, vae_latent_size)

            if stopping_criteria == 'latent':
                similarity = torch.nn.functional.cosine_similarity(canvas_sequence, pad_token, dim=-1) # (b, t+1)
                if similarity.size(1) >= stopping_after:
                    window = similarity[:, -stopping_after:] # (b, stopping_after)
                    cnt = (window > self.padding_token_threshold).to(torch.int).sum(dim=1)  # (b,)
                    new = (cnt >= (stopping_after - stopping_patience)) & active  # (b,)
                    active = active & (~new)

                    if not active.any():
                        break

            elif stopping_criteria == 'none':
                pass

        canvas_sequence = torch.stack(generated_latents, dim=1)  # (b, t, vae_latent_size)
        imgs = torch.clamp(self.vae.decode(self.z_rearrange(canvas_sequence)).sample, -1, 1)
        return imgs, canvas_sequence
    
    def _img_encode(
        self,
        img: torch.Tensor,
        noise: float = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode the input image into a latent representation using the VAE.
        Args:
            img (torch.Tensor): Input image tensor.
            noise (float): Standard deviation of noise to add to the latent sequence.
        Returns:
            Tuple containing:
                - decoder_inputs_embeds (torch.Tensor): Embeddings to be used as T5 decoder inputs.
                - z_sequence (torch.Tensor): Rearranged latent sequence from the VAE.
                - z (torch.Tensor): Sampled latent vector from the VAE.
        """
        posterior = self.vae.encode(img.float())
        z = posterior.latent_dist.sample()
        z_sequence = self.query_rearrange(z)

        noise_sequence = z_sequence
        if noise > 0:
            noise_sequence = z_sequence + torch.randn_like(z_sequence) * noise

        decoder_inputs_embeds = self.vae_to_t5(noise_sequence)
        sos = repeat(self.sos.weight, '1 d -> b 1 d', b=decoder_inputs_embeds.size(0))
        decoder_inputs_embeds = torch.cat([sos, decoder_inputs_embeds], dim=1)
        return decoder_inputs_embeds, z_sequence, z

    def compute_padding_token(self) -> None:
        """
        Compute and update the padding token.
        Raises:
            NotImplementedError: This method must be implemented.
        """
        raise NotImplementedError("compute_padding_token not implemented")

    def compute_padding_token_threshold(self) -> None:
        """
        Compute and update the padding token threshold.
        Raises:
            NotImplementedError: This method must be implemented.
        """
        raise NotImplementedError("compute_padding_token_threshold not implemented")
