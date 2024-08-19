
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        # 初始化时，加载预训练的CLIP模型，并删除transformer部分，只保留视觉编码器。
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        # 定义了一个分类头（cls_head），用于将编码器的输出映射到类别数量。
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # # 使用CLIP模型的transformer、positional_embedding、ln_final和text_projection。
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    # 处理提示（prompts）和分词后的提示（tokenized_prompts），生成文本特征。
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x.to(self.text_projection.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

# 已被2.0杂交版魔改
# 可以初始化上下文向量（context vectors）和类别标记（class tokens）。
# 提供了重置（reset）和更新类别名称（reset_classnames）的方法。
# forward方法构建了用于文本编码的提示。
class PromptLearner(nn.Module):
    # 初始化时，接受CLIP模型、类别名称等参数。
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = n_ctx
        self.batch_size = batch_size
        #self.ctx_e = torch.zeros(n_cls, n_ctx, n_ctx)  # 初始化 ctx
        #self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)
        # print("in PL self.ctx",self.ctx.shape)
        # print("in PL self.ctx", self.ctx)
        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized *******************
        # print(f"Initial context: {self.ctx.shape}") # torch.Size([4, 512])
        # print(f"Initial context: {self.ctx}")

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames


        # print("in PL", ctx.shape) # torch.Size([cls_n ,4, 512])
        if self.ctx.dim() == 2:
            self.ctx_e = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not self.ctx.size()[0] == self.n_cls:
            self.ctx_e = self.ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)


    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        self.ctx_e = self.ctx_e[:self.n_cls]
        #print("self.ctx_e ",self.ctx_e.shape )
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames
    def reset_ctx_e(self):
        if self.ctx.dim() == 2:
            self.ctx_e = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1).clone()
        elif not self.ctx.size()[0] == self.n_cls:
            self.ctx_e = self.ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1).clone()

    def update_prompts(self, pro_cache):
        if pro_cache is not None:
            # 遍历 pro_cache 中的键和值
            self.ctx_e = self.ctx_e.clone()
            for key, value in pro_cache.items():
                #print('key',key)
                # 确保 value 是一个列表且第一个元素是 (4, 512) 的张量
                assert isinstance(value, list) and len(value) > 0, "Value must be a list with at least one element"
                #print("values",value)
                assert value[0][0][0].shape == (4, 512), "The first element of value must be a tensor with shape (4, 512)"

                # 替换 ctx 中第 key 行的张量
                # self.ctx_e[key] = value[0][0][0]
                with torch.no_grad():
                    for param, new_value in zip(self.ctx_e[key], value[0][0][0]):
                        # print("param:",param)
                        # print("new_value:",new_value)
                        param.copy_(new_value)
            #     print("each",key,self.ctx_e[key])
            # print("update_prompts[0]",self.ctx_e[0])
            # print("update_prompts[88]",self.ctx_e[88])
    def update_prompts_ema(self, pro_cache, iter):
        if pro_cache is not None:
            # 遍历 pro_cache 中的键和值
            self.ctx_e = self.ctx_e.clone()
            for key, value in pro_cache.items():
                #print('key',key)
                # 确保 value 是一个列表且第一个元素是 (4, 512) 的张量
                assert isinstance(value, list) and len(value) > 0, "Value must be a list with at least one element"
                #print("values",value)
                assert value[0][0][0].shape == (4, 512), "The first element of value must be a tensor with shape (4, 512)"

                # 替换 ctx 中第 key 行的张量
                # self.ctx_e[key] = value[0][0][0]
                with torch.no_grad():
                    alpha_teacher = min(1 - 1 / (iter + 1), 0.999)
                    for param, new_value in zip(self.ctx_e[key], value[0][0][0]):
                        # print("param:",param)
                        # print("new_value:",new_value)
                        # param.copy_(new_value)

                        param.copy_(alpha_teacher * param + (1 - alpha_teacher) * new_value)
            #     print("each",key,self.ctx_e[key])
            # print("update_prompts[0]",self.ctx_e[0])
            # print("update_prompts[88]",self.ctx_e[88])
    def update_prompts_ema2(self, pred, pro_cache, iter, idx=0, alpha = 0.99):

        # 遍历 pro_cache 中的键和值
        self.ctx_e = self.ctx_e.clone()
        # for key, value in pro_cache.items():
        key = pred
        value = pro_cache[key]
        prompt = value[idx][0][0]
        #print('key',key)
        # 确保 value 是一个列表且第一个元素是 (4, 512) 的张量
        assert isinstance(value, list) and len(value) > 0, "Value must be a list with at least one element"
        #print("values",value)
        assert prompt.shape == (4, 512), "The first element of value must be a tensor with shape (4, 512)"

        # 替换 ctx 中第 key 行的张量
        # self.ctx_e[key] = value[0][0][0]

        with torch.no_grad():
            alpha_teacher = min(1 - 1 / (iter + 1), alpha)
            # alpha_teacher = alpha
            #print("key",key)
            #print('before',self.ctx_e[key])
            for param, new_value in zip(self.ctx_e[key], prompt ):
                param.copy_(alpha_teacher * param + (1 - alpha_teacher) * new_value)
            #print('after',self.ctx_e[key])
    # print("update_prompts[0]",self.ctx_e[0])
    # print("update_prompts[88]",self.ctx_e[88])

    def forward(self, init=None):
        # print("in for",self.n_cls) 100

        # the init will be used when computing CLIP directional loss
        # if init is not None:
        #     ctx = init
        # else:
        #     ctx = self.ctx
        # # print("in PL", ctx.shape) # torch.Size([4, 512])
        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        # elif not ctx.size()[0] == self.n_cls:
        #     ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        # print("in PL",ctx.shape) # torch.Size([cls_n, 4, 512])
        # update prompt for each class with cache
        ctx = self.ctx_e
        # print("forwad",ctx)


        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        # print("in PL prompts",prompts.shape) # prompts torch.Size([100, 77, 512])
        return prompts

# 初始化时，加载CLIP模型，创建图像和文本编码器，以及提示学习器。
# 提供了重置提示学习器状态（reset）和更新类别名称（reset_classnames）的方法。
# get_text_features方法用于获取文本特征。
# inference方法用于图像特征的推理。
# forward方法根据不同的输入执行不同的调整策略。
class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(ClipTestTimeTuning, self).__init__()
        clip, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)
    def get_text_features_one(self, idx):
        text_features = []
        prompts = self.prompt_learner()[idx,:,:].unsqueeze(0) # tesnor torch.Size([1, 77, 512])
        #print("in get_text_features")
        # print("prompt", prompts)
        #print("prompt", prompts.shape)
        tokenized_prompts = self.prompt_learner.tokenized_prompts[idx,:].unsqueeze(0) # tensor torch.Size([1, 77])
        #print("tokenized_prompts",tokenized_prompts.shape)
        # print("tokenized_prompts", tokenized_prompts)
        t_features = self.text_encoder(prompts, tokenized_prompts)
        #print("t_features", t_features.shape)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)
        #print(f"text_features{text_features.shape}")
        #print(f"returne{torch.mean(text_features, dim=0).shape}")
        return torch.mean(text_features, dim=0) # Size([n_cls, 512])

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner() # tesnor torch.Size([cls_n, 77, 512])
        # print("in get_text_features")
        # print("prompt", prompts)
        # print("prompt", prompts.shape)
        tokenized_prompts = self.prompt_learner.tokenized_prompts # tensor torch.Size([cls_n, 77])
        # print("tokenized_prompts",tokenized_prompts.shape)
        # print("tokenized_prompts", tokenized_prompts)
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)
        # print(f"get_text_features{text_features.shape}")
        return torch.mean(text_features, dim=0) # Size([1, n_cls, 512])

    def inference(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        # print("logits", logits.shape)
        return logits # torch.Size([64, 1000])

    def forward(self, input):
        if isinstance(input, Tuple):
            #print(1)
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            #print(2)
            return self.directional_prompt_tuning(input)
        else:
            #print(3)
            return self.inference(input)

# 根据测试集和参数，初始化并返回一个ClipTestTimeTuning模型。
def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model

