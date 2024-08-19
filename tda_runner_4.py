# 杂交版 3.0
import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime
import copy
import torch
import torch.nn.functional as F
import operator

import clip
from utils.utils_tda import *
# 杂交版
from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop # clip.custom_clip
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
# for TPT
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
# for TDA
def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')
    # TPT
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print_freq', default=1000, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    return args
# for 杂交版
def deep_copy_parameters(parameters):
    return [param.clone().detach() for param in parameters]
# for 杂交版 prompt learning
# test_time_tuning(model, images, optimizer, scaler, args)
# pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args)
# inputs: 输入数据，对于cocoop模型是 (图像特征, pgen_ctx) 对，对于其他模型是直接的输入图像。
# scaler: 用于自动混合精度（AMP）的 GradScaler 对象，帮助管理梯度的缩放。
# args: 包含运行参数的命名空间或对象。
def test_time_tuning(model, inputs, optimizer, scaler, args):
    # 说明是cocoop模型，需要特殊处理。将 pgen_ctx 设置为需要梯度，并创建一个新的优化器仅针对 pgen_ctx。
    # if inputs.dtype == torch.float32:
    #     print("Inputs are in single precision (FP32).")
    # elif inputs.dtype == torch.float16:
    #     print("Inputs are in half precision (FP16).")
    # else:
    #     print("Inputs are in a different precision.")
    # for name, param in model.named_parameters():
    #     print(name, param.dtype)  # 确保所有参数都是torch.float32
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)

    # selected_idx 用于存储选择的置信度高的样本的索引，初始为 None。
    selected_idx = None
    # 循环 args.tta_steps 次，每次都会执行前向传播，选择置信度高的样本，并执行优化步骤。
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            # 根据是否是cocoop模型，调用模型生成输出。
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(inputs)
            #print("test_time_tuning output 1", output.shape)

            # 如果之前已经选择了样本，则只考虑这些样本的输出；
            if selected_idx is not None:
                output = output[selected_idx]
            # 否则，使用 select_confident_samples 函数基于输出的熵选择置信度高的样本。
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)
            #print("test_time_tuning output 2", output.shape)
            loss = avg_entropy(output)


        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    # 如果是cocoop模型，返回调整后的 pgen_ctx；否则，函数不返回任何值。
    if args.cocoop:
        return pgen_ctx

    return
# 杂交版2.0 之简单替换
def replace_prompt(model, pred, pro_cache):

    if pred in pro_cache:
        # 获取指定类别的 prompt 和 loss
        prompt, loss = pro_cache[pred][0]  # 选第一个 一般loss最小
        # print("in replace_prompt")
        # for param in prompt:
        #     print(param.data)
        # print("tes*"*40)
        # 确保 prompt 的格式与模型的参数格式匹配
        # 通常需要将 prompt 转换为与 model.prompt_learner.parameters() 兼容的形式
        with torch.no_grad():
            for param, new_value in zip(model.prompt_learner.parameters(), prompt):
                # print("param:",param)
                # print("new_value:",new_value)
                param.copy_(new_value)
def check_and_return_existing(preds, cache):
    existing = [pred for pred in preds if pred in cache]
    return bool(existing), existing
# 杂交版 2.0 之cat替换
def replace_prompt_2(model, preds, pro_cache):
    exists, existing_ints = check_and_return_existing(preds, pro_cache)
    if exists:
        prompts = []
        # 获取指定类别的 prompt 和 loss
        print("*"*30)
        for pred in existing_ints:
            prompt, loss = pro_cache[pred][0]  # 选第一个 一般loss最小
            prompts.append(prompt[0])
            print(f'Pred {pred}= {loss}')
            print(f"Prompt {prompt[0]}")
        #print("prompts", prompts)
        if len(prompts) > 1:
            prompt_cat = torch.cat(prompts, dim=0)
        else:
            prompt_cat = prompts[0]

        print("prompt_cat",prompt_cat.shape)
        # 确保 prompt 的格式与模型的参数格式匹配
        # 通常需要将 prompt 转换为与 model.prompt_learner.parameters() 兼容的形式
        with torch.no_grad():
            param_list = list(model.prompt_learner.parameters())
            # 如果 prompt_cat 的维度大于参数的维度，则重新调整参数
            if prompt_cat.shape[0] > param_list[0].shape[0]:
                # 重新设置参数的形状
                for param in param_list:
                    new_param = torch.empty_like(prompt_cat)
                    new_param.copy_(prompt_cat)
                    param.data = new_param

            for param, new_value in zip(model.prompt_learner.parameters(), prompt_cat):
                # print("param:",param)
                # print("new_value:",new_value)
                param.copy_(new_value)
# for 杂交版 2.0
def update_cache_p(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    f = 0 # 是否有新东西加入
    new_item_index = 0  # Index of the new or updated item
    with torch.no_grad():
        # print("features_loss",type(features_loss))
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        # 每一类 进行一个存储
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
                f = 1

            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
                f = 1

            #cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
            #print(cache)
            new_item_index = len(cache[pred])-1

        else:
            cache[pred] = [item]
            f = 1
    return f, new_item_index
# 杂交版 2，0 new prompt
# if pos_enabled:
#     update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])
# if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
#     update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)
def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        # print("features_loss",type(features_loss))
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        # 每一类 进行一个存储
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)

            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item

            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]

# if pos_enabled and pos_cache:
#     final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
# if neg_enabled and neg_cache:
#     final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = [] # for feats
        cache_values = [] # for label
        # 函数遍历 cache 字典，该字典按类别索引排序，并存储每个类别的缓存项。
        for class_index in sorted(cache.keys()):
            # 对于每个缓存项：
            for item in cache[class_index]:
                cache_keys.append(item[0]) # 将图像特征（键）添加到 cache_keys 列表
                if neg_mask_thresholds: # 根据是否提供了 neg_mask_thresholds，将相应的类别索引或负掩码阈值添加到 cache_values 列表。
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)
        # 将 cache_keys 列表中的图像特征堆叠成一个张量，并进行排列，以便进行矩阵乘法。
        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        # 如果提供了 neg_mask_thresholds，则将 cache_values 转换为一个张量，并根据阈值进行处理，然后将其转换为半精度浮点数并移动到GPU上。
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        # 如果没有提供 neg_mask_thresholds，则使用 torch.one_hot 创建一个独热编码张量，表示类别索引，然后将其转换为半精度浮点数并移动到GPU上。
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()
        # 使用矩阵乘法计算图像特征和缓存键之间的亲和性（affinity）。
        affinity = image_features @ cache_keys
        affinity = affinity.to(cache_values.dtype)
        # 使用亲和性和缓存值计算缓存logits。这里使用了指数函数和alpha缩放因子。
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits
# model_state, optimizer, optim_state, scaler, args for 杂交版
def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, model_state, optimizer, optim_state, scaler, args):
    #with torch.no_grad():
    pos_cache, neg_cache, pro_cache, accuracies = {}, {}, {}, []

    #Unpack all hyperparameters
    pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

    # reset model and switch to evaluate mode
    clip_model.eval()
    if not args.cocoop:  # no need to reset cocoop because it's fixed
        with torch.no_grad():
            clip_model.reset()
    #Test-time adaptation
    for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
        # # 随机摇一个idx分配idx类最好的prompt
        # class_size = clip_weights.size(1)
        # idx= random.randint(0, class_size - 1)
        # with torch.no_grad():
        #     image_features, clip_logits, loss, prob_map, pred = get_clip_logits_2(images, clip_model, clip_weights, 1)
        # print("pred",pred)
        # print("pred[0]",type(pred[0]))
        # clip_model.prompt_learner.update_prompts(pro_cache) # 进行跟新
        assert args.gpu is not None
        
        # for TPT
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)
        if args.tpt:
            images = torch.cat(images, dim=0)
        # reset the tunable prompt to its initial state
        if not args.cocoop:  # no need to reset cocoop because it's fixed
            # if args.tta_steps > 0:
            #     with torch.no_grad():
            #         clip_model.reset()  # reset
            # prompt_tp = deep_copy_parameters(clip_model.prompt_learner.parameters())
            # print("clip_model.prompt_learner.parameters().shape",prompt_tp[0].shape)
            # replace_prompt_2(clip_model, preds, pro_cache)  # 马上改变此时TDA预测一个pred，改变prompt tokens
            # replace_prompt(clip_model, pred, pro_cache)  # 马上改变此时TDA预测一个pred，改变prompt tokens
            # prompt_tp = deep_copy_parameters(clip_model.prompt_learner.parameters())
            # print("after clip_model.prompt_learner.parameters().shape", prompt_tp[0].shape)

            # clip_model.prompt_learner.update_prompts_ema(pro_cache,i)  # 进行跟新
            if i > 1 and add:
                #print("yes")
                clip_model.prompt_learner.update_prompts_ema2(pred, pro_cache, i, new_item_index)
            optimizer.load_state_dict(optim_state)
            #print("before,",clip_model.prompt_learner.ctx_e)
            #print("before ctx,", clip_model.prompt_learner.ctx)
            test_time_tuning(clip_model, images, optimizer, scaler, args)
            #print("after,", clip_model.prompt_learner.ctx_e)
            #print("after ctx,", clip_model.prompt_learner.ctx)
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_feature, pgen_ctx = clip_model.gen_ctx(images, args.tpt)
            optimizer = None
            # replace_prompt_2(clip_model, preds, pro_cache)  # 马上改变此时TDA预测一个pred，改变prompt tokens
            # replace_prompt(clip_model, pred, pro_cache)  # 马上改变此时TDA预测一个pred，改变prompt tokens

            # clip_model.prompt_learner.update_prompts_ema(pro_cache,i)  # 进行跟新

            pgen_ctx = test_time_tuning(clip_model, (image_feature, pgen_ctx), optimizer, scaler, args)

        # for TDA
        with torch.no_grad():

            image_features, clip_logits, loss, prob_map, pred = get_clip_logits_2(images ,clip_model, clip_weights)

            target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)

            pro_enabled = 1
            if pro_enabled:
                # prompt_new len 1
                # prompt_new[0] torch.Size([4, 512])
                prompt_new = deep_copy_parameters(clip_model.prompt_learner.parameters())  # list
                # print("prompt_new",prompt_new)
                # print("prompt_new len", len(prompt_new))
                # print("prompt_new[0]",prompt_new[0].shape)
                add, new_item_index = update_cache_p(pro_cache, pred, [prompt_new, loss], pos_params['shot_capacity'])
                if add: # 选择跟新 clip_weights
                    #print("before,",clip_weights[:,pred])

                    clip_weights = update_weights(clip_model, clip_weights, pred, i)
                    #print("after,", clip_weights[:, pred])
                # if add:
                #     clip_model.prompt_learner.update_prompts_ema2(pred,pro_cache,i)
                # if add:
                #     clip_model.prompt_learner.update_prompts_ema(pro_cache, i)

            if pos_enabled:
                update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)


            final_logits = clip_logits.clone()

            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
            # print("final_logits",final_logits.shape)

            acc = cls_acc(final_logits, target)
            accuracies.append(acc)

            if args.wandb:
                wandb.log({"Averaged test accuracy": sum(accuracies)/len(accuracies)}, commit=True)
        if i % args.print_freq == 0:
            print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
    print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
    return sum(accuracies)/len(accuracies)



def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    # clip_model, preprocess = clip.load(args.backbone)
    # clip_model.eval()
    _, preprocess = clip.load(args.backbone)
    clip_model = get_coop(args.backbone, args.datasets, args.gpu, args.n_ctx, args.ctx_init)
    if args.load is not None:
        print("Use pre-trained soft prompt (CoOp) as initialization")
        pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
        assert pretrained_ctx.size()[0] == args.n_ctx
        with torch.no_grad():
            clip_model.prompt_learner[0].ctx.copy_(pretrained_ctx)
            clip_model.prompt_learner[0].ctx_init_state = pretrained_ctx
    model_state = None

    # for param in clip_model.prompt_learner.parameters():
    #     if param.dtype == torch.float16:
    #         param.data = param.data.float()  # 将 FP16 参数转换为 FP32

    for name, param in clip_model.named_parameters():
        # 全部干成FP32
        if param.dtype == torch.float16:
            param.data = param.data.float()

        #print("main:", name, param.dtype)  # 确保所有参数都是torch.float32
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
    print("=> Model created: visual backbone {}".format(args.backbone))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        clip_model = clip_model.cuda(args.gpu)
    # define optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        trainable_param = clip_model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"
    
    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name) # 获取数据指定config
        print("\nRunning dataset configurations:")
        print(cfg, "\n")

        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        #print("test_loader",test_loader)
        #print("test_loader", test_loader)
        # print("classnames2",classnames)
        # print("classnames2", len(classnames))
        #print("template",template)
        #print("nls 0",clip_model.prompt_learner.n_cls )
        clip_model.prompt_learner.n_cls = len(classnames)
        #print("nls len(classnames)", clip_model.prompt_learner.n_cls)
        if not args.cocoop:  # no need to reset cocoop because it's fixed
            clip_model.prompt_learner.reset_ctx_e()
        #print("nls reset_ctx_e", clip_model.prompt_learner.n_cls)

        if args.cocoop:
            clip_model.prompt_generator.reset_classnames(classnames, args.arch)
            clip_model = clip_model.cpu()
            model_state = clip_model.state_dict()
            clip_model = clip_model.cuda(args.gpu)
        else:
            clip_model.reset_classnames(classnames, args.backbone)
        #print("after nls", clip_model.prompt_learner.n_cls)
        # 将归一化的类别嵌入向量添加到 clip_weights 列表中。
        # clip_weights = clip_classifier_2(classnames, template, clip_model)
        # print("clip_weights",clip_weights)
        # print("clip_weights",clip_weights.shape)
        # Initialize CLIP model
        clip_model_0, preprocess_0 = clip.load(args.backbone)
        clip_model_0.eval()
        clip_weights = clip_classifier(classnames, template, clip_model_0)

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)
        #with torch.cuda.amp.autocast():
        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, model_state, optimizer, optim_state, scaler, args)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()