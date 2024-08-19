import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc
# for 杂交版
def clip_classifier_2(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            #class_embeddings = clip_model.encode_text(texts)
            prompts = clip_model.prompt_learner()
            tokenized_prompts = clip_model.prompt_learner.tokenized_prompts
            class_embeddings = clip_model.text_encoder(prompts, tokenized_prompts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            # 将归一化的类别嵌入向量添加到 clip_weights 列表中。
            clip_weights.append(class_embedding)
        # 使用 torch.stack 将 clip_weights 列表中的嵌入向量堆叠成一个张量，并使用 .cuda() 将其移动到GPU上。
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

# def get_clip_logits_2(images, clip_model, clip_weights):
#     with torch.no_grad():
#         if isinstance(images, list):
#             images = torch.cat(images, dim=0).cuda()
#         else:
#             images = images.cuda()
#
#         image_features = clip_model.image_encoder(images)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#
#         image_features = image_features.to(clip_weights.dtype)
#         clip_logits = 100. * image_features @ clip_weights
#
#         if image_features.size(0) > 1:
#             batch_entropy = softmax_entropy(clip_logits)
#             selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
#             output = clip_logits[selected_idx]
#             image_features = image_features[selected_idx].mean(0).unsqueeze(0)
#             clip_logits = output.mean(0).unsqueeze(0)
#
#             loss = avg_entropy(output)
#             prob_map = output.softmax(1).mean(0).unsqueeze(0)
#             pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
#         else:
#             loss = softmax_entropy(clip_logits)
#             prob_map = clip_logits.softmax(1)
#             pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])
#
#         return image_features, clip_logits, loss, prob_map, pred
# For 杂交版 2.0 找最可能的top-k个pred_list
def get_clip_logits_2(images, clip_model, clip_weights, k=1):
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()

        image_features = clip_model.image_encoder(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        image_features = image_features.to(clip_weights.dtype)

        clip_logits = 100. * image_features @ clip_weights
        #print("clip_logits",clip_logits.shape) # 64 * clsn
        #clip_logits = clip_model(images)
        #print("logits",logits.shape) # 64 * clsn

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            _, pred_indices = output.mean(0).unsqueeze(0).topk(k, 1, True, True)
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            _, pred_indices = clip_logits.topk(k, 1, True, True)

        # Return type based on k
        if k == 1:
            pred = int(pred_indices.squeeze().item())
        else:
            pred = pred_indices.squeeze().tolist()

        return image_features, clip_logits, loss, prob_map, pred
# for 杂交版 动态维护clip_weights
def update_weights(clip_model, clip_weights, idx, iter, alpha=0.99):
    with torch.no_grad():
        alpha_teacher = max(1 - 1 / (iter + 1), alpha)
        # alpha_teacher = alpha
        # Get new text features for the specified class
        text_features = clip_model.get_text_features_one(idx)  # Size([n_cls, 512])
        #print("text_features",text_features.shape) # text_features torch.Size([1, 512])
        #text_feat = text_features[idx, :].squeeze(0)  # Size([512])
        text_feat =  text_features.squeeze(0)

        # Get the old text features for the specified class
        old_text_feat = clip_weights[:, idx]  # Size([512])
        #print("old_text_feat",old_text_feat.shape)

        # Update using EMA: new_weight = alpha * old_weight + (1 - alpha) * new_weight
        updated_text_feat = alpha_teacher * old_text_feat + (1 - alpha_teacher) * text_feat

        # Update the weights tensor with the new features
        clip_weights[:, idx] = updated_text_feat

    return clip_weights





def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            # 将归一化的类别嵌入向量添加到 clip_weights 列表中。
            clip_weights.append(class_embedding)
        # 使用 torch.stack 将 clip_weights 列表中的嵌入向量堆叠成一个张量，并使用 .cuda() 将其移动到GPU上。
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def get_clip_logits(images, clip_model, clip_weights):
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()

        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        clip_logits = 100. * image_features @ clip_weights

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred


def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "K"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess):
    if dataset_name == 'I':
        #print("root_path",root_path)
        preprocess = get_ood_preprocess()
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=8, shuffle=True)
    
    elif dataset_name in ['A','V','R','K']:
        preprocess = get_ood_preprocess() # 这个东西导致进行了一个图像增广 64
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        preprocess = get_ood_preprocess()
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    
    return test_loader, dataset.classnames, dataset.template