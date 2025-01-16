import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
import math
import os
import sys
import time
import traceback

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torchvision import models, transforms
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns

import config
import sketch_utils as utils
from models.loss import Loss
from models.painter_params import Painter, PainterOptimizer
from IPython.display import display, SVG
import model_rnngcn
import display_img
from torch.utils.tensorboard import SummaryWriter
import torchvision
def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_paths, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    # 利用 U2Net 模型生成图像的蒙版 mask，遮蔽后的图像 masked_im
    masked_im, mask = utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)
    # 图像缩放调整
    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_, mask

#计算余弦相似度
def cos_Adjmatrix(feature):
    #feature:matrix[num_feature,vec_dim] eg.16,8
    # 计算 L2 范数
    norms = torch.norm(feature, p=2, dim=1, keepdim=True)  # [16, 1]
    # 标准化
    feature_normalized = feature / norms  # [16, 8]
    # 计算两两余弦相似度
    cos_max = torch.mm(feature_normalized, feature_normalized.T)  # [16, 16]
    # for i in range(cos_max.shape[0]):
        # print(f"cos_max: {i,cos_max[i]}")
    winner_1 = F.one_hot(torch.argmax(cos_max, dim=0), num_classes=16).T
    # print(f"the winner_1 is {winner_1}")
    winner_2 = F.one_hot(torch.argmax(cos_max - cos_max * winner_1, dim=0), num_classes=16).T
    # print(f"the winner_2 is {winner_2}")
    winner_3 = F.one_hot(torch.argmax(cos_max - cos_max * (winner_1 + winner_2), dim=0), num_classes=16).T
    # print(f"w2转置后给w3得出的第三大的值{torch.argmax(cos_max - cos_max * (winner_1 + winner_2_nex), dim=0)}")
    reg_matrix = cos_max * (0.7*winner_1 + winner_2 * 0.2 + winner_3 * 0.1)
    return reg_matrix,cos_max

#提取特征control_points:list
def point_render(control_points):
    img_beziers = []
    for i in range(len(control_points)):
        #1
        points = control_points[i].clone()
        #获得图片
        img = display_img.render(points, 224, 224)

        img = img.unsqueeze(0)  # 调整为 [batch_size, channels, height, width]
        img_beziers.append(img)

        # [num_beizer,features]
    img_beziers = torch.cat(img_beziers, dim=0)
    return img_beziers
def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter grad:\n{param.grad}")  # 打印参数的值
        print(f"Gradient:\n{param.requires_grad}")  # 打印参数的梯度
        print("-" * 50)  # 分隔线，用来分隔每个参数


def main(args):
    loss_func = Loss(args)
    inputs, mask = get_target(args)
    utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
    renderer = load_renderer(args, inputs, mask)

    writer = SummaryWriter("/home/dhu/zyl/CLIPasso/CLIPasso-main/runs")
    save_path = '/home/dhu/zyl/CLIPasso/CLIPasso-main/cos_matrix'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss, best_fc_loss = 100, 100
    best_iter, best_iter_fc = 0, 0
    min_delta = 1e-5
    terminate = False
    # 创建模型
    cnnModel = model_rnngcn.SimpleCNN().to(args.device)
    GCNmodel = model_rnngcn.GCN(input_dim=128, output_dim=4).to(args.device)  # 4 是每条bezier曲线控制点的数量
    model_parameters=list(cnnModel.parameters()) + list(GCNmodel.parameters())
    optimizer = PainterOptimizer(args,model_parameters,renderer)
    renderer.set_random_noise(0)
    img = renderer.init_image(stage=0)
    optimizer.init_optimizers()
    # not using tdqm for jupyter demo
    if args.display:

        epoch_range = range(args.num_iter)
    else:
        epoch_range = tqdm(range(args.num_iter))

    for epoch in epoch_range:
        if not args.display:
            epoch_range.refresh()
        renderer.set_random_noise(epoch)
        if args.lr_scheduler:
            optimizer.update_lr(counter)
        start = time.time()
        optimizer.zero_grad_()
    # 计算s‘
        for i ,path in enumerate(renderer.shapes):
            renderer.control_points_set[i] = path.points
        #绘制原图，获取掩码图片
        img_rare=display_img.diffvgProcess(renderer.control_points_set, 224, 224)
        img_beziers = point_render(renderer.control_points_set)
        img_beziers_clone=img_beziers.clone().permute(0,3,1,2)
        # print(img_beziers.shape)
        img_masked=display_img.mask_img(img_rare,img_beziers)
        img_masked=torch.stack(img_masked,dim=0)

        img_grid = torchvision.utils.make_grid(img_masked, nrow=8, padding=2)
        img_beziers_grid = torchvision.utils.make_grid(img_beziers_clone, nrow=8, padding=2)

        #送入模型
        img_masked = img_masked.permute(1, 0, 2, 3)
        feature = cnnModel(img_masked).view(16,-1)
        new_points=torch.sigmoid(feature).view(16,-1,2)
        # print(f"the rare feature {feature}")
        reg_matrix,cos_matrix=cos_Adjmatrix(feature)
            # print(feature)
        # print(reg_matrix.shape)
        # cos_matrix=reg_matrix.clone()

        # new_points = GCNmodel(new_points, reg_matrix).view(-1, 4, 2)
        # print(feature)
        sketches=display_img.render_sketch(new_points,renderer,epoch).to(args.device)

        # sketches_new=sketches.clone().squeeze(0)
        # writer.add_image(f"{epoch}_sketch", sketches_new)
        losses_dict = loss_func(sketches, inputs.detach(
        ), renderer.get_color_parameters(), renderer, counter, optimizer)
        loss = sum(list(losses_dict.values()))
        loss.backward()
        # if epoch%10==0:
        #     for i, path in enumerate(renderer.shapes):
        #         print(f"the {i}grad are {path.points.grad}")
        # # 打印 cnnModel 和 GCNmodel 的参数
        # print("cnnModel Parameters:")
        # print_model_parameters(cnnModel)
        # print(" Parameters:")

#任务：改变svg绘制，修改并加入mask，定时画出余弦相似度矩阵热力图
        optimizer.step_()
        if epoch % args.save_interval == 0:
            # 打印网格的形状,chw
            display_img.cs_heatmap(cos_matrix, save_path, epoch, "cos_matrix")
            writer.add_image(f'{epoch}images_grid', img_grid)
            writer.add_image(f'{epoch}img_beziers_grid', img_beziers_grid)


            # 保存为 .pt 文件（PyTorch 张量格式）
            torch.save(feature, f"{args.output_dir}/feature.pt")
            utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
                             use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
        if epoch % args.eval_interval== 0:


            # for i ,path in enumerate(renderer.shapes):
            #     print(f"the {i}grad are {path.points.grad}")
            with torch.no_grad():
                # # 绘制余弦相似度热力图

                losses_dict_eval = loss_func(sketches, inputs, renderer.get_color_parameters(
                ), renderer.get_points_parans(), counter, optimizer, mode="eval")
                loss_eval = sum(list(losses_dict_eval.values()))
                configs_to_save["loss_eval"].append(loss_eval.item())
                for k in losses_dict_eval.keys():
                    if k not in configs_to_save.keys():
                        configs_to_save[k] = []
                    configs_to_save[k].append(losses_dict_eval[k].item())
                if args.clip_fc_loss_weight:
                    if losses_dict_eval["fc"].item() < best_fc_loss:
                        best_fc_loss = losses_dict_eval["fc"].item(
                        ) / args.clip_fc_loss_weight
                        best_iter_fc = epoch
                print(
                    f"eval iter[{epoch}/{args.num_iter}] loss[{loss.item()}] time[{time.time() - start}]")

                cur_delta = loss_eval.item() - best_loss
                if abs(cur_delta) > min_delta:
                    if cur_delta < 0:
                        best_loss = loss_eval.item()
                        best_iter = epoch
                        terminate = False
                        utils.plot_batch(
                            inputs, sketches, args.output_dir, counter, use_wandb=args.use_wandb, title="best_iter.jpg")
                        renderer.save_svg(args.output_dir, "best_iter")

                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_loss_fc"] = best_fc_loss
                    wandb_dict = {"delta": cur_delta,
                                  "loss_eval": loss_eval.item()}
                    for k in losses_dict_eval.keys():
                        wandb_dict[k + "_eval"] = losses_dict_eval[k].item()
                    wandb.log(wandb_dict, step=counter)

                # if abs(cur_delta) <= min_delta:
                #     if terminate:
                #         break
                #     terminate = True

        if counter == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                             args.use_wandb, "{}/{}.jpg".format(
                                 args.output_dir, "attention_map"),
                             args.saliency_model, args.display_logs)

        if args.use_wandb:
            wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
            for k in losses_dict.keys():
                wandb_dict[k] = losses_dict[k].item()
            wandb.log(wandb_dict, step=counter)

        counter += 1

    renderer.save_svg(args.output_dir, "final_svg")
    path_svg = os.path.join(args.output_dir, "best_iter.svg")
    utils.log_sketch_summary_final(
        path_svg, args.use_wandb, args.device, best_iter, best_loss, "best total")

    return configs_to_save

if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()


