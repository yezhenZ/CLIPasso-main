import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydiffvg
import skimage
import skimage.io
import torch
import wandb
import PIL
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from skimage.transform import resize

from U2Net_.model import U2NET


def save_cosine_similarity_heatmap(reg_matrix, folder_path, epoch, name):
    """绘制余弦相似度矩阵的热力图

    Args:
        reg_matrix: 余弦相似度矩阵 tensor
        folder_path: 保存路径
        epoch: 当前训练轮数
        name: 文件名
    """
    # 设置图像大小和DPI以提高清晰度
    plt.figure(figsize=(10, 8), dpi=100)

    # 转换为numpy数组
    cos_sim_matrix_np = reg_matrix.detach().cpu().numpy()

    # 绘制热力图
    im = plt.imshow(cos_sim_matrix_np, cmap='YlOrRd', interpolation='nearest')

    # 添加数值标注,只标注非零值
    for i in range(cos_sim_matrix_np.shape[0]):
        for j in range(cos_sim_matrix_np.shape[1]):
            if cos_sim_matrix_np[i, j] > 0:
                plt.text(j, i, f'{cos_sim_matrix_np[i, j]:.2f}',
                         ha='center', va='center',
                         color='black' if cos_sim_matrix_np[i, j] < 0.7 else 'white',
                         fontsize=8)

    # 添加颜色条和标题
    plt.colorbar(im)
    plt.title(f'Cosine Similarity Matrix - Epoch {epoch}')

    # 调整布局并保存
    plt.tight_layout()
    save_dir = os.path.join(folder_path, f"{epoch}_{name}.png")
    plt.savefig(save_dir, bbox_inches='tight')
    plt.clf()
    plt.close()


# 绘制整张图片
def render_img_rgb_from_renderer(points, renderer,epoch,save_path):
    """
    使用renderer渲染完整的草图

    参数:
        points: 所有曲线的控制点,形状为(batch_size,num_points,2)
        renderer: 渲染器对象,包含画布大小等参数

    返回:
        img: 渲染后的RGB图像,形状为(N,C,H,W)
    """
    points = points * 224

    paths = []
    shape_groups = []

    stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])

    for i, control_points in enumerate(points):
        path = pydiffvg.Path(num_control_points=renderer.num_control_points,
                             points=control_points,
                             stroke_width=torch.tensor(renderer.width),
                             is_closed=False)
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([i]),
                                         fill_color=None,
                                         stroke_color=stroke_color)
        paths.append(path)
        shape_groups.append(path_group)

    scene_args = pydiffvg.RenderFunction.serialize_scene( \
        renderer.canvas_width, renderer.canvas_height, paths, shape_groups)

    # 背景图像
    background_image = torch.ones(renderer.canvas_height, renderer.canvas_width, 4)
    background_image[:, :, 0:3] = 1.0  # 设置RGB通道为1，表示白色
    background_image[:, :, 3] = 1.0  # 设置A通道为1，表示完全不透明background_image,
    render = pydiffvg.RenderFunction.apply
    img = render(renderer.canvas_width,  # width
                 renderer.canvas_height,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 0,  # seed
                 background_image,
                 *scene_args)

    opacity = img[:, :, 3:4]
    img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=renderer.device) * (1 - opacity)
    img = img[:, :, :3]
    if epoch % 10 == 0:
        save_dir = os.path.join(save_path, f"{epoch}_out.png")
        # print(img2.shape)
        img2 = img.clone().detach().cpu().numpy()
        # 使用 matplotlib 绘制图像
        plt.imshow(img2)
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(save_dir)
        plt.clf()
        plt.close()
    # Convert img from HWC to NCHW
    img = img.unsqueeze(0)

    img = img.permute(0, 3, 1, 2).to(renderer.device)  # NHWC -> NCHW




    return img
def imwrite(img, filename, gamma=2.2, normalize=False, use_wandb=False, wandb_name="", step=0, input_im=None):
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(img, np.ndarray):
        img = img.data.numpy()
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        # repeat along the third dimension
        img = np.expand_dims(img, 2)
    img[:, :, :3] = np.power(img[:, :, :3], 1.0/gamma)
    img = (img * 255).astype(np.uint8)

    skimage.io.imsave(filename, img, check_contrast=False)
    images = [wandb.Image(Image.fromarray(img), caption="output")]
    if input_im is not None and step == 0:
        images.append(wandb.Image(input_im, caption="input"))
    if use_wandb:
        wandb.log({wandb_name + "_": images}, step=step)


def plot_batch(inputs, outputs, output_dir, step, use_wandb, title):
    plt.figure()
    plt.subplot(2, 1, 1)
    grid = make_grid(inputs.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("inputs")

    plt.subplot(2, 1, 2)
    grid = make_grid(outputs, normalize=False, pad_value=2)
    npgrid = grid.detach().cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("outputs")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"output": wandb.Image(plt)}, step=step)
    plt.savefig("{}/{}".format(output_dir, title))
    plt.close()


def log_input(use_wandb, epoch, inputs, output_dir):
    grid = make_grid(inputs.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.tight_layout()
    if use_wandb:
        wandb.log({"input": wandb.Image(plt)}, step=epoch)
    plt.close()
    input_ = inputs[0].cpu().clone().detach().permute(1, 2, 0).numpy()
    input_ = (input_ - input_.min()) / (input_.max() - input_.min())
    input_ = (input_ * 255).astype(np.uint8)
    imageio.imwrite("{}/{}.png".format(output_dir, "input"), input_)


def log_sketch_summary_final(path_svg, use_wandb, device, epoch, loss, title):
    canvas_width, canvas_height, shapes, shape_groups = load_svg(path_svg)
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,   # num_samples_x
                  2,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)

    img = img[:, :, 3:4] * img[:, :, :3] + \
        torch.ones(img.shape[0], img.shape[1], 3,
                   device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    plt.imshow(img.cpu().numpy())
    plt.axis("off")
    plt.title(f"{title} best res [{epoch}] [{loss}.]")
    if use_wandb:
        wandb.log({title: wandb.Image(plt)})
    plt.close()


def log_sketch_summary(sketch, title, use_wandb):
    plt.figure()
    grid = make_grid(sketch.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    if use_wandb:
        wandb.run.summary["best_loss_im"] = wandb.Image(plt)
    plt.close()


def load_svg(path_svg):
    svg = os.path.join(path_svg)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        svg)
    return canvas_width, canvas_height, shapes, shape_groups


def read_svg(path_svg, device, multiply=False):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        path_svg)
    if multiply:
        canvas_width *= 2
        canvas_height *= 2
        for path in shapes:
            path.points *= 2
            path.stroke_width *= 2
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,   # num_samples_x
                  2,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + \
        torch.ones(img.shape[0], img.shape[1], 3,
                   device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    return img


def plot_attn_dino(attn, threshold_map, inputs, inds, use_wandb, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(2, attn.shape[0] + 2, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, 2)
    plt.imshow(attn.sum(0).numpy(), interpolation='nearest')
    plt.title("atn map sum")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 3)
    plt.imshow(threshold_map[-1].numpy(), interpolation='nearest')
    plt.title("prob sum")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 4)
    plt.imshow(threshold_map[:-1].sum(0).numpy(), interpolation='nearest')
    plt.title("thresh sum")
    plt.axis("off")

    for i in range(attn.shape[0]):
        plt.subplot(2, attn.shape[0] + 2, i + 3)
        plt.imshow(attn[i].numpy())
        plt.axis("off")
        plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 1 + i + 4)
        plt.imshow(threshold_map[i].numpy())
        plt.axis("off")
    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()


def plot_attn_clip(attn, threshold_map, inputs, inds, use_wandb, output_path, display_logs):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("atn map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    threshold_map_ = (threshold_map - threshold_map.min()) / \
        (threshold_map.max() - threshold_map.min())
    plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
    plt.title("prob softmax")
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()


def plot_atten(attn, threshold_map, inputs, inds, use_wandb, output_path, saliency_model, display_logs):
    if saliency_model == "dino":
        plot_attn_dino(attn, threshold_map, inputs,
                       inds, use_wandb, output_path)
    elif saliency_model == "clip":
        plot_attn_clip(attn, threshold_map, inputs, inds,
                       use_wandb, output_path, display_logs)


def fix_image_scale(im):
    im_np = np.array(im) / 255
    height, width = im_np.shape[0], im_np.shape[1]
    max_len = max(height, width) + 20
    new_background = np.ones((max_len, max_len, 3))
    y, x = max_len // 2 - height // 2, max_len // 2 - width // 2
    new_background[y: y + height, x: x + width] = im_np
    new_background = (new_background / new_background.max()
                      * 255).astype(np.uint8)
    new_im = Image.fromarray(new_background)
    return new_im


def get_mask_u2net(args, pil_im):
    w, h = pil_im.size[0], pil_im.size[1]
    im_size = min(w, h)
    data_transforms = transforms.Compose([
        transforms.Resize(min(320, im_size), interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(
            0.26862954, 0.26130258, 0.27577711)),
    ])

    input_im_trans = data_transforms(pil_im).unsqueeze(0).to(args.device)

    model_dir = os.path.join("./U2Net_/saved_models/u2net.pth")
    net = U2NET(3, 1)
    if torch.cuda.is_available() and args.use_gpu:
        net.load_state_dict(torch.load(model_dir))
        net.to(args.device)
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(input_im_trans.detach())
    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    predict = pred
    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1
    mask = torch.cat([predict, predict, predict], axis=0).permute(1, 2, 0)
    mask = mask.cpu().numpy()
    mask = resize(mask, (h, w), anti_aliasing=False)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    
    # predict_np = predict.clone().cpu().data.numpy()
    im = Image.fromarray((mask[:, :, 0]*255).astype(np.uint8)).convert('RGB')
    im.save(f"{args.output_dir}/mask.png")

    im_np = np.array(pil_im)
    im_np = im_np / im_np.max()
    im_np = mask * im_np
    im_np[mask == 0] = 1
    im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    im_final = Image.fromarray(im_final)

    return im_final, predict