import array
import pydiffvg
from PIL import Image,ImageOps
from typing import Optional, Tuple, Any
import os, random, pdb
import torch, numpy as np
from matplotlib.transforms import Affine2D
from numpy import ndarray, dtype
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


import struct
import sys
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import functional as F
import torch.nn.functional as F2
class StrokesDataset(Dataset):
    """
    ## Dataset

    This class loads and pre-processes the data.
    """

    def __init__(self, dataset: np.array, max_seq_length: int, scale: Optional[float] = None):
        """
        `dataset` is a list of numpy arrays of shape [seq_len, 3].
        It is a sequence of strokes, and each stroke is represented by
        3 integers.
        First two are the displacements along x and y ($\Delta x$, $\Delta y$)
        and the last integer represents the state of the pen, $1$ if it's touching
        the paper and $0$ otherwise.
        """

        data = []
        # We iterate through each of the sequences and filter
        for seq in dataset:
            # Filter if the length of the sequence of strokes is within our range
            if 10 < len(seq) <= max_seq_length:
                # Clamp $\Delta x$, $\Delta y$ to $[-1000, 1000]$
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                # Convert to a floating point array and add to `data`
                seq = np.array(seq, dtype=np.float32)
                data.append(seq)

        # We then calculate the scaling factor which is the
        # standard deviation of ($\Delta x$, $\Delta y$) combined.
        # Paper notes that the mean is not adjusted for simplicity,
        # since the mean is anyway close to $0$.
        if scale is None:
            scale = np.std(np.concatenate([np.ravel(s[:, 0:2]) for s in data]))
        self.scale = scale

        # Get the longest sequence length among all sequences
        longest_seq_len = max([len(seq) for seq in data])

        # We initialize PyTorch data array with two extra steps for start-of-sequence (sos)
        # and end-of-sequence (eos).
        # Each step is a vector $(\Delta x, \Delta y, p_1, p_2, p_3)$.
        # Only one of $p_1, p_2, p_3$ is $1$ and the others are $0$.
        # They represent *pen down*, *pen up* and *end-of-sequence* in that order.
        # $p_1$ is $1$ if the pen touches the paper in the next step.
        # $p_2$ is $1$ if the pen doesn't touch the paper in the next step.
        # $p_3$ is $1$ if it is the end of the drawing.
        self.data = torch.zeros(len(data), longest_seq_len + 2, 5, dtype=torch.float)
        # The mask array needs only one extra-step since it is for the outputs of the
        # decoder, which takes in `data[:-1]` and predicts next step.
        self.mask = torch.zeros(len(data), longest_seq_len + 1)

        for i, seq in enumerate(data):
            seq = torch.from_numpy(seq)
            len_seq = len(seq)
            # Scale and set $\Delta x, \Delta y$
            self.data[i, 1:len_seq + 1, :2] = seq[:, :2] / scale
            # $p_1$
            self.data[i, 1:len_seq + 1, 2] = 1 - seq[:, 2]
            # $p_2$
            self.data[i, 1:len_seq + 1, 3] = seq[:, 2]
            # $p_3$
            self.data[i, len_seq + 1:, 4] = 1
            # Mask is on until end of sequence
            self.mask[i, :len_seq + 1] = 1

        # Start-of-sequence is $(0, 0, 1, 0, 0)$
        self.data[:, 0, 2] = 1

    def __len__(self):
        """Size of the dataset"""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Get a sample"""
        return self.data[idx], self.mask[idx]

#处理一张图片,不放大
def diffvgProcess(sketch,canvas_width,canvas_height):
    shapes = []
    shape_groups = []
    #stroke [4,2]
    for i, stroke in enumerate(sketch):
        num = stroke.shape[0]
        # 控制点数
        num_control_points = torch.tensor([2], dtype=torch.int32)
        # 控制点
        points = stroke.contiguous()

        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points,
                             is_closed=False,
                             stroke_width=torch.tensor(1))
        shapes.append(path)

        # Create shape group 形状单元
        shape_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i], dtype=torch.int32),
            fill_color=None,
            stroke_color=torch.tensor([0, 0, 0, 1.0], dtype=torch.float32)
        )

        shape_groups.append(shape_group)
        # print(f"shape_groups[]: {shape_groups}")
    # serialize_scene 场景描述参数
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    # # 背景图像
    background_image = torch.ones(canvas_height, canvas_width, 4)
    background_image[:, :, 0:3] = 1.0  # 设置RGB通道为1，表示白色
    background_image[:, :, 3] = 1.0  # 设置A通道为1，表示完全不透明background_image,

    # Render 先定义函数再执行渲染
    render = pydiffvg.RenderFunction.apply
    # 多重采样率？
    # print(f"scene_args[]: {scene_args}")
    img = render(canvas_width, canvas_height, 2, 2, 0, background_image, *scene_args)

    # save
    img = img[:, :, :3]  # Drop alpha channel
    # img2 = img.detach().cpu().numpy()
    # # 使用 matplotlib 绘制图像
    # plt.imshow(img2)
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    # plt.savefig("img2")
    gray_img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    gray_img = gray_img.unsqueeze(2)

    return gray_img

#绘制一条bezier曲线，4个控制点，三阶
def render(control_points,canvas_width,canvas_height):
    # list_control_points=control_points #tensor
    num_points=2
    list_control_points=control_points.contiguous()
    path = pydiffvg.Path(
        num_control_points=torch.tensor([num_points]),
        points=list_control_points,
        stroke_width=torch.tensor(1),
        is_closed=False
    )
    shapes=[path]
    # Create shape group
    shape_group= pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0], dtype=torch.int32),
        fill_color=None
        , stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32), #黑色
        #  fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0], dtype=torch.float32)
    )
    shape_groups=[shape_group]
    # Serialize the scene
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    # # 背景图像
    background_image = torch.ones(canvas_height, canvas_width, 4)
    background_image[:, :, 0:3] = 1.0  # 设置RGB通道为1，表示白色
    background_image[:, :, 3] = 1.0  # 设置A通道为1，表示完全不透明background_image,

    # Render the image
    render = pydiffvg.RenderFunction.apply

    img = render(canvas_width, canvas_height, 2, 2, 0, background_image, *scene_args)

    gray_img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    gray_img = gray_img.unsqueeze(2)

    return gray_img
#绘制整张图片
def render_sketch(points, renderer,epoch):
    save_path = '/home/dhu/zyl/CLIPasso/CLIPasso-main/render_img'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_dir = os.path.join(save_path, f"{epoch}_render_img.png")
    points = points * 224
    # points batchsize,num,2
    paths = []
    shape_groups = []
    stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
    _render = pydiffvg.RenderFunction.apply
    # print(points.shape)
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

    img = _render(renderer.canvas_width,  # width
                  renderer.canvas_height,  # height
                  2,  # num_samples_x
                  2,  # num_samples_y
                  0,  # seed
                  background_image,
                  *scene_args)

    # # img = self.render_warp()
    opacity = img[:, :, 3:4]
    img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=renderer.device) * (1 - opacity)
    img = img[:, :, :3]
    img2=img.clone()
    # Convert img from HWC to NCHW
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2).to(renderer.device)  # NHWC -> NCHW
    if epoch%10==0:
        # print(img2.shape)
        img2 = img2.detach().cpu().numpy()
        # 使用 matplotlib 绘制图像
        plt.imshow(img2)
        plt.axis('off')  # 关闭坐标轴
        plt.show()
        plt.savefig(save_dir)
        plt.clf()
        # print(points)

    return img
def mask_img(img_rare,img_beziers):
    #img_rare : hwc;img_beziers:batchsize, chw    粗细要保证一致
    #可能更改大小药用到renderer
    img_masked=[]
    for i in range(img_beziers.shape[0]):
        img_mask=img_beziers[i]
        # 3. 二值化图片，小于阈值（0）的像素会变成 255（白色）。
        # #二值化mask，黑色0,255白色
        binary_mask = (img_mask > 0).float()  # 将值大于0的部分设置为1，小于等于0的部分设置为0
        binary_mask = binary_mask * 255  # 将 1 的部分设为 255，0 的部分设为 0(黑色）
        # 反转 mask，用于找到黑色（值为0）的边界
        reverse_mask = 255 - binary_mask  # 反转掩膜，此刻黑色是255
        mask_nonzero = reverse_mask.view(-1)  # 将 mask 展平成一维
        indices = mask_nonzero.nonzero().squeeze()  # 获取非零元素的索引
        if indices.numel() > 0:
            rows = indices // binary_mask.shape[1]  # 每个非零元素的行索引
            cols = indices % binary_mask.shape[1]  # 每个非零元素的列索引

            y_min = rows.min()
            y_max = rows.max()
            x_min = cols.min()
            x_max = cols.max()
            # 生成 mask，只保留外接矩形部分
            mask = torch.zeros_like(reverse_mask)
            mask[y_min:y_max+1, x_min:x_max+1] = 255
            image = img_rare * mask
            # img_test = image.clone()
            # img_test = img_test.detach().cpu().numpy()
            # plt.imshow(img_test, cmap='gray')  # 使用灰度色图显示
            # # plt.axis('off')  # 关闭坐标轴显示
            # plt.show()
            # plt.savefig(f"sketch_rare_{i}.png")
            # plt.clf()
            #image:224,224,1
            cropped_image = image[y_min:y_max+1, x_min:x_max+1, :]
            cropped_image = cropped_image.permute(2, 0, 1)

            # chw
            # padding填充，resize为正方形
            cropped_image = padding_resize(cropped_image, x_min, x_max, y_max, y_min)

        else:
            mask = torch.zeros_like(reverse_mask)
            cropped_image = img_rare * mask
            cropped_image = cropped_image.permute(2, 0, 1)


        #出来是chw
        cropped_image_resized= F.resize(cropped_image, (64, 64))


        # # 转换大小为 (h,w,c)
        # cropped_image_resized = F.resize(cropped_mask, (64, 64))

        image_final = cropped_image_resized / 255.0  # 如果原图是 0-255 的像素值
        img_masked.append(image_final)


    return img_masked

def padding_resize(mask,x_min,x_max,y_max,y_min):
    # print(f"the mask shape is {mask.shape}")
    #mask:height，width
    height, width = mask.shape[1], mask.shape[2]
    # print(f"the mask height is {height}")
    # print(f"the mask width is {width}")
    max_size = max(height, width)
    # 计算需要填充的像素数
    if height < max_size:
        padding = (0, 0, (max_size - height) // 2, (max_size - height) // 2)
        # print(f"the mask padding is {padding}")
    elif width < max_size:
        # 如果宽度小于高度，则在左右填充
        padding = ((max_size - width) // 2, (max_size - width) // 2, 0, 0)
        # print(f"the mask width is {padding}")
        # print("width")
    else:
        # print('same')
        return mask
    # 使用 F.pad 进行填充
    padded_mask = F2.pad(mask, padding, value=255)  # 填充颜色为白色
    # 输出新的形状
    # print("Padded mask shape:", padded_mask.shape)
    return padded_mask

def cs_heatmap(reg_matrix,folder_path,epoch,name):
    # reg_matrix:matrix
    save_dir = os.path.join(folder_path, f"{epoch}_{name}.png")
    # 2. 将余弦相似度矩阵转换为 numpy 数组
    cos_sim_matrix_np = reg_matrix.clone().detach().cpu().numpy()
    # 在每个单元格中添加文本标注
    for i in range(cos_sim_matrix_np.shape[0]):
        for j in range(cos_sim_matrix_np.shape[1]):
            if cos_sim_matrix_np[i, j] > 0:
                # 在矩阵中添加对应的像素值
                plt.text(j, i, f'{cos_sim_matrix_np[i, j]:.2f}', ha='center', va='center', color='black', fontsize=10)
    # 使用 imshow 绘制热力图
    plt.imshow(cos_sim_matrix_np, cmap='autumn', interpolation='nearest')
    # 显示颜色对应条
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.savefig(save_dir)
    plt.clf()






if __name__ == '__main__':
    # `npz` file path is `data/sketch/[DATASET NAME].npz`
    path = './cat.npz'
    # Load the numpy file
    dataset = np.load(str(path), encoding='latin1', allow_pickle=True)
    # print(dataset['train'].shape)
    max_seq_length = 200
    # Create training dataset
    train_dataset = StrokesDataset(dataset['train'], max_seq_length)
    # 使用采样器生成数据
    data, *_ = train_dataset[4:10]
    data = data.transpose(0, 1)



    for i in range(data.shape[1]):
        news=data[:,i,:].clone()
        # # print(news)
        news[:, 0:2]= torch.cumsum(news[:, 0:2], dim=0)

        news[:, 2] = news[:, 3]
        news = news[:, 0:3].detach().cpu().numpy()
        strokes = np.split(news, np.where(news[:, 2] > 0)[0] + 1)

        for s in strokes:

            plt.plot(s[:, 0], -s[:, 1])

        # Show the plot
        plt.savefig(f"sketch_rare_p{i}.png")
        plt.clf()

        datas = rare_process(data[:,i,:].clone())
        strokes = torch.cat(datas, dim=0)
        print(strokes.shape)
        strokes=strokes[:,:2]
        # print(datas)
        img = diffvgProcess(strokes, 32, 32,1)
        img_rare = img.detach().cpu().numpy()
        img_rare = (img_rare * 255).astype(np.uint8)
        img_rare = Image.fromarray(img_rare)

        img_rare.save(f"sketch_rare_d{i}.png")


