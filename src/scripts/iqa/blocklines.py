import numpy as np
import torch
import torch.fft

from src.scripts.utils.objects import IQANode

DEFAULT_BLOCK_SIZE = 8


try:
    """
    Author: Ziyang Hu
    Author's repo: https://github.com/zh217/torch-dct
    Licence: MIT
    """
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))

    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

except ImportError:

    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)


def dct(x, norm=None):
    """
    Author: Ziyang Hu
    Author's repo: https://github.com/zh217/torch-dct
    Licence: MIT

    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    width_r = torch.cos(k)
    width_i = torch.sin(k)

    V = Vc[:, :, 0] * width_r - Vc[:, :, 1] * width_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def dct_2d(x, norm=None):
    """
    Author: Ziyang Hu
    Author's repo: https://github.com/zh217/torch-dct
    Licence: MIT

    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def calc_margin(
    height: int,
    width: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[int, int, int, int]:
    """
    Calculate margins for DCT processing.
    """
    height_margin: int = height % block_size
    width_margin: int = width % block_size
    cal_height: int = height - (
        height_margin if height_margin >= 4 else height_margin + block_size
    )
    cal_width: int = width - (
        width_margin if width_margin >= 4 else width_margin + block_size
    )
    height_margin = (height_margin + block_size) if height_margin < 4 else height_margin
    width_margin = (width_margin + block_size) if width_margin < 4 else width_margin
    return cal_height, cal_width, height_margin, width_margin


def calc_v_torch(
    dct_img: torch.Tensor,
    height_block_num: int,
    width_block_num: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Calculate averaged V values from a batched DCT image.

    This function computes the V value for each block in the DCT image by extracting the
    central pixel and its four neighbors (left, right, top, bottom) from each block.
    The V value for each pixel is defined as:
      V = sqrt((b_val + c_val - 2 * a)² + (d_val + e_val - 2 * a)²)
    where:
      - a is the central pixel value,
      - b_val and c_val are the left and right neighbor values,
      - d_val and e_val are the top and bottom neighbor values.
    The computed V values are then averaged over all blocks for each image in the batch.

    The input tensor should have shape (B, H, W) or (B, 1, H, W), where B is the batch size.
    The block indices are computed using offsets derived from the block size.

    Args:
        dct_img: A tensor representing the DCT image with shape (B, H, W) or (B, 1, H, W).
        height_block_num: Number of blocks along the height of the image.
        width_block_num: Number of blocks along the width of the image.
        block_size: Size of each block. Defaults to DEFAULT_BLOCK_SIZE.

    Returns:
        A tensor of shape (B,) containing the averaged V values for each image in the batch.

    Raises:
        TypeError: If dct_img is not a torch.Tensor.
        ValueError: If dct_img has an unsupported number of dimensions, or if block dimensions are invalid.
    """
    if not isinstance(dct_img, torch.Tensor):
        raise TypeError(f"dct_img must be a torch.Tensor, got {type(dct_img)}")

    if dct_img.dim() not in (3, 4):
        raise ValueError(
            f"dct_img must have 3 or 4 dimensions (got {dct_img.dim()}). Expected (B, H, W) or (B, 1, H, W)."
        )

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if height_block_num <= 2 or width_block_num <= 2:
        raise ValueError(
            f"height_block_num and width_block_num must be greater than 2 (got {height_block_num} and {width_block_num})."
        )

    if dct_img.dim() == 4:
        dct_img = dct_img.squeeze(1)

    batch_size = dct_img.size(0)

    # compute number of offsets (blocks used for indexing)
    num_h = height_block_num - 3
    num_w = width_block_num - 3

    device = dct_img.device

    # compute starting offsets for each block
    # Offsets: block_size + (index * block_size) for index 1 to height_block_num-3 (inclusive)
    h_offsets = (
        block_size + torch.arange(1, height_block_num - 2, device=device) * block_size
    )  # shape: (num_h,)

    w_offsets = (
        block_size + torch.arange(1, width_block_num - 2, device=device) * block_size
    )  # shape: (num_w,)

    # build the row and column indices for blocks
    # r: shape (num_h, 1, block_size, 1), c: shape (1, num_w, 1, block_size)
    r = h_offsets.view(num_h, 1, 1, 1) + torch.arange(block_size, device=device).view(
        1, 1, block_size, 1
    )
    c = w_offsets.view(1, num_w, 1, 1) + torch.arange(block_size, device=device).view(
        1, 1, 1, block_size
    )

    # expand indices to full grid shape (num_h, num_w, block_size, block_size) and cast to int
    r = r.expand(num_h, num_w, block_size, block_size).to(torch.int)
    c = c.expand(num_h, num_w, block_size, block_size).to(torch.int)

    # create a batch index tensor of shape (B, 1, 1, 1, 1) for advanced indexing
    batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1, 1, 1)

    # expand r and c to include the batch dimension: shape becomes (B, num_h, num_w, block_size, block_size)
    r_exp = r.unsqueeze(0).expand(batch_size, num_h, num_w, block_size, block_size)
    c_exp = c.unsqueeze(0).expand(batch_size, num_h, num_w, block_size, block_size)

    # Extract the central value (a) and its four neighbors:
    # left (b_val), right (c_val), top (d_val), and bottom (e_val)
    a = dct_img[batch_idx, r_exp, c_exp]
    b_val = dct_img[batch_idx, r_exp, c_exp - block_size]
    c_val = dct_img[batch_idx, r_exp, c_exp + block_size]
    d_val = dct_img[batch_idx, r_exp - block_size, c_exp]
    e_val = dct_img[batch_idx, r_exp + block_size, c_exp]

    # compute V for each block and pixel within the block
    V = torch.sqrt((b_val + c_val - 2 * a) ** 2 + (d_val + e_val - 2 * a) ** 2)

    # average V over all blocks for each image
    normalization = (height_block_num - 2) * (width_block_num - 2)
    V_average = V.sum(dim=(1, 2)) / normalization

    return V_average


def blockwise_dct(
    gray_imgs: torch.Tensor,
    height_block_num: int,
    width_block_num: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Compute the DCT of an image block-wise using batched processing.

    This function divides the image into non-overlapping blocks of size block_size x block_size,
    applies a batched DCT transform (using dct2, which accepts batched input), and reconstructs the
    DCT image from the transformed blocks.

    Args:
        gray_imgs: Input image tensor with shape (H, W).
        dct: A DCT object with a method dct2 that accepts batched input.
        height_block_num: Number of blocks along the height.
        width_block_num: Number of blocks along the width.

    Returns:
        A tensor containing the DCT coefficients of the image blocks, arranged in the original block layout.
    """
    assert gray_imgs.dim() == 4, "Input tensor must have shape (B, 1, H, W)."
    batch_size, channel_size, *_ = gray_imgs.shape
    assert channel_size == 1, "Input tensor must have shape (B, 1, H, W)."

    if (
        gray_imgs.shape[-2] < height_block_num * block_size
        or gray_imgs.shape[-1] < width_block_num * block_size
    ):
        raise ValueError(
            f"Invalid image dimensions. Image must be at least {height_block_num * block_size} x {width_block_num * block_size} because of number of blocks and block size."
        )

    # divide the image into blocks of shape (height_block_num, width_block_num, block_size, block_size).
    blocks = gray_imgs.unfold(
        -2,
        block_size,
        block_size,
    ).unfold(
        -1,
        block_size,
        block_size,
    )
    blocks = blocks.contiguous().view(batch_size, -1, block_size, block_size)

    # apply the batched DCT transform to all blocks at once.
    dct_blocks_flat: torch.Tensor = dct_2d(blocks, norm="ortho")

    dct_blocks = dct_blocks_flat.view(
        batch_size,
        height_block_num,
        width_block_num,
        block_size,
        block_size,
    )
    dct_blocks = dct_blocks.permute(0, 1, 3, 2, 4).contiguous()
    dct_blocks = dct_blocks.view(
        batch_size,
        height_block_num * block_size,
        width_block_num * block_size,
    )
    return dct_blocks


def calculate_image_blockiness(
    gray_images: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
):
    """Calculate the blockiness metric for a batch of grayscale images.

    The blockiness is computed by comparing the blockwise DCT of the original image and an
    offset version of the image. The offset image is created by shifting the original image by
    4 pixels in both spatial dimensions.

    Args:
        gray_images: A 4D tensor with shape (B, 1, H, W) representing a batch of grayscale images.
        block_size: The size of each block used for the DCT computation. Defaults to DEFAULT_BLOCK_SIZE.

    Returns:
        A tensor containing the summed blockiness scores per image.

    Raises:
        TypeError: If gray_images is not a torch.Tensor.
        ValueError: If gray_images does not have 4 dimensions or if its spatial dimensions
                    are too small for offset calculation.
        RuntimeError: If any of the intermediate calculations fail.
    """
    if not isinstance(gray_images, torch.Tensor):
        raise TypeError("gray_images must be a torch.Tensor.")
    if gray_images.dim() != 4:
        raise ValueError("Input tensor must have shape (B, 1, H, W).")

    # Extract height and width from the spatial dimensions.
    height, width = gray_images.shape[-2:]
    if height < 4 or width < 4:
        raise ValueError(
            "Image height and width must be at least 4 pixels for offset calculation."
        )

    # Calculate the valid height and width margins.
    cal_height, cal_width, _, _ = calc_margin(height=height, width=width)

    height_block_num, width_block_num = (
        cal_height // block_size,
        cal_width // block_size,
    )

    gray_tensor_cut = gray_images[..., :cal_height, :cal_width]
    gray_offset = torch.zeros_like(gray_images)
    gray_offset[..., :-4, :-4] = gray_images[..., 4:, 4:]
    gray_offset = gray_offset[..., :cal_height, :cal_width]

    dct_imgs = blockwise_dct(
        gray_imgs=gray_tensor_cut,
        height_block_num=height_block_num,
        width_block_num=width_block_num,
        block_size=block_size,
    )

    dct_offset_imgs = blockwise_dct(
        gray_imgs=gray_offset,
        height_block_num=height_block_num,
        width_block_num=width_block_num,
        block_size=block_size,
    )
    v_average = calc_v_torch(
        dct_img=dct_imgs,
        height_block_num=height_block_num,
        width_block_num=width_block_num,
        block_size=block_size,
    )
    v_offset_average = calc_v_torch(
        dct_img=dct_offset_imgs,
        height_block_num=height_block_num,
        width_block_num=width_block_num,
        block_size=block_size,
    )
    epsilon = 1e-8
    d = torch.abs(v_offset_average - v_average) / (v_average + epsilon)
    d_sum = torch.sum(d, dim=(1, 2))
    return d_sum


def rgb_to_grayscale(tensor):
    # define luminance coefficients
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor.device).view(
        1, 3, 1, 1
    )

    # apply weighted sum across the channel dimension
    # (B, 1, H, W)
    grayscale = (tensor * weights).sum(dim=1, keepdim=True)
    return grayscale


class BlockinessThread(IQANode):
    def __init__(
        self,
        img_dir,
        batch_size: int = 8,
        thread: float = 0.5,
        median_thread=0,
        move_folder: str | None = None,
    ):
        super().__init__(
            img_dir, batch_size, thread, median_thread, move_folder, None, reverse=True
        )
        self.model = calculate_image_blockiness

    def forward(self, images):
        images = rgb_to_grayscale(images)
        return self.model(images)
