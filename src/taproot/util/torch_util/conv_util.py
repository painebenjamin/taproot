from typing import Tuple, Union

__all__ = [
    "get_conv_2d_output_size",
]

def get_conv_2d_output_size(
    input_size: Union[int, Tuple[int, int]],
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
) -> Tuple[int, int]:
    """
    Calculate the output size of a 2D convolution layer.
    :param input_size: Input size of the 2D convolution layer.
    :param kernel_size: Kernel size of the 2D convolution layer.
    :param stride: Stride of the 2D convolution layer.
    :param padding: Padding of the 2D convolution layer.
    :param dilation: Dilation of the 2D convolution layer.
    :return: Output size of the 2D convolution layer.
    """
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    output_size = (
        (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1,
        (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1,
    )

    return output_size
