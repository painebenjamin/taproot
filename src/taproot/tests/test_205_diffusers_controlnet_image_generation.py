from typing import Any, Tuple
from taproot import Task
from taproot.util import (
    get_test_image,
    save_test_image,
    make_grid,
)

def test_sd15_controlnet_image_generation() -> None:
    """
    Exercises all controlnets for SD1.5.
    """
    from PIL.Image import Image
    kwargs = {
        "prompt": "A woman in office interior with an art deco style",
        "scheduler": "k_dpm_2_discrete_karras",
        "num_inference_steps": 28,
        "seed": 12345,
        "highres_fix_strength": 0.4,
        "guidance_scale": 6.5,
        "height": 768,
    }
    sd = Task.get("image-generation", "stable-diffusion-v1-5-epicrealism-v5")
    assert sd is not None, "Task not found"
    pipe = sd()
    pipe.load()
    base_image = pipe(**kwargs)
    save_test_image(base_image, "sd15_base")

    kwargs["prompt"] = "A woman in an office interior with a modern, high-tech style"

    def run_test(
        task_name: str,
        model_name: str,
        controlnet_name: str,
        **detect_kwargs: Any
    ) -> Tuple[Image, Image, Image]:
        """
        :param task_name: The name of the task to run.
        :param model_name: The name of the model to use.
        :param controlnet_name: The name of the controlnet to use.
        :param detect_kwargs: Keyword arguments for the detection task.
        """
        detect_task = Task.get(task_name, model_name)
        assert detect_task is not None, "Task not found"
        detect = detect_task()
        detect.load()
        detect_image = detect(image=base_image, **detect_kwargs)
        detect.unload()
        save_test_image(detect_image, f"sd15_{controlnet_name}_detect")
        detect_result = pipe(
            control_image={controlnet_name: detect_image},
            **kwargs
        )
        save_test_image(
            detect_result,
            f"sd15_{controlnet_name}_result"
        )
        detect_i2i_result = pipe(
            control_image={controlnet_name: detect_image},
            image=base_image,
            strength=0.8,
            **kwargs
        )
        save_test_image(
            detect_i2i_result,
            f"sd15_{controlnet_name}_i2i_result"
        )
        return detect_image, detect_result, detect_i2i_result

    canny = run_test("edge-detection", "canny", "canny")
    hed = run_test("edge-detection", "hed", "hed")
    scribble = run_test("edge-detection", "hed", "scribble", scribble=True)
    pidi = run_test("edge-detection", "pidi", "softedge")
    depth = run_test("depth-detection", "midas", "depth")
    normal = run_test("depth-detection", "midas", "normal", mode="normal")
    lineart = run_test("line-detection", "informative-drawings", "lineart")
    anime = run_test("line-detection", "informative-drawings-anime", "anime")
    mlsd = run_test("line-detection", "mlsd", "mlsd")
    pose = run_test("pose-detection", "openpose", "pose")

    # QR Code is a bit of an outlier, we handle it manually
    kwargs["prompt"] = "a modern house facade"
    kwargs["height"] = 512
    kwargs["control_scale"] = 1.5
    kwargs["num_inference_steps"] = 50
    kwargs["scheduler"] = "dpmsolver_sde_multistep_karras"
    kwargs.pop("highres_fix_strength")
    qr_source = get_test_image(subject="qrcode", size="512x512")
    qr_result = pipe(
        control_image={"qr": qr_source},
        **kwargs
    )
    save_test_image(qr_result, "sd15_qr_result")
    qr_i2i_result = pipe(
        control_image={"qr": qr_source},
        image=qr_result,
        strength=0.8,
        **kwargs
    )
    save_test_image(qr_i2i_result, "sd15_qr_i2i_result")

    images = [
        (base_image, "base"),
        (canny[0], "canny detect"),
        (canny[1], "canny t2i result"),
        (canny[2], "canny i2i result"),
        (base_image, "base"),
        (hed[0], "hed detect"),
        (hed[1], "hed t2i result"),
        (hed[2], "hed i2i result"),
        (base_image, "base"),
        (scribble[0], "scribble detect"),
        (scribble[1], "scribble t2i result"),
        (scribble[2], "scribble i2i result"),
        (base_image, "base"),
        (pidi[0], "pidi detect"),
        (pidi[1], "pidi t2i result"),
        (pidi[2], "pidi i2i result"),
        (base_image, "base"),
        (depth[0], "depth detect"),
        (depth[1], "depth t2i result"),
        (depth[2], "depth i2i result"),
        (base_image, "base"),
        (normal[0], "normal detect"),
        (normal[1], "normal t2i result"),
        (normal[2], "normal i2i result"),
        (base_image, "base"),
        (lineart[0], "lineart detect"),
        (lineart[1], "lineart t2i result"),
        (lineart[2], "lineart i2i result"),
        (base_image, "base"),
        (anime[0], "anime detect"),
        (anime[1], "anime t2i result"),
        (anime[2], "anime i2i result"),
        (base_image, "base"),
        (mlsd[0], "mlsd detect"),
        (mlsd[1], "mlsd t2i result"),
        (mlsd[2], "mlsd i2i result"),
        (base_image, "base"),
        (pose[0], "pose detect"),
        (pose[1], "pose t2i result"),
        (pose[2], "pose i2i result"),
        (qr_source, "qr"),
        (qr_source, "qr"),
        (qr_result, "qr t2i result"),
        (qr_i2i_result, "qr i2i result"),
    ]

    grid = make_grid(
        images,
        image_size=(512, 768),
        num_columns=4,
        font_size=14,
    )

    save_test_image(grid, "sd15_controlnet_grid") # type: ignore[arg-type]

def test_sdxl_controlnet_image_generation() -> None:
    """
    Exercises all controlnets for SDXL.
    """
    from PIL.Image import Image
    kwargs = {
        "prompt": "A woman in office interior with an art deco style",
        "scheduler": "k_dpm_2_discrete_karras",
        "num_inference_steps": 30,
        "seed": 12345678,
        "highres_fix_strength": 0.4,
        "guidance_scale": 3.5,
        "height": 1344,
        "width": 768,
    }
    sdxl = Task.get("image-generation", "stable-diffusion-xl-stoiqo-newreality-pro")
    assert sdxl is not None, "Task not found"
    pipe = sdxl()
    pipe.load()
    base_image = pipe(**kwargs)
    save_test_image(base_image, "sdxl_base")

    kwargs["prompt"] = "A woman in an office interior with a modern, high-tech style"

    def run_test(
        task_name: str,
        model_name: str,
        controlnet_name: str,
        control_scale: float = 1.0,
        **detect_kwargs: Any
    ) -> Tuple[Image, Image, Image]:
        """
        :param task_name: The name of the task to run.
        :param model_name: The name of the model to use.
        :param controlnet_name: The name of the controlnet to use.
        :param detect_kwargs: Keyword arguments for the detection task.
        """
        detect_task = Task.get(task_name, model_name)
        assert detect_task is not None, "Task not found"
        detect = detect_task()
        detect.load()
        detect_image = detect(image=base_image, **detect_kwargs)
        detect.unload()
        save_test_image(detect_image, f"sdxl_{controlnet_name}_detect")
        detect_result = pipe(
            control_image={controlnet_name: detect_image},
            control_scale={controlnet_name: control_scale},
            control_end={controlnet_name: 0.75},
            **kwargs
        )
        save_test_image(
            detect_result,
            f"sdxl_{controlnet_name}_result"
        )
        detect_i2i_result = pipe(
            control_image={controlnet_name: detect_image},
            control_scale={controlnet_name: control_scale},
            image=base_image,
            strength=0.8,
            **kwargs
        )
        save_test_image(
            detect_i2i_result,
            f"sdxl_{controlnet_name}_i2i_result"
        )
        return detect_image, detect_result, detect_i2i_result

    canny = run_test("edge-detection", "canny", "canny", control_scale=0.2)
    scribble = run_test("edge-detection", "hed", "scribble", control_scale=0.2, scribble=True)
    depth = run_test("depth-detection", "midas", "depth", control_scale=0.4)
    pose = run_test("pose-detection", "openpose", "pose")

    # QR Code is a bit of an outlier, we handle it manually
    kwargs["prompt"] = "a modern house facade"
    kwargs["num_inference_steps"] = 50
    kwargs["width"] = 1024
    kwargs["height"] = 1024
    kwargs.pop("highres_fix_strength")
    qr_source = get_test_image(subject="qrcode", size="512x512")
    qr_source = qr_source.resize((1024, 1024), resample=0) # nearest neighbor
    qr_result = pipe(
        control_image={"qr": qr_source},
        control_scale={"qr": 1.2},
        **kwargs
    )
    save_test_image(qr_result, "sdxl_qr_result")
    qr_i2i_result = pipe(
        control_image={"qr": qr_source},
        image=qr_result,
        strength=0.8,
        **kwargs
    )
    save_test_image(qr_i2i_result, "sdxl_qr_i2i_result")

    images = [
        (base_image, "base"),
        (canny[0], "canny detect"),
        (canny[1], "canny t2i result"),
        (canny[2], "canny i2i result"),
        (base_image, "base"),
        (scribble[0], "scribble detect"),
        (scribble[1], "scribble t2i result"),
        (scribble[2], "scribble i2i result"),
        (base_image, "base"),
        (depth[0], "depth detect"),
        (depth[1], "depth t2i result"),
        (depth[2], "depth i2i result"),
        (base_image, "base"),
        (pose[0], "pose detect"),
        (pose[1], "pose t2i result"),
        (pose[2], "pose i2i result"),
        (qr_source, "qr"),
        (qr_source, "qr"),
        (qr_result, "qr t2i result"),
        (qr_i2i_result, "qr i2i result"),
    ]

    grid = make_grid(
        images,
        image_size=(1344, 768),
        num_columns=4,
        font_size=14,
    )

    save_test_image(grid, "sdxl_controlnet_grid") # type: ignore[arg-type]
