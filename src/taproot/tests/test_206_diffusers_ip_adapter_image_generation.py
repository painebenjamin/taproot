from taproot import Task
from taproot.util import (
    debug_logger,
    get_test_image,
    save_test_image,
    make_grid,
)

def _test_sd15_ip_adapter_image_generation() -> None:
    """
    Exercises all IP adapters for SD 1.5.
    """
    with debug_logger():
        kwargs = {
            "scheduler": "k_dpm_2_discrete_karras",
            "num_inference_steps": 28,
            "seed": 12345,
            "guidance_scale": 7.5,
        }
        sd = Task.get("image-generation", "stable-diffusion-v1-5-epicrealism-v5")
        assert sd is not None, "Task not found"
        pipe = sd()
        pipe.load()

        source_image = get_test_image(
            subject="person",
            size="512x512",
            number=10
        )
        result_images = []

        for ip_model in ["base", "light", "plus", "plus-face", "full-face"]:
            result_images.append((source_image, "source"))
            result = pipe(
                prompt="a photograph of a person",
                ip_adapter_image={ip_model: source_image},
                **kwargs
            )
            result_images.append((result, ip_model))
            save_test_image(result, f"sd15_ip_{ip_model}")

            i2i_result = pipe(
                prompt="a photograph of a person",
                image=source_image,
                strength=0.8,
                ip_adapter_image={ip_model: source_image},
                **kwargs
            )
            result_images.append((i2i_result, f"i2i_{ip_model}"))
            save_test_image(i2i_result, f"sd15_i2i_{ip_model}")

        grid = make_grid(
            result_images,
            num_columns=3
        )
        save_test_image(grid, "sd15_ip_grid") # type: ignore[arg-type]

def test_sdxl_ip_adapter_image_generation() -> None:
    """
    Exercises all IP adapters for SD XL.
    """
    with debug_logger():
        kwargs = {
            "prompt": "RAW photo, a photograph of a woman with shoulder-length dark brown hair standing outside, wearing sleveless brown top and necklace, holding up arm, overcast sky",
            "num_inference_steps": 28,
            "negative_prompt": "black-and-white, monochrome, sepia, grayscale",
            "seed": 12345,
            "guidance_scale": 4.5,
        }
        sdxl = Task.get("image-generation", "stable-diffusion-xl-juggernaut-v11")
        assert sdxl is not None, "Task not found"
        pipe = sdxl()
        pipe.load()

        source_image = get_test_image(
            subject="person",
            size="512x512",
            number=10
        )
        source_image = source_image.resize((1024, 1024))
        result_images = []

        for ip_model in ["base", "plus", "plus-face"]:
            result_images.append((source_image, "source"))
            result = pipe(
                ip_adapter_image={ip_model: source_image},
                **kwargs
            )
            result_images.append((result, ip_model))
            save_test_image(result, f"sdxl_ip_{ip_model}")

            i2i_result = pipe(
                image=source_image,
                strength=0.8,
                ip_adapter_image={ip_model: source_image},
                **kwargs
            )
            result_images.append((i2i_result, f"i2i_{ip_model}"))
            save_test_image(i2i_result, f"sdxl_i2i_{ip_model}")

        grid = make_grid(
            result_images,
            num_columns=3
        )
        save_test_image(grid, "sdxl_ip_grid") # type: ignore[arg-type]
