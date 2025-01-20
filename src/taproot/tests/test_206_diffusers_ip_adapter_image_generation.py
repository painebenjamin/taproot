from taproot import Task
from taproot.util import (
    debug_logger,
    get_test_image,
    save_test_image,
    make_grid,
)

def test_sd15_ip_adapter_image_generation() -> None:
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
