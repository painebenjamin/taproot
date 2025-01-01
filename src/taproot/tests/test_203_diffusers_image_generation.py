from taproot.util import (
    debug_logger,
    execute_task_test_suite,
    save_test_image
)

def run_image_generation_test_suite(
    model_name: str,
) -> None:
    """
    Runs a test using a specific model name to generate a preconfigured image.

    TODO: Add more tests beyond text-to-image.
    """
    with debug_logger() as logger:
        logger.info(f"Running image generation test suite for model: {model_name}")
        prompt = "An orange cat sleeping on a brown couch"
        seed = 12345
        [result] = execute_task_test_suite(
            "image-generation",
            model=model_name,
            assert_runtime_memory_ratio=None,
            num_exercise_executions=3,
            cases = [
                ({"prompt": prompt, "seed": seed},None),
            ]
        )
        save_test_image(result, model_name)

"""
Stable Diffusion 1.5
"""

def test_sd15() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5")

def test_sd15_abyssorange_mix() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-abyssorange-mix-v3")

def test_sd15_chillout_mix() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-chillout-mix-ni")

def test_sd15_dark_sushi_mix() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-dark-sushi-mix-v2-25d")

def test_sd15_divine_elegance_mix() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-divine-elegance-mix-v10")

def test_sd15_dreamshaper() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-dreamshaper-v8")

def test_sd15_epicphotogasm() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-epicphotogasm-ultimate-fidelity")

def test_sd15_epicrealism() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-epicrealism-v5")

def test_sd15_ghostmix() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-ghostmix-v2")

def test_sd15_lyriel() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-lyriel-v1-6")

def test_sd15_majicmix() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-majicmix-realistic-v7")

def test_sd15_meinamix() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-meinamix-v12")

def test_sd15_mistoon() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-mistoon-anime-v3")

def test_sd15_perfect_world() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-perfect-world-v6")

def test_sd15_photon() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-photon-v1")

def test_sd15_realcartoon() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-realcartoon3d-v17")

def test_sd15_realistic_vision_v51() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-realistic-vision-v5-1")

def test_sd15_realistic_vision_v60() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-realistic-vision-v6-0")

def test_sd15_rev_animated() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-rev-animated-v2")

def test_sd15_toonyou() -> None:
    run_image_generation_test_suite("stable-diffusion-v1-5-toonyou-beta-v6")

"""
Stable Diffusion XL
"""

def test_sdxl() -> None:
    run_image_generation_test_suite("stable-diffusion-xl")

def test_sdxl_albedobase() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-albedobase-v3-1")

def test_sdxl_animagine() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-animagine-v3-1")

def test_sdxl_anything() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-anything")

def test_sdxl_copax_timeless() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-copax-timeless-v13")

def test_sdxl_counterfeit() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-counterfeit-v2-5")

def test_sdxl_dreamshaper() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-dreamshaper-alpha-v2")

def test_sdxl_hello_world() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-helloworld-v7")

def test_sdxl_juggernaut() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-juggernaut-v11")

def test_sdxl_nightvision() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-nightvision-v9")

def test_sdxl_realvis() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-realvis-v5")

def test_sdxl_newreality() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-stoiqo-newreality-pro")

def test_sdxl_unstable_diffusers() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-unstable-diffusers-nihilmania")

def test_sdxl_zavychroma() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-zavychroma-v10")

def test_sdxl_lightning_8step() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-lightning-8-step")

def test_sdxl_lightning_4step() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-lightning-4-step")

def test_sdxl_lightning_2step() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-lightning-2-step")

def test_sdxl_turbo() -> None:
    run_image_generation_test_suite("stable-diffusion-xl-turbo")

"""SD3"""

def test_sd3() -> None:
    run_image_generation_test_suite("stable-diffusion-v3-medium")

def test_sd35_medium() -> None:
    run_image_generation_test_suite("stable-diffusion-v3-5-medium")

def test_sd35_large() -> None:
    run_image_generation_test_suite("stable-diffusion-v3-5-large")

def test_sd35_large_int8() -> None:
    run_image_generation_test_suite("stable-diffusion-v3-5-large-int8")

def test_sd35_large_nf4() -> None:
    run_image_generation_test_suite("stable-diffusion-v3-5-large-nf4")

"""FLUX"""

def test_flux_dev_int8() -> None:
    run_image_generation_test_suite("flux-v1-dev-int8")

def test_flux_dev_nf4() -> None:
    run_image_generation_test_suite("flux-v1-dev-nf4")

def test_flux_dev_stoiqo_newreality_int8() -> None:
    run_image_generation_test_suite("flux-v1-dev-stoiqo-newreality-alpha-v2-int8")

def test_flux_dev_stoiqo_newreality_nf4() -> None:
    run_image_generation_test_suite("flux-v1-dev-stoiqo-newreality-alpha-v2-nf4")
