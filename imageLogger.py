from collections import namedtuple
import os
from pytorch_lightning import Callback
import torch
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class InferenceCallback(Callback):
    def __init__(self):
        self.test_paths = []

        Test_tuple = namedtuple(
            "Test_tuple",
            ["reference_image_path", "bg_image_path", "bg_mask_path", "save_path"],
        )
        # TODO: change save_path for each checkpoint
        bg_upper_dirs = [
            [
                "./examples/SUS/BG/Eva_0.png",
                "./examples/SUS/BG/Eva_mask_upper.png",
                "upper_box",
            ],
            [
                "./examples/SUS/BG/Eva_0.png",
                "./examples/SUS/BG/Eva_mask_upper_smaller.png",
                "upper_small_box",
            ],
            [
                "./examples/SUS/BG/Eva_0.png",
                "./examples/SUS/BG/Eva_mask_upper_long_sleeves.png",
                "tailored_long_sleeves",
            ],
            [
                "./examples/SUS/BG/Eva_0.png",
                "./examples/SUS/BG/Eva_mask_upper_short_sleeves.png",
                "tailored_short_sleeves",
            ],
        ]
        bg_lower_dirs = [
            [
                "./examples/SUS/BG/Eva_0.png",
                "./examples/SUS/BG/Eva_mask_lower.png",
                "lower_box",
            ],
            [
                "./examples/SUS/BG/Eva_0.png",
                "./examples/SUS/BG/Eva_mask_lower_smaller.png",
                "lower_small_box",
            ],
            [
                "./examples/SUS/BG/Eva_0.png",
                "./examples/SUS/BG/Eva_mask_lower_short.png",
                "lower_tailored_short",
            ],
            [
                "./examples/SUS/BG/Eva_0.png",
                "./examples/SUS/BG/Eva_mask_lower_long.png",
                "lower_tailored_long",
            ],
        ]

        root_directory = "./examples/SUS/FG"
        for index, path_tuple in enumerate(bg_upper_dirs):
            for dirpath, dirnames, filenames in os.walk(
                os.path.join(root_directory, "upper")
            ):
                for filename in filenames:
                    file_path = os.path.join(
                        dirpath, filename
                    )  # Construct full file path
                    file_id = filename.split("_")[0]
                    test_tuple = Test_tuple(
                        reference_image_path=file_path,
                        bg_image_path=path_tuple[0],
                        bg_mask_path=path_tuple[1],
                        save_path=os.path.join(
                            f"./examples/SUS/GEN/{path_tuple[2]}",
                            f"{file_id}_upper_{path_tuple[2]}.png",
                        ),
                    )
                    test_paths.append(test_tuple)

        for index, path_tuple in enumerate(bg_lower_dirs):
            for dirpath, dirnames, filenames in os.walk(
                os.path.join(root_directory, "lower")
            ):
                for filename in filenames:
                    file_path = os.path.join(
                        dirpath, filename
                    )  # Construct full file path
                    file_id = filename.split("_")[0]
                    test_tuple = Test_tuple(
                        reference_image_path=file_path,
                        bg_image_path=path_tuple[0],
                        bg_mask_path=path_tuple[1],
                        save_path=os.path.join(
                            f"./examples/SUS/GEN/{path_tuple[2]}",
                            f"{file_id}_lower_{path_tuple[2]}.png",
                        ),
                    )
                    test_paths.append(test_tuple)

        # used to test the mask generation code
        # mask_test = Test_tuple(
        #     reference_image_path="./examples/SUS/FG/lower/1003368624002_seg.png",
        #     bg_image_path="./examples/SUS/BG/Eva_0.png",
        #     bg_mask_path="./examples/SUS/BG/Eva_mask_lower_long.png",
        #     save_path="./examples/SUS/GEN/test.png",
        # )

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the validation epoch with the inferred outputs.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The LightningModule instance (the model).
        """

        # Perform inference
        pl_module.eval()  # Set the model to eval mode
        with torch.no_grad():  # Disable gradient computation
            for test in self.test_paths:
                reference_image_path = test.reference_image_path
                bg_image_path = test.bg_image_path
                bg_mask_path = test.bg_mask_path
                save_path = test.save_path
                file_id = save_path.split("/")[-1].split(".")[0]

                # reference image + reference mask
                # target image should have the alpha channel set, to generate the mask
                # You could use the demo of SAM to extract RGB-A image with masks
                # https://segment-anything.com/demo
                image = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)
                self.check_image_format(image)
                mask = (image[:, :, -1] > 128).astype(np.uint8)

                image = image[:, :, :-1]
                image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                ref_image = image
                ref_mask = mask

                # background image
                back_image = cv2.imread(bg_image_path).astype(np.uint8)
                back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

                # background mask
                tar_mask = cv2.imread(bg_mask_path)[:, :, 0] > 128
                tar_mask = tar_mask.astype(np.uint8)

                gen_image = self._inference_single_image(
                    ref_image, ref_mask, back_image.copy(), tar_mask, file_id
                )

                h, w = back_image.shape[0], back_image.shape[0]
                ref_image = cv2.resize(ref_image, (w, h))
                vis_image = cv2.hconcat([ref_image, back_image, gen_image])

                cv2.imwrite(save_path, vis_image[:, :, ::-1])

    def _check_image_format(self, image):
        # Check if the image was loaded successfully
        if image is None:
            logger.error("Error: Image not loaded. Check the file path.")
            return False
        # Check if the image has 4 channels (RGBA, including alpha channel)
        if image.shape[2] != 4:
            logger.error(
                "Error: Image does not have 4 channels. It is required to have an alpha channel."
            )
            return False
        return True

    def _inference_single_image(
        self, ref_image, ref_mask, tar_image, tar_mask, file_id, guidance_scale=4
    ):
        item = process_pairs(ref_image, ref_mask, tar_image, tar_mask, file_id)
        ref = item["ref"] * 255
        tar = item["jpg"] * 127.5 + 127.5
        hint = item["hint"] * 127.5 + 127.5

        hint_image = hint[:, :, :-1]
        hint_mask = item["hint"][:, :, -1] * 255
        hint_mask = np.stack([hint_mask, hint_mask, hint_mask], -1)
        ref = cv2.resize(ref.astype(np.uint8), (512, 512))

        seed = random.randint(0, 65535)
        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        ref = item["ref"]
        tar = item["jpg"]
        hint = item["hint"]
        num_samples = 1

        control = torch.from_numpy(hint.copy()).float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        clip_input = torch.from_numpy(ref.copy()).float().cuda()
        clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
        clip_input = einops.rearrange(clip_input, "b h w c -> b c h w").clone()

        guess_mode = False
        H, W = 512, 512

        cond = {
            "c_concat": [control],  # the Detail Maps
            "c_crossattn": [
                model.get_learned_conditioning(clip_input)
            ],  # the ID tokens, get method is the original diffusion method model
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [
                model.get_learned_conditioning(
                    [torch.zeros((1, 3, 224, 224))] * num_samples
                )
            ],
        }
        shape = (4, H // 8, W // 8)

        if save_memory:
            model.low_vram_shift(is_diffusing=True)

        # ====
        num_samples = (
            1  # gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
        )
        image_resolution = 512  # gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
        strength = 1.0  # gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
        guess_mode = False  # gr.Checkbox(label='Guess Mode', value=False)
        # detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
        ddim_steps = (
            50  # gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
        )
        scale = guidance_scale  # gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
        seed = (
            -1
        )  # gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
        eta = 0.0  # gr.Number(label="eta (DDIM)", value=0.0)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )
        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
        )  # .clip(0, 255).astype(np.uint8)

        result = x_samples[0][:, :, ::-1]
        result = np.clip(result, 0, 255)

        pred = x_samples[0]
        pred = np.clip(pred, 0, 255)[1:, :, :]
        sizes = item["extra_sizes"]
        tar_box_yyxx_crop = item["tar_box_yyxx_crop"]
        gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop)
        return gen_image

    def _process_pairs(self, ref_image, ref_mask, tar_image, tar_mask, file_id):
        # ========= Reference ===========
        # ref expand
        ref_box_yyxx = get_bbox_from_mask(ref_mask)

        # ref filter mask
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (
            1 - ref_mask_3
        )

        y1, y2, x1, x2 = ref_box_yyxx

        masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
        ref_mask = ref_mask[y1:y2, x1:x2]

        ratio = np.random.randint(12, 13) / 10
        masked_ref_image, ref_mask = expand_image_mask(
            masked_ref_image, ref_mask, ratio=ratio
        )
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)

        # to square and resize
        masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
        masked_ref_image = cv2.resize(masked_ref_image, (224, 224)).astype(np.uint8)

        ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value=0, random=False)
        ref_mask_3 = cv2.resize(ref_mask_3, (224, 224)).astype(np.uint8)
        ref_mask = ref_mask_3[:, :, 0]

        # ref aug
        masked_ref_image_aug = masked_ref_image  # aug_data(masked_ref_image)

        # collage aug
        masked_ref_image_compose, ref_mask_compose = (
            masked_ref_image,
            ref_mask,
        )  # aug_data_mask(masked_ref_image, ref_mask)
        masked_ref_image_aug = masked_ref_image_compose.copy()
        ref_mask_3 = np.stack(
            [ref_mask_compose, ref_mask_compose, ref_mask_compose], -1
        )
        ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose / 255)
        # cv2.imwrite(f"processed_images/{file_id}_hf_map.png", ref_image_collage)

        # ========= Target ===========
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1, 1.2])

        # crop
        tar_box_yyxx_crop = expand_bbox(
            tar_image, tar_box_yyxx, ratio=[1.1, 1.2]
        )  # 1.2 1.6
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)  # crop box
        y1, y2, x1, x2 = tar_box_yyxx_crop

        cropped_target_image = tar_image[y1:y2, x1:x2, :]
        tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
        y1, y2, x1, x2 = tar_box_yyxx
        # cv2.imwrite(
        #     f"processed_images/{file_id}_cropped_and_squared_target_image.png",
        #     cropped_target_image,
        # )

        # collage
        ref_image_collage = cv2.resize(ref_image_collage, (x2 - x1, y2 - y1))
        ref_mask_compose = cv2.resize(
            ref_mask_compose.astype(np.uint8), (x2 - x1, y2 - y1)
        )
        ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)
        # cv2.imwrite(f"processed_images/{file_id}_ref_image_collage.png", ref_image_collage)

        collage = cropped_target_image.copy()
        collage[y1:y2, x1:x2, :] = ref_image_collage
        # cv2.imwrite(f"processed_images/{file_id}_collage.png", collage)

        collage_mask = cropped_target_image.copy() * 0.0
        collage_mask[y1:y2, x1:x2, :] = 1.0

        # the size before pad
        H1, W1 = collage.shape[0], collage.shape[1]
        cropped_target_image = pad_to_square(
            cropped_target_image, pad_value=0, random=False
        ).astype(np.uint8)
        collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value=-1, random=False).astype(
            np.uint8
        )
        # cv2.imwrite(f"processed_images/{file_id}_collage_after_square_padding.png", collage)

        # the size after pad
        H2, W2 = collage.shape[0], collage.shape[1]
        cropped_target_image = cv2.resize(cropped_target_image, (512, 512)).astype(
            np.float32
        )
        collage = cv2.resize(collage, (512, 512)).astype(np.float32)
        collage_mask = (
            cv2.resize(collage_mask, (512, 512)).astype(np.float32) > 0.5
        ).astype(np.float32)
        # cv2.imwrite(f"processed_images/{file_id}_collage_after_resize.png", collage)

        masked_ref_image_aug = masked_ref_image_aug / 255
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0
        collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)

        item = dict(
            ref=masked_ref_image_aug.copy(),
            jpg=cropped_target_image.copy(),
            hint=collage.copy(),
            extra_sizes=np.array([H1, W1, H2, W2]),
            tar_box_yyxx_crop=np.array(tar_box_yyxx_crop),
        )
        return item
