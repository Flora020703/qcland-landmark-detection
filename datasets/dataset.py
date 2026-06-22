# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import json
from pathlib import Path
from typing import Callable, Optional
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: Path,  # MODIFIED: zip→folder (was zip_path)
        img_suffix: str,
        target_parser: Callable,
        check_empty_targets: bool,
        transforms: Optional[Callable] = None,
        only_annotations_json: bool = False,
        target_suffix: str = None,
        stuff_classes: Optional[list[int]] = None,
        img_stem_suffix: str = "",
        target_stem_suffix: str = "",
        target_data_path: Optional[Path] = None,  # MODIFIED: zip→folder (was target_zip_path)
        target_instance_data_path: Optional[Path] = None,  # MODIFIED: zip→folder (was target_instance_zip_path)
        img_folder_path: Path = Path("./"),  # MODIFIED: zip→folder (was img_folder_path_in_zip)
        target_folder_path: Path = Path("./"),  # MODIFIED: zip→folder (was target_folder_path_in_zip)
        target_instance_folder_path: Path = Path("./"),  # MODIFIED: zip→folder (was target_instance_folder_path_in_zip)
        annotations_json_path: Optional[Path] = None,  # MODIFIED: zip→folder (was annotations_json_path_in_zip)
    ):
        self.data_path = data_path  # MODIFIED: zip→folder
        self.target_parser = target_parser
        self.transforms = transforms
        self.only_annotations_json = only_annotations_json
        self.stuff_classes = stuff_classes
        self.target_data_path = target_data_path  # MODIFIED: zip→folder
        self.target_instance_data_path = target_instance_data_path  # MODIFIED: zip→folder
        self.img_folder_path = img_folder_path  # MODIFIED: zip→folder
        self.target_folder_path = target_folder_path  # MODIFIED: zip→folder
        self.target_instance_folder_path = target_instance_folder_path  # MODIFIED: zip→folder
        self.labels_by_id = {}
        self.polygons_by_id = {}
        self.is_crowd_by_id = {}

        # MODIFIED: zip→folder — read JSON directly from filesystem instead of zip
        if annotations_json_path is not None:
            json_file_path = (target_data_path or data_path) / annotations_json_path
            with open(json_file_path, "r") as file:
                annotation_data = json.load(file)

            image_id_to_file_name = {
                image["id"]: image["file_name"] for image in annotation_data["images"]
            }

            for annotation in annotation_data["annotations"]:
                img_filename = image_id_to_file_name[annotation["image_id"]]

                if "segments_info" in annotation:
                    self.labels_by_id[img_filename] = {
                        segment_info["id"]: segment_info["category_id"]
                        for segment_info in annotation["segments_info"]
                    }
                    self.is_crowd_by_id[img_filename] = {
                        segment_info["id"]: bool(segment_info["iscrowd"])
                        for segment_info in annotation["segments_info"]
                    }
                else:
                    if img_filename not in self.labels_by_id:
                        self.labels_by_id[img_filename] = {}

                    if img_filename not in self.polygons_by_id:
                        self.polygons_by_id[img_filename] = {}

                    if img_filename not in self.is_crowd_by_id:
                        self.is_crowd_by_id[img_filename] = {}

                    self.labels_by_id[img_filename][annotation["id"]] = annotation[
                        "category_id"
                    ]
                    self.polygons_by_id[img_filename][annotation["id"]] = annotation[
                        "segmentation"
                    ]
                    self.is_crowd_by_id[img_filename][annotation["id"]] = bool(
                        annotation["iscrowd"]
                    )

        self.imgs = []
        self.targets = []
        self.targets_instance = []

        # MODIFIED: zip→folder — resolve absolute folder paths on disk
        img_folder_full_path = (data_path / img_folder_path).resolve()
        target_folder_full_path = ((target_data_path or data_path) / target_folder_path).resolve()
        target_instance_folder_full_path = (
            (target_instance_data_path or data_path) / target_instance_folder_path
        ).resolve()

        # MODIFIED: zip→folder — iterate files via rglob instead of zip.infolist()
        img_pattern = f"*{img_stem_suffix}{img_suffix}"
        img_files = sorted(img_folder_full_path.rglob(img_pattern), key=lambda p: p.as_posix())

        for img_path in img_files:
            if not img_path.is_file():
                continue

            target_path = None
            if not only_annotations_json:
                rel_path = img_path.relative_to(img_folder_full_path)
                target_parent = target_folder_full_path / rel_path.parent
                target_stem = rel_path.stem.replace(img_stem_suffix, target_stem_suffix)
                target_path = (target_parent / f"{target_stem}{target_suffix}").resolve()

            if self.labels_by_id:
                if img_path.name not in self.labels_by_id:
                    continue

                if not self.labels_by_id[img_path.name]:
                    continue
            else:
                # Only check target_path if it was defined (not only_annotations_json)
                if target_path is not None:
                    # MODIFIED: zip→folder — check file existence instead of zip namelist
                    if not target_path.exists():
                        continue

                    if check_empty_targets:
                        try:
                            min_val, max_val = Image.open(target_path).getextrema()
                        except Exception:
                            continue
                        if min_val == max_val:
                            continue

            # MODIFIED: zip→folder — use data path instead of zip path
            if target_instance_data_path is not None:
                target_instance_path = (target_instance_folder_full_path / (target_stem + target_suffix)).resolve()

                if check_empty_targets:
                    try:
                        extrema = Image.open(target_instance_path).getextrema()
                    except Exception:
                        # If instance file missing or unreadable, treat as empty -> skip if no labels
                        _, labels, _ = self.target_parser(
                            target=tv_tensors.Mask(Image.open(target_path)),
                            target_instance=None,
                            stuff_classes=self.stuff_classes,
                        )
                        if not labels:
                            continue
                    else:
                        if all(min_val == max_val for min_val, max_val in extrema):
                            _, labels, _ = self.target_parser(
                                target=tv_tensors.Mask(Image.open(target_path)),
                                target_instance=tv_tensors.Mask(Image.open(target_instance_path)),
                                stuff_classes=self.stuff_classes,
                            )
                            if not labels:
                                continue

            # MODIFIED: zip→folder — store Path objects instead of posix strings
            self.imgs.append(img_path)
            if not only_annotations_json:
                self.targets.append(target_path)
            if target_instance_data_path is not None:
                self.targets_instance.append(target_instance_path)

    def __getitem__(self, index: int):
        # MODIFIED: zip→folder — open files directly instead of via zip
        img = tv_tensors.Image(Image.open(self.imgs[index]).convert("RGB"))

        target = None
        if not self.only_annotations_json:
            target = tv_tensors.Mask(Image.open(self.targets[index]), dtype=torch.long)

            if img.shape[-2:] != target.shape[-2:]:
                target = F.resize(
                    target,
                    list(img.shape[-2:]),
                    interpolation=F.InterpolationMode.NEAREST,
                )

        target_instance = None
        if self.targets_instance:
            target_instance = tv_tensors.Mask(
                Image.open(self.targets_instance[index]), dtype=torch.long
            )

        masks, labels, is_crowd = self.target_parser(
            target=target,
            target_instance=target_instance,
            stuff_classes=self.stuff_classes,
            polygons_by_id=self.polygons_by_id.get(self.imgs[index].name, {}),
            labels_by_id=self.labels_by_id.get(self.imgs[index].name, {}),
            is_crowd_by_id=self.is_crowd_by_id.get(self.imgs[index].name, {}),
            width=img.shape[-1],
            height=img.shape[-2],
        )

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels),
            "is_crowd": torch.tensor(is_crowd),
        }

        if self.transforms is not None:
            result = self.transforms(img, target)

            # Handle dual-output mode (returns dict with teacher/student versions)
            if isinstance(result, dict) and 'image_teacher' in result:
                # Dual mode: return dict as-is for collate function to handle
                return result
            else:
                # Single mode: unpack tuple
                img, target = result

        return img, target

    def __len__(self):
        return len(self.imgs)
