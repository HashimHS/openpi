"""
Script to convert Kuka hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/kuka_real/convert_kuka_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>

Subtask (atomic action) labels
------------------------------
With `--subtasks` (the default) each frame is labeled with the instruction of the
atomic action that is being executed at that frame, instead of with the instruction of
the whole demonstration. The labels are read per episode directory from either

  * `subtasks.json`                  - the key frames of the transitions between the
                                       atomic actions plus their instructions, produced
                                       by the RoboTwin pipeline (`envs/_base_task.py` ->
                                       `subtasks/episode<i>.json` ->
                                       `description/utils/generate_subtask_instructions.py`
                                       -> `policy/pi05/scripts/process_data.py`), or
  * `instructions_frame_number.json` - one instruction per frame, the format of the real
                                       Kuka recordings.

Add `--split-subtask-episodes` to emit one LeRobot episode per atomic action instead of
one episode per demonstration.
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_arm_A1",
        "left_arm_A2",
        "left_arm_A3",
        "left_arm_A4",
        "left_arm_A5",
        "left_arm_A6",
        "left_arm_A7",
        "left_gripper",
        "right_arm_A1",
        "right_arm_A2",
        "right_arm_A3",
        "right_arm_A4",
        "right_arm_A5",
        "right_arm_A6",
        "right_arm_A7",
        "right_gripper",
    ]

    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                data = cv2.imdecode(data, cv2.IMREAD_COLOR)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                imgs_array.append(data)
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:].astype(np.float32))
        action = torch.from_numpy(ep["/action"][:].astype(np.float32))

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:].astype(np.float32))

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:].astype(np.float32))

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


NO_INSTRUCTION = "no instruction"


def load_subtask_segments(
    dir_path: str,
    num_frames: int,
    desc_type: str = "seen",
) -> list[tuple[int, int, str]] | None:
    """
    Read the key frames of the atomic actions of an episode from `subtasks.json` and
    return one `(start_frame, end_frame, instruction)` segment per atomic action.
    `end_frame` is exclusive. Returns None when the episode has no subtask labels.

    The file is written by `policy/pi05/scripts/process_data.py` from the key frames
    recorded during the data collection, one instruction phrasing is drawn per segment.
    """
    json_path = os.path.join(dir_path, "subtasks.json")
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r") as f_instr:
        subtask_data = json.load(f_instr)

    segments = []
    for subtask in subtask_data.get("subtasks", []):
        start_frame = max(0, int(subtask["start_frame"]))
        end_frame = min(num_frames, int(subtask["end_frame"]))
        if end_frame <= start_frame:
            continue
        instructions = subtask.get(desc_type) or subtask.get("seen") or []
        if not instructions:
            print(f"WARNING: {json_path}: subtask {subtask.get('index')} "
                  f"({subtask.get('action')}) has no instruction, its frames are dropped")
            continue
        segments.append((start_frame, end_frame, str(np.random.choice(instructions))))

    return segments


def load_frame_instruction_segments(dir_path: str, num_frames: int) -> list[tuple[int, int, str]]:
    """
    Fall back for data that stores one instruction per frame in
    `instructions_frame_number.json` (the format of the real Kuka recordings).
    Consecutive frames sharing an instruction are grouped into a single segment.
    """
    with open(os.path.join(dir_path, "instructions_frame_number.json"), "r") as f_instr:
        instructions = json.load(f_instr)["instructions"]

    segments = []
    for i in range(min(num_frames, len(instructions))):
        instruction = instructions[i]
        if instruction == NO_INSTRUCTION:
            continue
        if segments and segments[-1][2] == instruction and segments[-1][1] == i:
            segments[-1] = (segments[-1][0], i + 1, instruction)
        else:
            segments.append((i, i + 1, instruction))
    return segments


def get_episode_segments(
    dir_path: str,
    num_frames: int,
    subtasks: bool,
    desc_type: str = "seen",
) -> list[tuple[int, int, str]]:
    """Frame ranges of an episode together with the instruction to train them with."""
    if not subtasks:
        with open(os.path.join(dir_path, "instructions.json"), "r") as f_instr:
            instructions = json.load(f_instr)["instructions"]
        return [(0, num_frames, str(np.random.choice(instructions)))]

    segments = load_subtask_segments(dir_path, num_frames, desc_type=desc_type)
    if segments is None:
        segments = load_frame_instruction_segments(dir_path, num_frames)
    return segments


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    subtasks: bool = True,
    split_subtask_episodes: bool = False,
    desc_type: str = "seen",
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]
        # add prompt
        dir_path = os.path.dirname(ep_path)
        segments = get_episode_segments(dir_path, num_frames, subtasks, desc_type=desc_type)

        if not segments:
            print(f"WARNING: {ep_path} has no labeled frame, skipping it")
            continue

        def add_frames(start_frame: int, end_frame: int, instruction: str):
            for i in range(start_frame, end_frame):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": instruction,
                }

                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]

                if velocity is not None:
                    frame["observation.velocity"] = velocity[i]
                if effort is not None:
                    frame["observation.effort"] = effort[i]
                dataset.add_frame(frame)

        if split_subtask_episodes:
            # One LeRobot episode per atomic action, so that action chunks never
            # cross the transition from one subtask to the next.
            for start_frame, end_frame, instruction in segments:
                add_frames(start_frame, end_frame, instruction)
                dataset.save_episode()
        else:
            # One LeRobot episode per demonstration, every frame labeled with the
            # instruction of the atomic action it belongs to. Unlabeled frames are
            # dropped.
            for start_frame, end_frame, instruction in segments:
                add_frames(start_frame, end_frame, instruction)
            dataset.save_episode()

    return dataset


def port_kuka(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    subtasks: bool = False,
    split_subtask_episodes: bool = False,
    desc_type: str = "seen",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        # download_raw(raw_dir, repo_id=raw_repo_id)
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "bimanual_kuka",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        subtasks=subtasks,
        split_subtask_episodes=split_subtask_episodes,
        desc_type=desc_type,
        episodes=episodes,
    )
    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_kuka) # To recognize you need to add the arguments, you can run `python convert_kuka_data_to_lerobot_robotwin.py --help`
