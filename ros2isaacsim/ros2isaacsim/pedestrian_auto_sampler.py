from __future__ import annotations

import gzip
import heapq
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_CHARACTER_NAMES = [
    "original_female_adult_business_02",
    "original_female_adult_medical_01",
    "original_female_adult_police_01",
    "original_female_adult_police_02",
    "original_female_adult_police_03",
    "original_male_adult_construction_01",
    "original_male_adult_construction_02",
    "original_male_adult_construction_03",
    "original_male_adult_construction_05",
    "original_male_adult_medical_01",
    "original_male_adult_police_04",
]


@dataclass
class AutoPedestrianConfig:
    map_path: str
    count: int = 2
    seed: int = 12345
    clearance: float = 0.35
    min_path_length: float = 1.2
    max_path_length: float = 5.5
    boundary_margin: float = 0.20
    velocity_min: float = 0.35
    velocity_max: float = 0.50
    stage_prefix: str = "social_ped_"
    character_names: list[str] | None = None
    loop_path: bool = True


class VoxelPedestrianSampler:

    def __init__(self, config: AutoPedestrianConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self._load_map()

    def _load_map(self) -> None:
        path = Path(self.config.map_path)
        if not path.exists():
            raise FileNotFoundError(f"voxel map not found: {path}")
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

        self.resolution = float(data.get("resolution", 0.05))
        origin = data.get("origin", [0.0, 0.0, 0.0])
        self.origin_x = float(origin[0])
        self.origin_y = float(origin[1])

        grid_bounds = data.get("grid_bounds")
        if not grid_bounds or len(grid_bounds) < 4:
            raise RuntimeError("voxel map missing grid_bounds")
        self.min_ix = int(grid_bounds[0])
        self.max_ix = int(grid_bounds[1])
        self.min_iy = int(grid_bounds[2])
        self.max_iy = int(grid_bounds[3])
        self.width = self.max_ix - self.min_ix + 1
        self.height = self.max_iy - self.min_iy + 1

        occupied = np.zeros((self.height, self.width), dtype=np.uint8)
        columns = data.get("columns") or []
        if columns:
            for column in columns:
                if len(column) < 2:
                    continue
                ix = int(column[0])
                iy = int(column[1])
                row = iy - self.min_iy
                col = ix - self.min_ix
                if 0 <= row < self.height and 0 <= col < self.width:
                    occupied[row, col] = 1
        else:
            for voxel in data.get("voxels", []):
                if len(voxel) < 2:
                    continue
                ix = int(voxel[0])
                iy = int(voxel[1])
                row = iy - self.min_iy
                col = ix - self.min_ix
                if 0 <= row < self.height and 0 <= col < self.width:
                    occupied[row, col] = 1
        self.occupied = occupied
        self.passable = self._build_passable_mask()

    def _build_passable_mask(self) -> np.ndarray:
        clearance_cells = max(0, int(math.ceil(self.config.clearance / self.resolution)))
        free = (self.occupied == 0).astype(np.uint8)
        if clearance_cells > 0:
            passable = self._all_free_window_mask(free, clearance_cells)
        else:
            passable = free.astype(bool)

        margin_cells = max(0, int(math.ceil(self.config.boundary_margin / self.resolution)))
        if margin_cells > 0 and self.height > 2 * margin_cells and self.width > 2 * margin_cells:
            passable[:margin_cells, :] = False
            passable[-margin_cells:, :] = False
            passable[:, :margin_cells] = False
            passable[:, -margin_cells:] = False
        return passable

    def _all_free_window_mask(self, free: np.ndarray, radius: int) -> np.ndarray:
        window = int(2 * radius + 1)
        padded = np.pad(free.astype(np.int32), radius, mode="constant", constant_values=0)
        integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant", constant_values=0)
        integral = integral.cumsum(axis=0).cumsum(axis=1)
        sums = (
            integral[window:, window:]
            - integral[:-window, window:]
            - integral[window:, :-window]
            + integral[:-window, :-window]
        )
        return sums == (window * window)

    def _grid_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        ix = self.min_ix + col
        iy = self.min_iy + row
        x = self.origin_x + (ix + 0.5) * self.resolution
        y = self.origin_y + (iy + 0.5) * self.resolution
        return float(x), float(y)

    def _neighbors(self, cell: tuple[int, int]):
        row, col = cell
        for drow, dcol in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            nrow = row + drow
            ncol = col + dcol
            if nrow < 0 or ncol < 0 or nrow >= self.height or ncol >= self.width:
                continue
            if not self.passable[nrow, ncol]:
                continue
            yield (nrow, ncol), (math.sqrt(2.0) if drow and dcol else 1.0)

    def _astar(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        if not self.passable[start] or not self.passable[goal]:
            return []

        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor, step_cost in self._neighbors(current):
                tentative = g_score[current] + step_cost
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heuristic = math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                heapq.heappush(open_heap, (tentative + heuristic, neighbor))

        return []

    def _line_is_free(self, start: tuple[int, int], goal: tuple[int, int]) -> bool:
        srow, scol = start
        grow, gcol = goal
        steps = int(max(abs(grow - srow), abs(gcol - scol)))
        if steps <= 0:
            return bool(self.passable[start])
        for idx in range(steps + 1):
            t = idx / steps
            row = int(round(srow + (grow - srow) * t))
            col = int(round(scol + (gcol - scol) * t))
            if row < 0 or col < 0 or row >= self.height or col >= self.width:
                return False
            if not self.passable[row, col]:
                return False
        return True

    def _simplify_path(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        anchor = path[0]
        probe = 1
        while probe < len(path):
            if not self._line_is_free(anchor, path[probe]):
                last_ok = path[probe - 1]
                if last_ok != simplified[-1]:
                    simplified.append(last_ok)
                anchor = last_ok
            else:
                probe += 1
        if simplified[-1] != path[-1]:
            simplified.append(path[-1])
        return simplified

    def _path_length(self, path: list[tuple[int, int]]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for start, goal in zip(path[:-1], path[1:]):
            total += math.hypot(goal[0] - start[0], goal[1] - start[1])
        return total * self.resolution

    def _sample_cell(self, candidates: np.ndarray) -> tuple[int, int]:
        row, col = candidates[self.rng.randrange(len(candidates))]
        return int(row), int(col)

    def sample_path(self, *, max_tries: int = 128) -> list[list[float]]:
        candidates = np.argwhere(self.passable)
        if len(candidates) == 0:
            raise RuntimeError("no passable cells available in voxel map")

        min_len = max(0.2, float(self.config.min_path_length))
        max_len = max(min_len, float(self.config.max_path_length))
        for _ in range(max_tries):
            start = self._sample_cell(candidates)
            goal = self._sample_cell(candidates)
            if start == goal:
                continue
            straight = math.hypot(goal[0] - start[0], goal[1] - start[1]) * self.resolution
            if straight < min_len or straight > max_len:
                continue
            path = self._astar(start, goal)
            if len(path) < 2:
                continue
            length = self._path_length(path)
            if length < min_len or length > max_len:
                continue
            simplified = self._simplify_path(path)
            result = []
            for cell in simplified:
                x, y = self._grid_to_world(cell)
                result.append([x, y, 0.0])
            if len(result) >= 2:
                return result
        raise RuntimeError(f"failed to sample pedestrian route after {max_tries} tries")

    def sample_agents(self) -> dict[str, dict[str, Any]]:
        character_names = self.config.character_names or DEFAULT_CHARACTER_NAMES
        if not character_names:
            raise RuntimeError("empty character name list for pedestrian auto sampling")

        agents: dict[str, dict[str, Any]] = {}
        for idx in range(max(0, int(self.config.count))):
            path = self.sample_path()
            start = path[0]
            goal = path[-1]
            yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
            velocity = self.rng.uniform(self.config.velocity_min, self.config.velocity_max)
            agent_name = f"agent_{idx}"
            agents[agent_name] = {
                "stage_prefix": f"{self.config.stage_prefix}{idx}",
                "character_name": self.rng.choice(character_names),
                "initial_pose": start,
                "goal_pose": goal,
                "path_points": path[1:],
                "loop_path": bool(self.config.loop_path),
                "orientation": float(yaw),
                "controller_stats": False,
                "velocity": float(velocity),
            }
        return agents


def dump_agents_yaml(path: str, agents: dict[str, dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(agents, f, sort_keys=False, allow_unicode=False)
