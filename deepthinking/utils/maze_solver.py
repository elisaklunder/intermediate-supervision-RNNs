import os
import sys

project_root = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np

Position = Tuple[int, int]  # (row, col) on the patch grid
RGB = np.ndarray  # shape (3, H, W)


class MazeSolver:
    """
    Maze solver using different algorithms to find the optimal path in a maze.
    The maze is a grid of rgb patches (cells) where one patch = 2x2 pixels
    - Black (0, 0, 0) = Wall
    - White (1, 1, 1) = Free space
    - Red   (1, 0, 0) = Start patch (all 4 pixels red)
    - Green (0, 1, 0) = Goal  patch (all 4 pixels green)
    """

    def __init__(self, mode: Optional[str] = "incremental"):
        self.mode = mode

    def _crop_outer_wall(self, rgb: RGB) -> Tuple[RGB, Tuple[int, int]]:
        """Crop outermost ring of pixels and return the offset."""
        return (rgb[:, 1:-1, 1:-1], (1, 1))

    def _view_as_patches(self, arr: np.ndarray) -> np.ndarray:
        """Return a view shaped (H2, 2, W2, 2) where H2 = H // 2 and W2 = W // 2."""
        h, w = arr.shape
        if (h | w) & 1:
            raise ValueError("Image dimensions must be multiples of 2.")
        return arr.reshape(h // 2, 2, w // 2, 2)

    def _parse_maze(self, rgb: RGB) -> Tuple[np.ndarray, Position, Position]:
        """Return free-mask, start-patch, goal-patch on the patch grid."""
        r, g, b = rgb
        pr, pg, pb = map(self._view_as_patches, (r, g, b))

        free = np.all((pr + pg + pb) != 0, axis=(1, 3))
        start = np.all((pr == 1) & (pg == 0) & (pb == 0), axis=(1, 3))
        goal = np.all((pr == 0) & (pg == 1) & (pb == 0), axis=(1, 3))

        s_idx, g_idx = map(np.argwhere, (start, goal))
        if s_idx.size == 0 or g_idx.size == 0:
            raise ValueError("Start or goal patch not found.")
        return free, tuple(s_idx[0]), tuple(g_idx[0])

    def _bfs_trace(
        self, free: np.ndarray, start: Position, goal: Position
    ) -> Tuple[List[Position], List[Position]]:
        """Perform BFS and return (optimal_path, discovery_sequence)."""
        h2, w2 = free.shape
        visited = np.zeros_like(free, bool)
        parent = {}

        q = deque([start])
        visited[start] = True
        discovered: List[Position] = [start]

        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                break
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < h2
                    and 0 <= nc < w2
                    and free[nr, nc]
                    and not visited[nr, nc]
                ):
                    visited[nr, nc] = True
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))
                    discovered.append((nr, nc))

        if goal not in parent and start != goal:
            return [], discovered
        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return path, discovered

    def _patch_to_pixels(
        self, pr: int, pc: int, offs: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Convert patch coordinates to pixel coordinates."""
        return 2 * pr + offs[0], 2 * pc + offs[1]

    def _toggle_patch(
        self, mask: np.ndarray, patch: Position, offs: Tuple[int, int], val: int
    ) -> None:
        """Set the pixels of a patch to a given value."""
        r0, c0 = self._patch_to_pixels(*patch, offs)
        mask[r0 : r0 + 2, c0 : c0 + 2] = val

    def _mask_from_patches(
        self, patches: Sequence[Position], shape: Tuple[int, int], offs: Tuple[int, int]
    ) -> np.ndarray:
        """Create a mask with originall maze shape from a list of patches."""
        m = np.zeros(shape, np.uint8)
        for p in patches:
            self._toggle_patch(m, p, offs, 1)
        return m

    def _mask_from_patches_with_values(
        self,
        patches_values: List[Tuple[Position, int]],
        shape: Tuple[int, int],
        offs: Tuple[int, int],
    ) -> np.ndarray:
        """Create a mask with different values for different patch types."""
        m = np.zeros(shape, np.uint8)
        for patch, value in patches_values:
            self._toggle_patch(m, patch, offs, value)
        return m

    def _frames_remove(
        self,
        sequence: Sequence[Position],
        offs: Tuple[int, int],
        start_mask: np.ndarray,
        step: int = 1,
    ) -> List[np.ndarray]:
        """Yield frames while clearing patches from start_mask."""
        m = start_mask.copy()
        frames = [m.copy()]
        for i, p in enumerate(sequence):
            self._toggle_patch(m, p, offs, 0)
            if (i + 1) % step == 0:  # append frame every step frames
                frames.append(m.copy())
        return frames + [m.copy()]  # + final state

    @staticmethod
    def _manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def get_dfs_masks(self, input_rgb, step=1) -> List[np.ndarray]:
        """
        DFS with backtracking
        """

        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        H, W = input_rgb.shape[1:]

        h2, w2 = free.shape
        visited = np.zeros_like(free, bool)
        current_path = []
        frames = []
        frame_counter = 0
        found = False

        def dfs(patch: Position):
            nonlocal found, frame_counter

            if found:
                return True

            visited[patch] = True
            current_path.append(patch)

            frame_counter += 1
            if frame_counter % step == 0:
                frame = self._mask_from_patches(current_path, (H, W), offs)
                frames.append(frame)

            if patch == goal:
                found = True
                return True

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = patch[0] + dr, patch[1] + dc
                neighbor = (nr, nc)

                if (
                    (0 <= nr < h2)
                    and (0 <= nc < w2)
                    and free[nr, nc]
                    and not visited[nr, nc]
                ):
                    if dfs(neighbor):
                        return True

            current_path.pop()

            frame_counter += 1
            if frame_counter % step == 0:
                frame = self._mask_from_patches(current_path, (H, W), offs)
                frames.append(frame)

            return False

        dfs(start)

        if found:
            solution_visited = np.zeros_like(free, bool)
            solution_path = []

            def find_solution_path(patch: Position, path: List[Position]):
                solution_visited[patch] = True
                path.append(patch)

                if patch == goal:
                    return True

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = patch[0] + dr, patch[1] + dc
                    neighbor = (nr, nc)

                    if (
                        (0 <= nr < h2)
                        and (0 <= nc < w2)
                        and free[nr, nc]
                        and not solution_visited[nr, nc]
                    ):
                        if find_solution_path(neighbor, path):
                            return True

                path.pop()
                solution_visited[patch] = False
                return False

            find_solution_path(start, solution_path)

            final_frame = self._mask_from_patches(solution_path, (H, W), offs)

            for _ in range(10):
                frames.append(final_frame)

        return frames

    def get_incremental_path_masks(self, input_rgb, step) -> List[np.ndarray]:
        """Incrementally light up the optimal path."""
        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        path, _ = self._bfs_trace(free, start, goal)

        if not path:
            return []

        H, W = input_rgb.shape[1:]
        steps = [
            self._mask_from_patches(path[:i], (H, W), offs)
            for i in range(1, len(path) + 1, step)
        ]
        for i in range(10):
            steps.append(self._mask_from_patches(path, (H, W), offs))

        return steps

    def get_incremental_path_masks_bidirectional(
        self, input_rgb, step
    ) -> List[np.ndarray]:
        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        path, _ = self._bfs_trace(free, start, goal)

        if not path:
            return []

        H, W = input_rgb.shape[1:]
        steps = []
        path_len = len(path)

        for i in range(0, path_len // 2 + 1, step):
            current_path = path[:i] + path[-(i if i > 0 else 0) :]
            steps.append(self._mask_from_patches(current_path, (H, W), offs))

        for _ in range(1, 10):
            steps.append(self._mask_from_patches(path, (H, W), offs))

        return steps

    def get_reverse_exploration_masks(self, input_rgb, step) -> List[np.ndarray]:
        """Start with all explored, then peel off dead ends."""
        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        path, discovered = self._bfs_trace(free, start, goal)

        H, W = input_rgb.shape[1:]
        full = self._mask_from_patches(discovered, (H, W), offs)

        dead_ends = [p for p in reversed(discovered) if p not in set(path)]
        frames = self._frames_remove(dead_ends, offs, full, step=step)

        final = self._mask_from_patches(path, (H, W), offs)
        for _ in range(10):
            frames.append(final)

        return frames

    def get_bfs_masks(self, input_rgb, step) -> List[np.ndarray]:
        """Return masks of the BFS discovery sequence."""
        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        optimal_path, discovered = self._bfs_trace(free, start, goal)

        H, W = input_rgb.shape[1:]
        discovery_masks = [
            self._mask_from_patches(discovered[:i], (H, W), offs)
            for i in range(1, len(discovered) + 1)
        ]
        discovery_masks.append(self._mask_from_patches(optimal_path, (H, W), offs))
        return discovery_masks

    def get_intermediate_supervision_masks(self, input_rgb, step=1) -> List[np.ndarray]:
        if self.mode == "incremental":
            return self.get_incremental_path_masks(input_rgb, step=step)
        elif self.mode == "reverse":
            return self.get_reverse_exploration_masks(input_rgb, step=step)
        elif self.mode == "bidirectional":
            return self.get_incremental_path_masks_bidirectional(input_rgb, step=step)
        elif self.mode == "dfs":
            return self.get_dfs_masks(input_rgb, step=step)
