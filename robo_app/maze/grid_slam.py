"""Grid-constrained Rao-Blackwellized particle SLAM for the maze robot.

Each particle carries a candidate cell and its own wall map. Motion creates
pose alternatives; repeat ToF scans score those alternatives against their maps.
The winning map is still exported in the maze_shooter JSON format.
"""

import copy
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .mapping import MazeMap
from .settings import Cell, CELL_M

Scan = Dict[int, Optional[float]]  # absolute direction -> ToF millimeters
MAX_USABLE_RANGE_MM = 4000


@dataclass
class Particle:
    cell: Cell
    heading: int
    maze: MazeMap
    evidence: Dict[Tuple[Cell, Cell], int]


class GridSLAM:
    def __init__(self, maze: MazeMap, start: Cell, heading: int, count: int = 96,
                 seed: Optional[int] = None, wall_offset_m: float = 0.25):
        if count < 8:
            raise ValueError("At least eight particles are required")
        self.rng = random.Random(seed)
        evidence = {key: 4 if status == 1 else -4 for key, status in maze.edges.items()}
        self.particles = [Particle(start, heading, copy.deepcopy(maze), evidence.copy())
                          for _ in range(count)]
        self.cell, self.heading = start, heading
        self.confidence = 1.0
        self.wall_offset_m = wall_offset_m

    @property
    def maze(self) -> MazeMap:
        return self.particles[0].maze

    @staticmethod
    def ray(cell: Cell, direction: int, millimeters: Optional[float], maze: MazeMap,
            wall_offset_m: float = 0.25
            ) -> List[Tuple[Cell, Cell, int]]:
        """Decode the wall at offset + n*CELL_M meters; include open edges before it."""
        if millimeters is None or millimeters < 20:
            return []
        wall_seen = millimeters <= MAX_USABLE_RANGE_MM
        open_count = max(0, round((millimeters / 1000 - wall_offset_m) / CELL_M))
        if not wall_seen:
            open_count = min(open_count, MAX_USABLE_RANGE_MM // int(CELL_M * 1000))
        edges = []
        current = cell
        for _ in range(open_count):
            nxt = maze.step(current, direction)
            if not maze.inside(nxt):
                return edges  # map boundary is already known
            edges.append((current, nxt, 1))
            current = nxt
        if wall_seen:
            edges.append((current, maze.step(current, direction), -1))
        return edges

    def predict_move(self, direction: int):
        """A one-cell command can undershoot or overshoot; keep those hypotheses."""
        for index, particle in enumerate(self.particles):
            draw = self.rng.random()
            steps = (1 if index == 0 else 0 if index == 1 else 2 if index == 2
                     else 1 if draw < 0.94 else 0 if draw < 0.99 else 2)
            particle.heading = direction
            for _ in range(steps):
                nxt = particle.maze.step(particle.cell, direction)
                if not particle.maze.can_drive(particle.cell, nxt):
                    break
                particle.cell = nxt

    def calibrate_from_boundary(self, scan: Scan):
        """At startup, use arena boundary readings to estimate the ToF mounting offset."""
        measurements = [distance / 1000 for direction, distance in scan.items()
                        if distance is not None and 80 <= distance <= 450
                        and not self.maze.inside(self.maze.step(self.cell, direction))]
        if measurements:
            self.wall_offset_m = statistics.median(measurements)

    def _log_likelihood(self, particle: Particle, scan: Scan) -> float:
        score = 0.0
        for direction, millimeters in scan.items():
            if millimeters is not None and 20 <= millimeters <= MAX_USABLE_RANGE_MM:
                open_count = max(0, round((millimeters / 1000 - self.wall_offset_m) / CELL_M))
                end = particle.cell
                for _ in range(open_count):
                    end = particle.maze.step(end, direction)
                    if not particle.maze.inside(end):
                        # Long ToF rays occasionally miss a wall or pass through a gap.
                        # One such ray must not outweigh the other three directions
                        # and the motion prediction by itself.
                        outlier_probability = 0.03 if millimeters > 800 else 0.001
                        score += math.log(outlier_probability)
                        break
            for a, b, observed in self.ray(particle.cell, direction, millimeters,
                                           particle.maze, self.wall_offset_m):
                known = particle.maze.edge(a, b)
                if known:
                    score += math.log(0.96 if known == observed else 0.01)
        return score

    def _apply_scan(self, particle: Particle, scan: Scan):
        particle.maze.visited.add(particle.cell)
        for direction, millimeters in scan.items():
            for a, b, observed in self.ray(particle.cell, direction, millimeters,
                                           particle.maze, self.wall_offset_m):
                key = MazeMap._key(a, b)
                old = particle.evidence.get(key, 0)
                particle.evidence[key] = max(-8, min(8, old + (2 if observed == 1 else -3)))
                strength = particle.evidence[key]
                if strength >= 1:
                    particle.maze.set_edge(a, b, 1)
                elif strength <= -1:
                    particle.maze.set_edge(a, b, -1)
                else:
                    particle.maze.edges.pop(key, None)

    def observe(self, scan: Scan, heading: int) -> float:
        """Correct pose from a scan, then update each particle's map and resample."""
        for particle in self.particles:
            particle.heading = heading
        logs = [self._log_likelihood(particle, scan) for particle in self.particles]
        peak = max(logs)
        weights = [math.exp(value - peak) for value in logs]
        total = sum(weights)
        weights = [weight / total for weight in weights]
        pose_mass = defaultdict(float)
        for particle, weight in zip(self.particles, weights):
            pose_mass[particle.cell] += weight
        pose = max(pose_mass, key=pose_mass.get)
        self.confidence = pose_mass[pose]
        representative = max((i for i, p in enumerate(self.particles) if p.cell == pose),
                             key=lambda i: weights[i])
        for particle in self.particles:
            self._apply_scan(particle, scan)

        # Systematic resampling with an explicit representative for the exported map.
        cumulative = []
        running = 0.0
        for weight in weights:
            running += weight
            cumulative.append(running)
        selected = [copy.deepcopy(self.particles[representative])]
        offset = self.rng.random() / len(self.particles)
        index = 0
        for slot in range(1, len(self.particles)):
            point = offset + slot / len(self.particles)
            while index < len(cumulative) - 1 and cumulative[index] < point:
                index += 1
            selected.append(copy.deepcopy(self.particles[index]))
        self.particles = selected
        self.cell, self.heading = pose, heading
        return self.confidence

    def record_wall(self, direction: int):
        """Confirmed bumper stop updates each particle's map at its own pose."""
        for particle in self.particles:
            neighbor = particle.maze.step(particle.cell, direction)
            particle.maze.set_edge(particle.cell, neighbor, -1)
            particle.evidence[MazeMap._key(particle.cell, neighbor)] = -8
