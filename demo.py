"""
Demo that showcases:
 1. RGB-D capture from Habitat Replica scene
 2. Depth → point cloud
 3. Stubbed SpatialLM to return a mock semantic goal
 4. Simple rule-based navigation toward goal

Designed to run on a laptop with Replica sample scenes.
"""

import argparse
import math
import numpy as np
import habitat
from habitat.config.default import get_config

# --- Utilities ---

def depth_to_pointcloud(depth, rgb, fx, fy, cx, cy):
    H, W = depth.shape
    mask = depth > 0
    ys, xs = np.where(mask)
    z = depth[ys, xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    return pts

def quat_to_yaw(wxyz):
    w, x, y, z = wxyz
    siny_cosp = 2 * (w * y + z * x)
    cosy_cosp = 1 - 2 * (y * y + x * x)
    return math.atan2(siny_cosp, cosy_cosp)

# --- Stubbed SpatialLM ---
class SpatialLMClient:
    def infer_target(self, pcd, agent_pos):
        if pcd.shape[0] == 0:
            return agent_pos + np.array([1.0, 0, 0])
        centroid = np.median(pcd, axis=0)
        centroid[1] = 0.0
        return agent_pos + centroid

# --- Controller ---
class GoToPointController:
    def __init__(self, step_thresh=0.3, turn_thresh=0.2):
        self.step_thresh = step_thresh
        self.turn_thresh = turn_thresh

    def step(self, pos, yaw, goal):
        v = goal - pos
        v[1] = 0
        dist = np.linalg.norm(v)
        if dist < self.step_thresh:
            return "STOP"
        desired = math.atan2(v[2], v[0])
        dyaw = (desired - yaw + math.pi) % (2*math.pi) - math.pi
        if abs(dyaw) > self.turn_thresh:
            return "TURN_LEFT" if dyaw > 0 else "TURN_RIGHT"
        return "MOVE_FORWARD"

# --- Habitat setup ---
def make_env(scene_path):
    cfg = get_config("configs/tasks/pointnav.yaml")
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene_path
    cfg.SIMULATOR.RGB_SENSOR.WIDTH = 320
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = 240
    cfg.SIMULATOR.DEPTH_SENSOR.WIDTH = 320
    cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT = 240
    cfg.freeze()
    return habitat.Env(config=cfg)

# --- Main loop ---
def main(args):
    env = make_env(args.scene)
    obs = env.reset()
    ctrl = GoToPointController()
    spatiallm = SpatialLMClient()

    for step in range(args.max_steps):
        rgb = obs["rgb"]
        depth = obs["depth"][..., 0]
        H, W = depth.shape
        fx = fy = W / (2*math.tan(math.radians(45)))
        cx, cy = W/2, H/2
        pcd = depth_to_pointcloud(depth, rgb, fx, fy, cx, cy)

        state = env.sim.get_agent_state()
        pos = np.array(state.position)
        yaw = quat_to_yaw([state.rotation.w, state.rotation.x,
                           state.rotation.y, state.rotation.z])

        goal = spatiallm.infer_target(pcd, pos)
        action = ctrl.step(pos, yaw, goal)

        if action == "STOP":
            print("Reached goal in", step, "steps")
            break

        obs = env.step(action)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to Replica sample .glb scene")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    main(args)
