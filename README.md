This repo has 2 standalone components. One is the RGBIntoPointCloud.py demo, which takes an RGB-D image and turns into an animated point cloud.
The second component is a mock project that demonstrates the pipeline for an embodied AI agent similar to that of CAM-SLAM.
Steps:
1. Collect RGB-D from a photorealistic Habitat environment.
2. Convert depth into a 3D point cloud.
3. Use a stubbed SpatialLM module to extract a semantic target.
4. Navigate toward the target with a simple closed-loop controller.

## Quickstart
```bash
git clone https://github.com/nishia1/embodied-ai-demo
cd embodied-ai-demo
conda create -n embai python=3.10 -y
conda activate embai
pip install -r requirements.txt
python demo.py --scene data/scene_datasets/replica/apartment_0/apartment_0.glb

