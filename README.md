🌐 3D Point Cloud Segmentation 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🌍 This 3D Point Cloud Segmentation Python application leverages the Segment Anything Model (SAM) to process and segment 3D point clouds. It transforms point cloud data into a spherical projection, applies SAM-based segmentation to generate masks, and maps the segmented colors back to the 3D point cloud, producing a visually rich .las file for visualization in tools like CloudCompare.

This code is useful when you have a point cloud that has no data annotation and you don't have to go through the hassle of annotating a point cloud. The code performs a zero shot segmentation

---
![pointcloud](segmented_pointcloud.gif)
--
## ✨ Features

📝 Input Processing: Load .las point cloud files (e.g., data/unreal.las) and configure spherical projection resolution.
🌐 Spherical Projection: Transform 3D point clouds into 2D spherical images for segmentation.
🧠 SAM Segmentation: Utilize the Segment Anything Model to generate high-quality masks on 2D projections.
🎨 Color Mapping: Apply segmented colors back to the 3D point cloud for enhanced visualization.
🖼️ Visualization: Generate static 2D images of spherical projections and segmented results.
💾 Output Export: Save segmented point clouds as .las files, compatible with tools like CloudCompare.
🐳 Docker Support: Deploy the application in a Docker container with CUDA 11.8 for GPU-accelerated processing.

---
## 📁 File Structure
```text
3d_point_cloud_segmentation/
├── .gitignore                    # 🙈 Git ignore file
├── README.md                     # 📖 This project documentation
├── requirements.txt              # 📜 Python dependencies
├── Dockerfile                    # 🐳 Docker configuration for GPU support
├── data/
│   └── unreal.las                # 📉 Input point cloud file
├── modules/
│   ├── __init__.py               # 📦 Makes modules a Python package
│   ├── data_processing.py        # 🌐 Point cloud loading and transformation
│   ├── sam.py                    # 🧠 SAM model initialization and segmentation
│   └── visualize.py              # 📈 Visualization functions
├── main.py                       # 🎈 Main pipeline orchestration
├── weights/
│   └── sam_vit_h_4b8939.pth      # 🧠 SAM model weights
└── result/                       # 📂 Auto-generated output directory
```
---
## 🛠️ Prerequisites

🐍 Python 3.9+: Required for local setup.
🐳 Docker: For containerized deployment (with NVIDIA Container Toolkit for GPU support).
💻 GPU (Optional): NVIDIA GPU with CUDA 11.8 for accelerated SAM processing.
📊 CloudCompare: For visualizing output .las files (optional).
🐍Anaconda: For managing python environment.

---
## ⚙️ Local Setup

**Clone the Repository:**
```bash
git clone https://github.com/Danny024/3d_point_cloud_segmentation.git
cd 3d_point_cloud_segmentation
```


**Set up PYTHONPATH (Important!):**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**Create a Virtual Environment:**
```bash
conda create -n venv python=3.9
conda activate venv
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```


**Download SAM Model Weights:**

* The weights are automatically downloaded in the Docker setup. For local setup, download manually:

```bash
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights
```




**Prepare Input Data:**
`
* Place your .las point cloud file (e.g., unreal.las) in the data/ directory.
* Ensure the file contains point coordinates (x, y, z) and color information (red, green, blue).

**Run the Application:**
```bash
python3 main.py
```

---

## 🐳 Docker Setup


Install Docker and NVIDIA Container Toolkit:
```bash
* **Docker (Ubuntu):**
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

* **NVIDIA Container Toolkit (for GPU support):**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```


**Build Docker Image:**
```bash
docker build -t 3d-point-cloud-segmentation .
```

Run Docker Container:
With GPU:
```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/result:/app/result 3d-point-cloud-segmentation
```

CPU-only:
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/result:/app/result 3d-point-cloud-segmentation
```

---

## 🚀 Usage

**Prepare Input:**

* Place your .las point cloud file in data/ (default: unreal.las).
* Ensure weights/sam_vit_h_4b8939.pth is available (auto-downloaded in Docker).


* **Run the Pipeline:**

* Execute python3 main.py locally or use the Docker command above.
The pipeline will:
* Load the point cloud from data/unreal.las.
* Generate a spherical projection (result/sphere_projection.jpg).
* Apply SAM segmentation (result/2d_segmented_image.jpg).
* Color and export the segmented point cloud (result/3d_segmened_point_cloud.las).



* **Visualize Outputs:**

* View sphere_projection.jpg and 2d_segmented_image.jpg in result/ using any image viewer.
* Open 3d_segmened_point_cloud.las in CloudCompare for 3D visualization.


---

## 🤔 Troubleshooting

**ModuleNotFoundError:**

* Ensure dependencies are installed: pip install -r requirements.txt.
* Verify PYTHONPATH includes the project directory

```bash
 export PYTHONPATH=$PYTHONPATH:$(pwd)
 ```
* Check modules/__init__.py exists.


**FileNotFoundError:**

* Confirm data/unreal.las and weights/sam_vit_h_4b8939.pth exist.
* Download weights if missing: 

```bash
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights.
```


**Docker Build Fails:**

* Ensure requirements.txt, main.py, and modules/ are present.
* Verify internet access for downloading weights and packages.


**Docker Container Fails to Run:**

* Check logs: docker logs <container_id_or_name>.
* Port Conflicts: Not typically an issue, as no ports are exposed, but check for other Docker issues.
* GPU Issues: Test GPU setup: docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi.


**Output Issues:**

* Ensure result/ has write permissions: chmod -R u+rw result.
* Verify output files in result/ after running.


---

## 📌 Further Work

* Support additional point cloud formats (e.g., .ply, .pcd).
* Implement interactive visualization for spherical projections.
* Optimize SAM processing for larger point clouds.
* Add configuration file for pipeline parameters.

---

## 📜 License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

---

## 🙏 Acknowledgments

* Segment Anything Model (SAM): Developed by Meta AI Research.
* CloudCompare: For visualizing .las point cloud outputs.


