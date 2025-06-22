# CatToy 项目说明文档

## 一、项目概述
CatToy 是一个涉及图像数据处理、三维重建、模型训练和渲染的项目。该项目包含多个模块，如数据加载、数据预处理、模型训练和渲染评估等。主要功能包括：
1. **数据加载**：支持多种数据集格式。
2. **数据预处理**：包括将视频转换为图像、运行 COLMAP 进行三维重建、将 COLMAP 结果转换为 NeRF 格式等。
3. **模型训练**：可以设置多种训练参数，如学习率、损失权重等。
4. **渲染评估**：对渲染结果进行评估，计算 PSNR、SSIM 等指标，并保存渲染结果和评估指标。

## 二、项目结构
项目主要包含以下文件和目录：
- `cattoy/`
  - `dataLoader/`：数据加载模块，包含不同数据集的加载器和数据预处理脚本。
    - `ray_utils.py`：读取 PFM 文件的工具函数。
    - `popmart.py`：数据集加载器。
    - `split_train_val.py`：将数据集划分为训练集和测试集。
    - `colmap2nerf.py`：将 COLMAP 结果转换为 NeRF 格式的 `transforms.json` 文件。
  - `log/`：日志和模型检查点存储目录。
  - `data/`：数据存储目录，包含图像、COLMAP 结果等。
  - `utils.py`：包含渲染对比可视化的工具函数。
  - `renderer.py`：渲染评估脚本，计算渲染结果的评估指标并保存渲染结果。
  - `opt.py`：配置参数解析脚本，用于设置训练和渲染的各种参数。

## 三、环境要求
### 依赖库
- `torch`
- `cv2`
- `json`
- `tqdm`
- `os`
- `PIL`
- `torchvision`
- `pycolmap`
- `imageio`
- `configargparse`
- `numpy`
- `math`
- `sys`
- `shutil`

### 安装依赖
可以使用以下命令安装所需的依赖库：pip install torch torchvision opencv-python tqdm pillow imageio configargparse numpy pycolmap
## 四、使用方法
### 1. 数据预处理
#### 视频转换为图像
如果有视频数据，可以使用 `colmap2nerf.py` 中的 `run_ffmpeg` 函数将视频转换为图像：python cattoy/dataLoader/colmap2nerf.py --video_in your_video.mp4 --video_fps 2 --images ./data/images
#### 运行 COLMAP 进行三维重建
使用 `colmap2nerf.py` 中的 `run_colmap` 函数运行 COLMAP：python cattoy/dataLoader/colmap2nerf.py --run_colmap --images ./data/images --colmap_db colmap.db --text colmap_text
#### 将 COLMAP 结果转换为 NeRF 格式
使用 `colmap2nerf.py` 将 COLMAP 的文本结果转换为 NeRF 格式的 `transforms.json` 文件：python cattoy/dataLoader/colmap2nerf.py --text colmap_text --out transforms.json
#### 划分训练集和测试集
使用 `split_train_val.py` 将数据集均匀划分为训练集和测试集：python cattoy/dataLoader/split_train_val.py --json_in transforms.json --out_dir splits --train_size 70
### 2. 模型训练
可以使用 `opt.py` 配置训练参数，并进行模型训练：python train.py --config config.txt其中 `config.txt` 是配置文件，包含训练所需的各种参数，如学习率、数据集路径等。

### 3. 渲染评估
使用 `renderer.py` 对渲染结果进行评估：python renderer.py --datadir ./data/images --split test --savePath ./log/render_results
## 五、配置参数说明
`opt.py` 主要参数说明：
- `--config`：配置文件路径。
- `--expname`：实验名称。
- `--basedir`：日志和检查点存储目录。
- `--datadir`：输入数据目录。
- `--dataset_name`：数据集名称，如 `popmart` 或 `blender`。
- `--batch_size`：批量大小。
- `--n_iters`：训练迭代次数。
- `--lr_init`：初始学习率。
- `--lr_decay_iters`：学习率衰减的迭代次数。
- `--lr_decay_target_ratio`：学习率衰减的目标比例。

## 六、注意事项
- 在运行 COLMAP 时，确保系统中已经安装了 COLMAP 软件。
- 配置文件中的参数可以根据实际需求进行调整，以获得更好的训练和渲染效果。
- 数据路径和文件名需要根据实际情况进行修改，确保程序能够正确读取和保存数据。
## 六、参考：git@github.com:apchenstu/TensoRF.git
    
