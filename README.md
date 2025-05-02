# Self-learning code for Generative AI
Follow the lecture of Prof. Wu Yi on [BiliBili清华交叉信息院吴翼主讲-蚂蚁深度学习公开课](https://www.bilibili.com/video/BV1yFXWY1E7t/?share_source=copy_web&vd_source=992a5013db1a9bf76d72a0742386a522), I am leanring foundamentals of Generative AI by coding some important algorithms. 

During the learning, I use ChatGPT or DeepSeek to do the code refactoring. ChatGPT 4.1 is powerfull. 

```bash
git clone https://github.com/DrChiZhang/GenerativeAI.git
cd GenerativeAI 
mkdir checkpoints

conda create -n ml-learning python=3.10
conda activate ml-learning

#CUDA12.1 as default
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
python -c "import torch; print(torch.__version__)"

pip install -U matplotlib

# mkdir chpt
```c