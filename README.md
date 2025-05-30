# Code Repository for ICML 2025 Paper  
**Title:** [On the Generalization Ability of Next-Token-Prediction Pretraining]  

---

## 🔧 1. Environment Setup  
### Installation    
```bash 
# Create and activate a virtual environment
conda create -n minintp python=3.10 
conda activate minintp

# Install torch
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install deepspeed
pip install deepspeed==0.16.8

# Install flash-atten
pip install flash-attn==2.7.1.post1 
# if erro, download flash_attn-2.7.1.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl first

# Install all dependencies
pip install -r requirements.txt
```  

## 🚀 2. Running Experiments  

```bash  
bash run_pretrain.sh
```  

<!-- ## 📖 4. Citation  
If you use this work, please cite our ICML 2025 paper:  
```bibtex  
@inproceedings{yourname2025title,  
  title    = {Your Paper Title Here},  
  author   = {Author1 and Author2 and Author3},  
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},  
  year     = {2025},  
  volume   = {XXX},  
  pages    = {XXXX--XXXX}  
}  
```   -->