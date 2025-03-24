# Virtual_Try_On


If you find this project helpful, please consider starring ⭐ the repository!

## Teaser
![teaser](https://github.com/user-attachments/assets/4366c91a-92e6-4591-99ba-bb23fc1f46f8)

![teaser2](https://github.com/user-attachments/assets/e2c713c5-37ff-48cd-b0bf-5208ea077df9)


---

## Requirements
### Installation
Clone the repository and set up the environment:
```bash
git clone https://github.com/yisol/IDM-VTON.git
cd IDM-VTON

conda env create -f environment.yaml
conda activate idm
```

---

## Data Preparation
### VITON-HD Dataset
You can download the **VITON-HD** dataset [here](https://github.com/shadow2496/VITON-HD).

After downloading, move the JSON files into the respective dataset folders:
- `vitonhd_test_tagged.json` → `test` folder
- `vitonhd_train_tagged.json` → `train` folder

The dataset directory should be structured as follows:
```
train
|-- image
|-- image-densepose
|-- agnostic-mask
|-- cloth
|-- vitonhd_train_tagged.json

test
|-- image
|-- image-densepose
|-- agnostic-mask
|-- cloth
|-- vitonhd_test_tagged.json
```

### DressCode Dataset
You can download the **DressCode** dataset [here](https://github.com/aimagelab/dresscode).

We provide pre-computed **densepose images** and **captions** for garments [here](#).

For obtaining densepose images, we used **Detectron2**. You can refer [here](https://github.com/facebookresearch/detectron2) for more details.

Organize the dataset as follows:
```
DressCode
|-- dresses
    |-- images
    |-- image-densepose
    |-- dc_caption.txt
|-- lower_body
    |-- images
    |-- image-densepose
    |-- dc_caption.txt
|-- upper_body
    |-- images
    |-- image-densepose
    |-- dc_caption.txt
```

---

## Training
### Preparation
Download the pre-trained **IP-Adapter for SDXL** and **image encoder** [here](https://huggingface.co/h94/IP-Adapter):

Move the models to:
```
ckpt/ip_adapter
ckpt/image_encoder
```

### Start Training
Run the following command:
```bash
accelerate launch train_xl.py \
    --gradient_checkpointing --use_8bit_adam \
    --output_dir=result --train_batch_size=6 \
    --data_dir=DATA_DIR
```
Or use the script:
```bash
sh train_xl.sh
```

---

## Inference
### VITON-HD Inference
```bash
accelerate launch inference.py \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" \
    --unpaired \
    --data_dir "DATA_DIR" \
    --seed 42 \
    --test_batch_size 2 \
    --guidance_scale 2.0
```
Or use:
```bash
sh inference.sh
```

### DressCode Inference
For DressCode dataset, specify the category:
```bash
accelerate launch inference_dc.py \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" \
    --unpaired \
    --data_dir "DATA_DIR" \
    --seed 42 \
    --test_batch_size 2 \
    --guidance_scale 2.0 \
    --category "upper_body"
```
Or use:
```bash
sh inference.sh
```

---

## Local Gradio Demo
### Checkpoints Download
Download checkpoints for **human parsing** [here](#).

Place them under the `ckpt` folder:
```
ckpt
|-- densepose
    |-- model_final_162be9.pkl
|-- humanparsing
    |-- parsing_atr.onnx
    |-- parsing_lip.onnx
|-- openpose
    |-- ckpts
        |-- body_pose_model.pth
```
### Run the Gradio Demo
```bash
python gradio_demo/app.py
```

---

## Acknowledgements
- **ZeroGPU** for free GPU support.
- **IP-Adapter** for base codes.
- **OOTDiffusion** & **DCI-VTON** for masking generation.
- **SCHP** for human segmentation.
- **Densepose** for human densepose processing.

---

## Citation
If you use this work, please cite:
```bibtex
@article{choi2024improving,
  title={Improving Diffusion Models for Authentic Virtual Try-on in the Wild},
  author={Choi, Yisol and Kwak, Sangkyung and Lee, Kyungmin and Choi, Hyungwon and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2403.05139},
  year={2024}
}
```

---

## License
The code and models in this repository are licensed under **CC BY-NC-SA 4.0**.
