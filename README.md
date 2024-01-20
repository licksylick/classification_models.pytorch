# Classification models
Project with Neural Networks for Image Classification based on PyTorch. 

-----
## 🔥 Advantages  
* High level API (just few lines to create a neural network)
* All models architectures from timm.
* All models have pre-trained weights for faster and better convergence

## 🚀 Train

### 1. Install all necessary libs:
  ```sh
  pip3 install -r requirements.txt
  ```
Install torch.
Note: if you are using a GPU, then you need to install CUDA and replace the torch version in `requirements.txt` with the GPU-enabled version.
Otherwise, the processor will be used.
-----
### 2. Dataset structure
Directory with 2 subdirectories: `tran_val` with the number of subdirectories equal to num classes and `test` with the number of subdirectories equal to num classes:  
 ~~~~
    dataset
     |- train_val
         |- class1
             |- image.jpg
             ...
         |- class2
             |- image.jpg
             ...
     |- test
        |- class1
             |- image.jpg
             ...
         |- class2
             |- image.jpg
             ...
  ~~~~

-----
### 3. Edit `config.yaml`
It has many of setting.  
The most important:
* `path` (in `dataset`) - set path to your data
* `num_classes` (in `model`) - set num_classes appropriate to the task
* `val_size` (in `dataset`) - percentage / 100 for split on train/val set
* `use_cross_validation` (in `common`) - bool value for use crossvalidation teqnique
* `max_epochs` (in `trainer`) - number of epochs to train
* `callbacks` - pytorch-lightning callbacks for your train

Other:
* `exp_name` - name of your experiment (for new experiments change this name) 
* `trainer` - in `params` you can add arguments for pytorch-lightning trainer
* `model` - in `params` you can change `arch` (supports all CNN models from timm)
* `optimizers` - you can change optimizer from torch.optim
* `scheduler` - you can change scheduler from torch.optim.lr_scheduler
-----

### 4. Run the training script specifying path to your config:

```sh
python3 train.py --config config.yaml
  ```

-----
-----
## ✅ Inference
Run `inference.py`, specifying the required config (which used for training), path to checkpoint and image:
  ```sh
  python3 inference.py --config config.yaml --model_path path/to/model.ckpt --image path/to/image.jpg
  ```
