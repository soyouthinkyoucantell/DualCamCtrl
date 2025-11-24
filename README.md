## Installation

Clone repo and create an enviroment with Python 3.11:

```
git clone https://github.com/soyouthinkyoucantell/DualCamCtrl.git
conda create -n dualcamctrl python=3.11
conda activate dualcamctrl
```
Install DiffSynth-Studio dependencies from source code:

```
cd DualCamCtrl
pip install -e .
```

Then install GenFusion dependencies:
```
mkdir dependency
cd dependency
git clone https://github.com/rmbrualla/pycolmap.git 
cd pycolmap
pip install -e .
pip install numpy==1.26.4 peft accelerate==1.9.0 decord==0.6.0 deepspeed diffusers omegaconf  
```


## Inference

Test with our demo pictures and depth:
```
cd ../.. # make sure you are at the root dir 
export PYTHONPATH=.
python -m test_script.test_demo
```
