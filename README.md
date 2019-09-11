# Spatio-Temporal Modeling

Pytorch implementation for the paper "Comparative Analysis of CNN-based Spatiotemporal Reasoning in Videos".

**Maintainers:** [Okan Köpüklü](https://github.com/okankop) and [Fabian Herzog](https://github.com/fubel)

The structure was inspired by the project [TRN-pytorch](https://github.com/metalbubble/TRN-pytorch)

## Results and Pretrained Models

The pretrained models can be found in our [Google Drive](https://drive.google.com/drive/folders/13x6ClKowbfPLf4RgA7ITt4mVEqtReqWI?usp=sharing). 

## Setup

### Installing Requirements

Clone the repo with the following command:
```bash
git clone git@github.com:fubel/stmodeling.git
```

The project requirements can be found in the file `requirements.txt`. To run the code, create a Python >= 3.6 virtual environment and install 
the requirements with 

```
pip install -r requirements.txt
```

**NOTE:** This project assumes that you have a GPU with CUDA support. 


### Dataset Preparation
Download the [jester dataset](https://20bn.com/datasets/jester) or [something-something-v2 dataset](https://20bn.com/datasets/something-something/v2). Decompress them into the same folder and use [process_dataset.py](process_dataset.py) to generate the index files for train, val, and test split. Poperly set up the train, validatin, and category meta files in [datasets_video.py](datasets_video.py).
To convert the something-something-v2 dataset, you can use the ``extract_frames.py`` from [TRN-pytorch](https://github.com/metalbubble/TRN-pytorch/blob/master/extract_frames.py).


Assume the structure of data directories is the following:

```misc
~/stmodeling/
   datasets/
      jester/
         rgb/
            .../ (directories of video samples for Jester)
                .../ (jpg color frames)
      something/
         rgb/    
            .../ (directories of video samples for Something-Something)
    model/
       .../(saved models for the last checkpoint and best model)
```



### Running the Code
Followings are some examples for training under different scenarios:

* Train 8-segment network for Jester with MLP and squeeznet backbone 
```bash
python main.py jester RGB --arch squeezenet1_1 --num_segments 8 \
--consensus_type MLP --batch-size 16
```

* Train 16-segment network for Something-Something with TRN-multiscale and BNInception backbone
```bash
python main.py something RGB --arch BNInception --num_segments 16 \ 
--consensus_type TRNmultiscale --batch-size 16
```

## Reference

## Acknowledgement 

This project was build on top of [TRN-pytorch](https://github.com/metalbubble/TRN-pytorch), which itself was build on top of [TSN-Pytorch](https://github.com/yjxiong/temporal-segment-networks). We thank the authors for sharing their code publicly.
