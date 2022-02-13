# Falcon-Release
PyTorch implementation of FALCON: Fast Visual Concept Learning by Integrating Images, Linguistic descriptions, and Conceptual Relations

![](assets/teaser.png)

FALCON: Fast Visual Concept Learning by Integrating Images, Linguistic descriptions, and Conceptual Relations

Lingjie Mei, Jiayuan Mao, Ziqi Wang, Chuang Gan, Joshua B. Tenenbaum

ICLR 2022

[Paper](https://openreview.net/pdf?id=htWIlvDcY8) [Website](https://people.csail.mit.edu/jerrymei/projects/falcon/) 

![](assets/model.png)
## Getting started
### Prerequisites
+ Linux
+ Python3
+ PyTorch 1.6 with CUDA support
+ Other required python packages specified by `requirements.txt`.
### Installation
1. Clone this repository

    ```bash
    git clone https://github.com/JerryLingjieMei/FALCON-Release
    cd FALCON-Release
    ```

1. Create a conda environment for FALCON Model and install the requirements. 
    
    ```bash
    conda create --n falcon-model
    conda activate falcon-model
    pip install -r requirements.txt
    conda install pytorch=1.6.0 cuda100 -c pytorch #Assume you use cuda version 10.0
    ```
 
1. Change `DATASET_ROOT` in `tools.dataset_catalog` to the folder where the datasets are stored. 
    Download and unpack the base CUB, CLEVR and GQA datasets into 
    `DATASET/CUB-200-2011`,  `DATASET/CLEVR_v1.0`,  `DATASET/GQA`,  respectively. 
    Download datasets for fast concept learning. 

    ```bash
    . scripts/download_cub_data.sh ${DATASET_ROOT}
    . scripts/download_clevr_data.sh ${DATASET_ROOT}
    . scripts/download_gqa_data.sh ${DATASET_ROOT}
    ```

1. Download our weights for FALCON-G model.

    ```bash
    . scripts/download_cub_model.sh
    . scripts/download_clevr_model.sh
    . scripts/download_gqa_model.sh
    ```


### Experiments(Final Testing)

1. Run the fast concept learning experiments via the config file `cub/cub_fewshot_graphical_box.yaml`, 
    `clevr/clevr_fewshot_graphical_0.yaml` 
    or `gqa/gqa_fewshot_graphical_box.yaml`. 
    
    ```bash
    export NAME=cub/cub_fewshot_graphical_box; python tools/test_net.py --config-file experiments/${NAME}.yaml
    export NAME=clevr/clevr_fewshot_graphical_0; python tools/test_net.py --config-file experiments/${NAME}.yaml
    export NAME=gqa/gqa_fewshot_graphical_box; python tools/test_net.py --config-file experiments/${NAME}.yaml
    ```
   

### Experiments(Training)

1. Here we use the CUB dataset as an example. Uncomment in `scripts/download_cub_data.sh` 
    and `scripts/download_cub_data.sh`. Re-run them
    
    ```bash
    . scripts/download_cub_data.sh ${DATASET_ROOT}
    . scripts/download_cub_model.sh
    ```
   
2. Train optionally and test on the parser.
    
    ```bash
    export NAME=cub/cub_fewshot_build; python tools/train_net.py --config-file experiments/${NAME}.yaml
    export NAME=cub/cub_fewshot_build; python tools/test_net.py --config-file experiments/${NAME}.yaml
    ```
   
3. Train optionally the concept embeddings and feature extractor from the training concepts.
    
    ```bash
    export NAME=cub/cub_support_box; python tools/train_net.py --config-file experiments/${NAME}.yaml
    ```

4. Train optionally the fast concept learning models, e.g. FALCON-G.
    ```bash
    export NAME=cub/cub_fewshot_graphical_box; python tools/train_net.py --config-file experiments/${NAME}.yaml
    export NAME=cub/cub_fewshot_graphical_box; python tools/test_net.py --config-file experiments/${NAME}.yaml
   ```
 
### Experiments (Additional)

1. Additional experiments can be configured by specifying:
    + `TEMPLATE` to represent the training stages, base datasets and embedding spaces. 
    + `MODEL.NAME` to represent the type of fast concept learning models. 
    + `DATASETS` to represent the datasets in the evaluations. 
    
2. For other experiments, please fill free to contact the author via email or GitHub.
