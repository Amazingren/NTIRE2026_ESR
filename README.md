# [NTIRE 2026 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

<div align=center>
<img src="https://github.com/Amazingren/NTIRE2026_ESR/blob/main/figs/logo.png" width="400px"/> 
</div>

## News
- :t-rex: February 6th, 2026: Our Challenge Repo. is ready!


## About the Challenge

In collaboration with the NTIRE workshop, we are hosting a challenge focused on Efficient Super-Resolution ([NTIRE2026_ESR](https://www.codabench.org/competitions/13553/)). This involves the task of enhancing the resolution of an input image by a factor of x4, utilizing a set of pre-existing examples comprising both low-resolution and their corresponding high-resolution images. The challenge encompasses one :trophy: main track which consists of three :gem: sub-tracks, i.e., the Inference Runtime, FLOPs (Floating Point Operations Per Second), and Parameters. The baseline method in NTIRE2026_ESR is [SPAN](https://arxiv.org/abs/2311.12770) (*Cheng Yan, 2024*), the 1st place for the overall performance of NTIRE2024 Efficient Super-Resolution Challenge. Details are shown below:

- :trophy: Main-track: **Overall Performance** (Runtime, Parameters, FLOPs,) the aim is to obtain a network design / solution with the best overall performance in terms of inference runtime, FLOPS, and parameters on a common GPU (e.g., NVIDIA RTX A6000 GPU) while being constrained to maintain or improve the PSNR results.

- :gem: Sub-track 1: **Inference Runtime**, the aim is to obtain a network design / solution with the lowest inference time (runtime) on a common GPU (e.g., NVIDIA RTX A6000 GPU) while being constrained to maintain or improve over the baseline method SPAN in terms of number of parameters, FLOPs, and the PSNR result.

- :gem: Sub-track 2: **FLOPs**, the aim is to obtain a network design / solution with the lowest amount of FLOPs on a common GPU (e.g., NVIDIA RTX A6000 GPU) while being constrained to maintain or improve the inference runtime, the parameters, and the PSNR results of SPAN.

- :gem: Sub-track 3: **Parameters**, the aim is to obtain a network design / solution with the lowest amount of parameters on a common GPU (e.g., NVIDIA RTX A6000 GPU) while being constrained to maintain the FLOPs, the inference time (runtime), and the PSNR result of SPAN.

It's important to highlight that to determine the final ranking and challenge winners, greater weight will be given to teams or participants who demonstrate improvements in more than one aspect (runtime, FLOPs, and parameters) over the provided reference solution.

🌟 **New This Year:**

Participants are encouraged to explore model compression and acceleration techniques, such as quantization and pruning, to further reduce inference runtime and overall computational cost. Submissions that achieve efficiency gains without sacrificing reconstruction quality will be viewed favorably in the final evaluation.

To ensure fairness in the evaluation process, it is imperative to adhere to the following guidelines:
- **Avoid Training with Specific Image Sets:**
    Refrain from training your model using the validation LR images, validation HR images, or testing LR images. The test datasets will not be disclosed, making PSNR performance on the test datasets a crucial factor in the final evaluation.

- **PSNR Threshold and Ranking Eligibility:**
    Methods with a PSNR below the specified threshold (i.e., 26.90 dB on DIV2K_LSDIR_valid and, 26.99 dB on DIV2K_LSDIR_test) will not be considered for the subsequent ranking process. It is essential to meet the minimum PSNR requirement to be eligible for further evaluation and ranking.


## The Environments

The evaluation environments adopted by us is recorded in the `requirements.txt`. After you built your own basic Python (Python = 3.9 in our setting) setup via either *virtual environment* or *anaconda*, please try to keep similar to it via:

- Step1: install Pytorch first:
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`

- Step2: install other libs via:
```pip install -r requirements.txt```

or take it as a reference based on your original environments.

## The Validation datasets
After downloaded all the necessary validate dataset ([DIV2K_LSDIR_valid_LR](https://drive.google.com/file/d/1YUDrjUSMhhdx1s-O0I1qPa_HjW-S34Yj/view?usp=sharing) and [DIV2K_LSDIR_valid_HR](https://drive.google.com/file/d/1z1UtfewPatuPVTeAAzeTjhEGk4dg2i8v/view?usp=sharing)), please organize them as follows:

```
|NTIRE2026_ESR_Challenge/
|--DIV2K_LSDIR_valid_HR/
|    |--000001.png
|    |--000002.png
|    |--...
|    |--000100.png
|    |--0801.png
|    |--0802.png
|    |--...
|    |--0900.png
|--DIV2K_LSDIR_valid_LR/
|    |--000001x4.png
|    |--000002x4.png
|    |--...
|    |--000100x4.png
|    |--0801x4.png
|    |--0802x4.png
|    |--...
|    |--0900.png
|--NTIRE2026_ESR/
|    |--...
|    |--test_demo.py
|    |--...
|--results/
|--......
```

## How to test the baseline model?

1. `git clone https://github.com/Amazingren/NTIRE2026_ESR.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 0
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
3. More detailed example-command can be found in `run.sh` for your convenience.

As a reference, we provide the results of SPAN (baseline method) below:
- Average PSNR on DIV2K_LSDIR_valid: 26.94 dB
- Average PSNR on DIV2K_LSDIR_test: 27.01 dB
- Number of parameters: 0.151 M
- Runtime: 5.59 ms (Average runtime of 5.62 ms on DIV2K_LSDIR_valid data and 5.57 ms on DIV2K_LSDIR_test data)
- FLOPs on an LR image of size 256×256: 9.83 G

    Note that the results reported above are the average of 5 runs, and each run is conducted on the same device (e.g., NVIDIA RTX A6000 GPU).


## How to add your model to this baseline?

1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/11JuxcS78C6Gxc8B436L4Zk4_m5soHaTcw3cnF8h5ctE/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in `./models/[Your_Team_ID]_[Your_Model_Name].py`
   - Please add **only one** file in the folder `./models`. **Please do not add other submodules**.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02 
3. Put the pretrained model in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].[pth or pt or ckpt]`
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02  
4. Add your model to the model loader `./test_demo/select_model` as follows:
    ```python
        elif model_id == [Your_Team_ID]:
            # define your model and load the checkpoint
    ```
   - Note: Please set the correct data_range, either 255.0 or 1.0
5. Send us the command to download your code, e.g, 
   - `git clone [Your repository link]`
   - We will do the following steps to add your code and model checkpoint to the repository.
   

## How to calculate the number of parameters, FLOPs, and activations

```python
    from utils.model_summary import get_model_flops, get_model_activation
    from models.team00_SPAN import SPAN
    from fvcore.nn import FlopCountAnalysis

    model = SPAN()
    
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    # The FLOPs calculation in previous NTIRE_ESR Challenge
    # flops = get_model_flops(model, input_dim, False)
    # flops = flops / 10 ** 9
    # print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    # fvcore is used in NTIRE2026_ESR for FLOPs calculation
    input_fake = torch.rand(1, 3, 256, 256).to(device)
    flops = FlopCountAnalysis(model, input_fake).total()
    flops = flops/10**9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```

## How the Ranking Strategy Works?

After the organizers receive all the submitted codes/checkpoints/results, four steps are adopted for the evaluation:

- Step1: The organizers will execute each model five times to reevaluate all submitted methods on the same device, specifically the NVIDIA RTX A6000. The average results of these five runs will be documented for each metric.
- Step2: To ensure PSNR consistency with the baseline method SPAN, PSNR checks will be conducted for all submitted methods. Any method with a PSNR below 26.90 dB on the DIV2K_LSDIR_valid dataset or less than 26.99 on the DIV2K_LSDIR_test datasets will be excluded from the comparison list for the remaining rankings. 
- Step3: For the rest, the *Score_Runtime*, *Score_FLOPs*, and the *Score_Params* will be calculated as follows:

```
     Score_Runtime = exp(2*Runtime / Runtime_SPAN)
    
     Score_FLOPs = exp(2*FLOPs / FLOPs_SPAN)
     
     Score_Params = exp(2*Params / Params_SPAN)
```
-   Step4: The final comparison score will be calculated as follows:
```
    Score_Final = 0.8*Score_Runtime + 0.1*Score_FLOPs + 0.1*Score_Params
```
Let's take the baseline as an example, given the results (i.e., average Runtime_SPAN = 5.59 ms, FLOPs_SPAN = 9.83 G, and Params_SPAN = 0.151 M) of SPAN, we have:
```
    Score_Runtime = 7.3891
    Score_FLOPs   = 7.3891
    Score_Params  = 7.3891
    Score_Final   = 7.3891
```
:heavy_exclamation_mark:The ranking for each sub-track will be generated based on the corresponding Score (i.e., *Score_Runtime*, *Score_FLOPs*, and *Score_Params*), while for the main track, the ranking will be determined by the *Score_Final*.


## References
If you feel this codebase and the report paper is useful for you, please cite our challenge report:
```
@inproceedings{ren2025tenth,
  title={The tenth NTIRE 2025 efficient super-resolution challenge report},
  author={Ren, Bin and Guo, Hang and Sun, Lei and Wu, Zongwei and Timofte, Radu and Li, Yawei and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={917--966},
  year={2025}
}

@inproceedings{ren2024ninth,
  title={The ninth NTIRE 2024 efficient super-resolution challenge report},
  author={Ren, Bin and Li, Yawei and Mehta, Nancy and Timofte, Radu and Yu, Hongyuan and Wan, Cheng and Hong, Yuxin and Han, Bingnan and Wu, Zhuoyuan and Zou, Yajun and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={6595--6631},
  year={2024}
}
```

## Organizers
- **Bin Ren**, MBZUAI, United Arab Emirates (bin.ren@mbzuai.ac.ae)
- **Hang Guo**, Tsinghua University, China (cshguo@gmail.com)
- **Yan Shu**, UNITN, Italy (yan.shu@unitn.it)
- **Jiaqi Ma**, MBZUAI, United Arab Emirates (jiaqi.ma@mbzuai.ac.ae)
- **Ziteng Cui**, Univesity of Tokyo, Japan (cui@mi.t.u-tokyo.ac.jp)
- **Shuhong Liu**, Univesity of Tokyo, Japan (s-liu@mi.t.u-tokyo.ac.jp)
- **Guofeng Mei**, FBK, Italy (gmei@fbk.eu)
- **Lei Sun**, INSAIT, BG (lei.sun@insait.ai)
- **Zongwei Wu**, University of Wuerzburg, Germany (zongwei.wu@uni-wuerzburg.de)
- **Salman Khan**, MBZUAI, United Arab Emirates (salman.khan@mbzuai.ac.ae)
- **Fahad Shahbaz Khan**, MBZUAI, United Arab Emirates (fahad.khan@mbzuai.ac.ae)
- **Radu Timofte**, University of Wuerzburg, Germany (radu.timofte@uni-wuerzburg.de) 
- **Yawei Li**, ETH Zurich, Switzerland (yawei.li.ai@gmail.com)

If you have any question, feel free to reach out the contact persons and direct managers of the NTIRE challenge.


## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 