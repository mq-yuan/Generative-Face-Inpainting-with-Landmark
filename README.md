
![result_1](https://typora-ilgzh.oss-cn-beijing.aliyuncs.com/202303241316072.png)

## 实验环境

我们的实验基于以下环境：

* Python: 3.7.6
* torch: 1.9.0 (cuda 11.1)
* torchvision: 0.10.0
* tqdm: 4.61.1
* Pillow: 8.2.0
* opencv-python: 4.5.2.54
* opencv-python-headless: 4.1.2.30
* numpy: 1.19.2
* insightface: 0.7.2
* onnxruntime: 1.13.1
* GPU: Geforce GTX 2080Ti (11GB RAM)

可以通过以下命令进行环境配置

```sh
# in <path-to-this-repo>/
pip install -r requirements.txt
```

## Download Dataset CelebA

实验所使用的数据集为CelebA-HQ高清数据.

实验选取celebA_hq_1024的前30000张图片进行训练与测试。

链接：https://pan.baidu.com/s/1ZF1G2MQILZSFNjD1YLVgZA

提取码：t76m

- celebA_hq_64 ——提取码：6hf3
- celebA_hq_128——提取码：xk6d
- celebA_hq_256——提取码：016n
- celebA_hq_512——提取码：byix
- celebA_hq_1024——提取码：bszu

## Test

确保预训练生成模型``./models/phase_3_model_generator_epoch64``和config配置``./config/config.json``存在。

将需测试的图片放入``./images/test``文件夹中，运行以下命令，在``./results``中查看测试结果。

```bash
# in <path-to-this-repo>/
python predict.py ./models/phase_3_model_generator_epoch64 ./config/config.json ./results/
```
<a name="arguments"></a>
**Arguments**  
* `<model>` (必须): 预训练生成模型路径。
* `<config>` (必须): 配置文件路径。
* `<result>` (必须): 测试结果路径。
* `[--mode (str)]`(可选):如果你希望将所有测试图片结果拼成一张图片可以输入`cat`。(默认为`None`)

## 训练

训练使用celebA-HQ高清人脸数据集，选用整个数据集的前30000张图片进行训练，确保已将所有训练图片放入``./datasets``文件夹中。

训练基于原始论文的预训练模型，确保``demo/model_cn``文件存在。

运行以下命令，进行训练。

```bash
# in <path-to-this-repo>
python train.py --model_G ./demo/model_cn
```

<a name="arguments"></a>
**Arguments**  
* `<--model_G>` (可选): 预训练生成模型路径。(默认为``./demo/model_cn``)
* `<--model_D>` (可选): 已训练鉴别模型路径。(默认为``None``)
* `<--phase1>` (可选): phase1 训练次数。(默认为8)
* `<--phase2>` (可选): phase2 训练次数。(默认为2)
* `<--phase3>` (可选): phase3 训练次数。(默认为40)


训练的结果将会按阶段存放到 ``./results/phasex``中.

训练过程包括如下阶段：  
* **Phase 1**: 只训练生成过程
* **Phase 2**: 固定生成网络，只训练鉴别器。
* **Phase 3**: 同时训练生成器和鉴别器。


## 实验结果

### 视觉效果

![result_1](https://typora-ilgzh.oss-cn-beijing.aliyuncs.com/202303241316074.png)
<center> 图1 视觉效果比较图 </center>

### 关键点

![result_2](https://typora-ilgzh.oss-cn-beijing.aliyuncs.com/202303241316075.png)
<center> 图2 关键点比较图 </center>

### 图像质量定性分析

表1 图像质量定量分析表

|                    |       MSE     |      PSNR    |      SSIM    |
|:------------------:|:-------------:|:------------:|:------------:|
|         RAW        |       0.0     |      Inf     |      1.0     |
|         MASK       |     0.0145    |     18.76    |     0.804    |
|     No Landmark    |     0.0080    |     21.68    |     0.842    |
|         Our        |     0.0079    |     21.72    |     0.857    |

</center>

### 下游任务测试

表2 人脸检测结果表

|                    |     Prob(%)    |
|--------------------|----------------|
|     RAW            |     99.8       |
|     MASK           |     49.8       |
|     No Landmark    |     99.7       |
|     Our            |     99.9       |

表3 人脸识别结果表

<table border=0 cellpadding=0 cellspacing=0 width=370 style='border-collapse:
 collapse;table-layout:fixed;width:280pt'>
 <col width=90 style='mso-width-source:userset;mso-width-alt:2892;width:68pt'>
 <col width=70 span=4 style='width:53pt'>
 <tr height=19 style='height:14.4pt'>
  <td rowspan=2 height=38 class=xl68 width=90 style='border-bottom:.5pt solid black;
  height:28.5pt;width:68pt'><span lang=EN-US>　</span></td>
  <td colspan=2 class=xl69 width=140 style='width:106pt'><span lang=EN-US>Top
  1(%)</span></td>
  <td colspan=2 class=xl69 width=140 style='width:106pt'><span lang=EN-US>Top
  5(%)</span></td>
 </tr>
 <tr height=19 style='height:14.1pt'>
  <td height=19 class=xl70 width=70 style='height:14.1pt;width:53pt'><span
  lang=EN-US>Acc</span></td>
  <td class=xl71 width=70 style='border-top:none;width:53pt'><span lang=EN-US>Error</span></td>
  <td class=xl70 width=70 style='width:53pt'><span lang=EN-US>Acc</span></td>
  <td class=xl70 width=70 style='width:53pt'><span lang=EN-US>Error</span></td>
 </tr>
 <tr height=19 style='height:14.1pt'>
  <td height=19 class=xl65 width=90 style='height:14.1pt;width:68pt'><span
  lang=EN-US>RAW</span></td>
  <td class=xl65 width=70 style='width:53pt'><span lang=EN-US>86.9</span></td>
  <td class=xl65 width=70 style='width:53pt'><span lang=EN-US>13.1</span></td>
  <td class=xl65 width=70 style='width:53pt'><span lang=EN-US>93.9</span></td>
  <td class=xl65 width=70 style='width:53pt'><span lang=EN-US>6.1</span></td>
 </tr>
 <tr height=34 style='height:25.8pt'>
  <td height=34 class=xl65 width=90 style='height:25.8pt;width:68pt'><span
  lang=EN-US>No Landmark</span></td>
  <td class=xl65 width=70 style='width:53pt'><span lang=EN-US>68.9</span></td>
  <td class=xl65 width=70 style='width:53pt'><span lang=EN-US>31.7</span></td>
  <td class=xl65 width=70 style='width:53pt'><span lang=EN-US>86.4</span></td>
  <td class=xl65 width=70 style='width:53pt'><span lang=EN-US>17.6</span></td>
 </tr>
 <tr height=19 style='height:14.4pt'>
  <td height=19 class=xl66 width=90 style='height:14.4pt;width:68pt'><span
  lang=EN-US>Our</span></td>
  <td class=xl67 width=70 style='width:53pt'><span lang=EN-US>82.4</span></td>
  <td class=xl67 width=70 style='width:53pt'><span lang=EN-US>17.6</span></td>
  <td class=xl67 width=70 style='width:53pt'><span lang=EN-US>91.2</span></td>
  <td class=xl67 width=70 style='width:53pt'><span lang=EN-US>8.8</span></td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=90 style='width:68pt'></td>
  <td width=70 style='width:53pt'></td>
  <td width=70 style='width:53pt'></td>
  <td width=70 style='width:53pt'></td>
  <td width=70 style='width:53pt'></td>
 </tr>
 <![endif]>
</table>

## 参考文献

```
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}
@article{Iizuka2017GloballyAL,
  title={Globally and locally consistent image completion},
  author={Satoshi Iizuka and Edgar Simo-Serra and Hiroshi Ishikawa},
  journal={ACM Transactions on Graphics (TOG)},
  year={2017},
  volume={36},
  pages={1 - 14}
}
@inproceedings{deng2019accurate,
    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
    year={2019}
}
```
