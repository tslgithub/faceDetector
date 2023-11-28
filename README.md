# RetinaFace in PyTorch

## support
> run on ubuntu20, ubuntu18, python3


## Installation
> ```bash
> sudo apt-get install python3-pip3
> pip3 install -r requirements.txt
> ```

## download weight
> https://drive.usercontent.google.com/download?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW&export=download&authuser=0&confirm=t&uuid=1c047ae2-05e4-4f33-b66e-e0d73a711f5b&at=APZUnTWD5sWebJKA6TbMpsDvAHFJ:1700533565125
>
> * save the weight in folder named `weigths`

## run script
> ```bash
> python3 script.py  images/
> python3 script.py  images/1.jpg
> ```
>  * use defaults vis_thres is 0.6
>  * result will be saved in folder named result
>

## evaluation
> 1. download the val images in 
>`https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view`   
>  
> 2. unzip the `WIDER_val.zip`
> 3. download the val annotation label in `http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip`
> 4. unzip `wider_face_split.zip` in folder of `WIDER_val`, and rename `wider_face_split` as `annotation`


> ```bash
> python3 metric.py         # use default iou_thres=0.5
> python3 metric.py  0.7    # use  iou_thres=0.7
> ```
> * 0.7 means iou_thres=0.7, or defaults is 0.5
> * result will be saved in folder named result
>

## unit test
> 1. download test dataset in 
> `https://drive.google.com/file/d/1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T/view`
> 2. unzip the test dataset
> 3. run test: 
> ```python
> python3 script.py WIDER_test/images/0--Parade
> python3 script.py WIDER_test/images/1--Handshaking
> ...
> ...
> ...
> python3 script.py WIDER_test/images/61--Street_Battle
> ```
>

## answers
> 1. add metric evaluation
> 2. add unit test
> 3. simply codo
> 4. the train.py is not belong to my code, because this code is from  `https://github.com/TreB1eN/InsightFace_Pytorch`, and I have not do any training.