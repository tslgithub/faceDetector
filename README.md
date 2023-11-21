# RetinaFace in PyTorch

## Installation
> ```bash
> pip3 install -r requirements.txt
> ```

## download weight
> https://drive.usercontent.google.com/download?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW&export=download&authuser=0&confirm=t&uuid=1c047ae2-05e4-4f33-b66e-e0d73a711f5b&at=APZUnTWD5sWebJKA6TbMpsDvAHFJ:1700533565125
>
> save the weight in folder named `weigths`

## run1
> ```bash
> python3 script.py  images/
> python3 script.py  images/1.jpg
> ```
>  * use defaults vis_thres is 0.6
>  * result will be saved in folder named result
>

## run3
> ```bash
> python3 script.py  images/ 0.7
> python3 script.py  images/1.jpg 0.7
> ```
> * 0.7 means vis_thres=0.7, or defaults is 0.6
> * result will be saved in folder named result
>
