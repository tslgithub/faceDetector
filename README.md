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
> save the weight in folder named `weigths`

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
>`https://doc-14-1g-docs.googleusercontent.com/docs/securesc/nvcv7bsaqhjdod5uutgegrlcjkn8boeh/441thtg8gmbvdbeok188rg0b43c657m7/1700713875000/05015110782597501518/17043419117647764458/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q?e=download&ax=AEqgLxkhWJvKaQqJ_2RG-ezDPEh57YtCIAKI_v0zvdJfgBZp5yr8ZTK5bmPcOrA35YYY9aVy8_HQVJnYx3DzKuqpKagS7O9hu9y_tfOxfMJCYxxNMNHJcN3Cb-bLpPMJDZlqKDrsTr-ut7OkImOc-Gfq5adl40r1tEgy4ro9O0D0H3FlDRN5Fab-Y2Q-F5QuRpn0wK9I4Zl_iEtzkyzXOFO1b0GhCwzUbroZMHACPco1v0uSg_0yXGXOSpmiLoro_bS-yELBeHucQ_T2ji-XzdBy0joulOES2SWQd0CmoEIvj0QW6WAL8xnF5oJNniqQxsdj8w64OaEam8g7rqHzV9r-jnmkn6Q08Ry49PkEDVwTq6MAzqPEJ3SCHZP357aMJ8Rm1FHtypZ5yRnkn82yDIc9r5IrvtjYsBlEUMdieFLTVRYxlzVDji1a1a11dBV_gqFAba38mOfiZC4HmrFjgpqfTJPR1CXYcxFKpo5HwL6jbttpcV_5CjlAjbK0xIIta4Wqc2HumMzXth9zsFXJB4_ChodfaZtORjnY8nr3xlWtCVXvdQk6IO_wQ1snfH0YvpRaZfHZbz48mGy7ZOdUrPLCITEuO2Jqp-5WX-jP-AxDUrrMF9HvGaCp_BpFgRjgSgTDaRcvXL6Pj55XzKwn_TgzTrDFceuxx9Q32hlZV52Cw08O5tSxQKrIbhDTMLQ2gs98FZrIslMGufnSi6wGCIfE3AZyaKlo2mg3Y8Eaf8SVcLF7eYLBsfDxAQ1VDvKQy9cZLKmfCLsQtCboAaJJyxDn6LfO9QlOciivAHcYFoNqyNfQ3nxjnT2bv8tefhfA7Fo_T8HpFOZ1sE9tP1X9n4MX2r3-qWHsWfctTpzn3wr6Zkgss9SFmf219iMkfX2-kt4&uuid=1842e780-12c0-4bee-ad0e-4a503b4dc2e1&authuser=0&nonce=h1psitbdhegbi&user=17043419117647764458&hash=g8enjvc7pbl88uofsckdti1qum387ijp`   
>  
> 2. unzip the `wider_face_split.zip`
> 3. download the val annotation label in `http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip`
> 4. unzip wider_face_split.zip in folder of "WIDER_val", and rename "wider_face_split" as "annotation"  

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
> 2. unzip the test dataset, and take all of the test-dataset into images
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

