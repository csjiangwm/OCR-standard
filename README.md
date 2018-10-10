# CTPN+CRNN_pytorch/DenseNet_keras+CTC for Chinese character recognition





## Deployment
```Bash
cd lib
chmod +x make.sh
./make.sh
```

## Test
Download the CTPN model from [BaiduyunDisk](https://pan.baidu.com/s/1CGwxKrJr5gtznGtM05Y4HA) with extract code: "48j2"<br>
Download the CRNN model from [BaiduyunDisk](https://pan.baidu.com/s/1wTM-mBX0Beg-xkLZuhLwog) with extract code: "2yww"<br>
Or download the DenseNet model from [BaiduyunDisk](https://pan.baidu.com/s/1qUJ4NNY2tKGE0ll_QGuGXA) with extract code: "uafx"<br>
If you want to detect the text direction and rotate your image, you should download the VGG model from [BaiduyunDisk](https://pan.baidu.com/s/1L14LJYmkU37S-aAXl06CAg) with extract code: "ass4". Note that this VGG model can only detect 0, 90, 180, 270 degrees.<br>

```Bash
python ctpn_crnn.py --CTPN_MODEL your_ctpn_path \
		    --CRNN_MODEL your_crnn_path \
		    --VGG_MODEL your_vgg_path \
		    --ADJUST_ANGLE your_choice
```
Or
```Bash
python ctpn_densenet.py --CTPN_MODEL your_ctpn_path \
			--DENSENET_MODEL your_densenet_path \
			--VGG_MODEL your_vgg_path \
			--ADJUST_ANGLE your_choice
```

## Some results
![](https://github.com/csjiangwm/OCR-standard/blob/master/images/timg.jpg)<br>
Detected result<br>
![](https://github.com/csjiangwm/OCR-standard/blob/master/images/result.jpg)<br>
CRNN result<br>
![](https://github.com/csjiangwm/OCR-standard/blob/master/images/crnn.png)<br>
DenseNet result<br>
![](https://github.com/csjiangwm/OCR-standard/blob/master/images/densenet.png) <br>

It can be seen that the method can achieve a good performance in Chinese character recognition, but achieve a bad performance in numeric character and English character. Training data is the main reason for this result. So you may want to train your own model.


## Train CTPN
Download the training dataset from [BaiduyunDisk](https://pan.baidu.com/s/1ut_8j4ndwpzWG4sWAjLbrA) with extract code: "45g7"<br>
Download the pretrained CTPN model from [BaiduyunDisk](https://pan.baidu.com/s/136nhJP-0gMCupvTcSjcqKw) with extract code: "jsp4"<br>

```Bash
python CTPN_train.py --PRETRAINED_MODEL your_path \
		     --DATA_DIR your_data_path \
		     --SAVED_PATH the_path_you_want_to_save_your_model \
		     --CTPN_LOGGER your_logger_path
```

## Train DenseNet
Download the training dataset from [BaiduyunDisk](https://pan.baidu.com/s/1IRdf7P6JDV6HZQkJFGfgwA) with extract code: "uqic"<br>


```Bash
python DenseNet_train.py --PRETRAINED_MODEL your_path \
		     --DATA_DIR your_data_path \
		     --SAVED_PATH the_path_you_want_to_save_your_model \
		     --CTPN_LOGGER your_logger_path
```





## Reference

- [pytorch_crnn](https://github.com/meijieru/crnn.pytorch.git)    
- [tensorflow-ctpn](https://github.com/eragonruan/text-detection-ctpn )
- [keras-densenet](https://github.com/YCG09/chinese_ocr)

