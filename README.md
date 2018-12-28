# CaVINet
Code for Deep Cross modal learning for Caricature Verification and Identification (CaVINet), ACM MM, 2018

# Dataset
Please check `CaVINet_Dataset` folder for the visual as well as the caricature images (both present in independent folders).
Note : The `CaVI Dataset` must be used for research only and not for any profit/commercial purpose.
Dataset Directory structure:
```
CaVINet_Dataset
└───Real
│    └───Angela_Merkel
│    │   │   Angela_Merkel_r_0.jpg
│    │   │   Angela_Merkel_r_1.jpg
│    │   │   ...
│    │   
│    └───Bill_Gates
│    |   │   Bill_Gates_r_0.jpg
│    |   │   Bill_Gates_r_1.jpg
│    |   |   ...
│    |
│    └─── ...
│
└───Caricature
    └───Angela_Merkel
    │   │   Angela_Merkel_c_0.jpg
    │   │   Angela_Merkel_c_1.jpg
    │   │   ...
    │   
    └───Bill_Gates
    |   │   Bill_Gates_c_0.jpg
    |   │   Bill_Gates_c_1.jpg
    |   |   ...
    |
    └─── ...

```
# Requirements
- Keras with TensorFlow backend
- [Keras-VggFace](https://github.com/rcmalli/keras-vggface) - `pip install keras_vggface`
- `python 2.7`

# Architecture
![](arch.png)

# Pre-trained weights
Link to be added soon. Please stay tuned.
