# Active learning with generated synthetic image for instrument segmentation

The 3 '.ipynb' script can be run in Google Colab. To run it on your Colab, paths in the following cells should be modified to fit your Google Drive path.

```
import sys
path_workspace = '/content/active_learning_v1'
path_source = '/content/drive/MyDrive/active_learning_ws/active_learning_v1'
sys.path.append(path_source)
```

Source files for UW Sinus datasets can be found in 'src/active_syn_UW_sinus'

Source files for Endovis datasets can be found in 'src/active_syn_Endovis'

```
import shutil
!mkdir '/content/active_learning_v1'

shutil.copy("/content/drive/MyDrive/active_learning_ws/active_learning_v1/labels.json", 
            "/content/active_learning_v1/labels.json")

!unzip -q "/content/drive/MyDrive/active_learning_ws/data/database_folder12_l.zip" -d '/content/active_learning_v1'
!unzip -q "/content/drive/MyDrive/active_learning_ws/data/background_for_syn_folder12_l.zip" -d '/content/active_learning_v1'
!unzip -q "/content/drive/MyDrive/active_learning_ws/active_learning_v1/mobilenet_v1_1.0_224.zip" -d '/content/active_learning_v1'
```

[Link to UW Sinus Live/Cadaver dataset](https://digital.lib.washington.edu/researchworks/handle/1773/45396)

[Link to EndoVis 2017 dataset](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Data/)

[Link to mobilenet_v1_1.0_224](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

[Link to ResNet50 V2](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
