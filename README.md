# semantic-segmentation-for-self-driving-cars
We build a modified U-Net model which is different from the original model described
in the UNet paper, by introducing the mish activation function since the mish activation was found
to outperform the ReLU activation in deep convolutional neural networks, the
modified model also uses instance normalization since it was found to be useful
in a similar application.
# Model architecture

he model consists of a contracting path and an expansive path along with skip connections from the contracting to the expansive path, first of all the input image is normalized then the contracting path consists of the repeated application of 3x3 convolutional
layer followed by normalization layer (Instance normalization) ,then comes the activation function (mish), after each two convolution layers a 2x2 MaxPool layer is used for
down-sampling , for each such block we double the number of feature channels, we repeat the same block four times ,The expansive path consist of repeated application of
de-convolutional layer followed by 3x3 convolutional layer, instance normalization layer
, the activation function (mish) ,3x3 convolutional layer then instance normalization layer
and finally the activation function (mish) ,for each such block we reduce the number of
feature channels by the half, the input of each block is the output of the previous block
concatenated with skip connection from the block parallel to it from the contracting path
, the block is repeated four times , then a final convolutional layer is added after the final block of the expansive path the number of output features of that layer is equal to the
number of object classes.

# The Dataset

In order to train and test the model we used the virtual KITTI dataset, the dataset is
created using the Unity game engine, it contains photos from 50 high-resolution videos
created from five different virtual worlds under various weather and image conditions.
The dataset was designed to train and evaluate models for different computer vision tasks
including semantic segmentation. Images below are some examples from the dataset for
Different shots of various images and weather conditions of the same view.

# Results

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/96c6dd82-8fdd-4224-8197-25bfa407ad55" alt="semantic result 1" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/dfc5849d-f6ef-43c2-b6f7-497190a356a4" alt="semantic result 2" width="400"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/7a6399bc-2829-457d-9d48-a09b452d06f9" alt="semantic result 3" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/53c0e89d-c6fe-4732-968b-f6b099e2b5f2" alt="semantic results 4" width="400"/></td>
  </tr>
</table>

