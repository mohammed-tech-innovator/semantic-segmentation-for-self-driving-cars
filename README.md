# semantic-segmentation-for-self-driving-cars
A modified U-Net model, different from the original model described in the UNet paper, was built. The mish activation function was introduced into this modified model since it had been found to outperform the ReLU activation in deep convolutional neural networks. Additionally, instance normalization was used in the modified model as it had been found to be useful in a similar application.
# Model architecture

The model is composed of a contracting path and an expansive path, interconnected by skip connections.

Initially, the input image is normalized. The contracting path is constructed by repeatedly applying a 3x3 convolutional layer followed by instance normalization and mish activation. Down-sampling is achieved by employing a 2x2 MaxPool layer after every two convolutional layers. The number of feature channels is doubled for each such block, which is repeated four times.

The expansive path is formed by repeatedly applying a deconvolutional layer, followed by two sequential blocks of 3x3 convolutional layer, instance normalization, and mish activation. The number of feature channels is halved for each block. The input to each block is generated by concatenating the output of the previous block with the corresponding skip connection from the contracting path. This block is repeated four times.

Finally, a convolutional layer with the number of output features equal to the number of object classes is added after the final block of the expansive path.
<td><img src="https://github.com/user-attachments/assets/33009da9-8e1b-460d-b6c0-825283fe42d5" alt="UNet" width="400"/></td>

# The Dataset

The virtual KITTI dataset, generated using the Unity game engine, was employed for model training and testing. This dataset comprises photographs extracted from 50 high-resolution videos, captured across five distinct virtual worlds under varying weather and lighting conditions. Designed to facilitate training and evaluation of models for diverse computer vision tasks, including semantic segmentation, the dataset offers a rich collection of images showcasing different perspectives and weather effects of identical scenes.

# Results
- Avg test accuracy : 96.64%
- Avg test IoU : 86.14%
- Here're some results samples :
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

