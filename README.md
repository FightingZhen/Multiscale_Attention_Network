## Pulmonary Textures Classification via a Multi-Scale Attention Network

The prior work that using original image patches and corresponding eigen-value matrices as inputs requires large cost of calculation before deep learning training. Besides, we notice that different textures of diffuse lung diseases exhibit in various scales.

Taking advantage of attention mechanism, we proposed a multi-scale attention network for pulmonary textures classification, the architecture is shown as follows:

![avatar](/fig/network_structure.png)

Detail structure of attention block is presented as follows:

![avatar](/fig/attention.png)

## Searching for Baseline Network

We conducted a series of experiments to search for optimal baseline network structure, detail results are presented as follows:

![avatar](/fig/find_baseline.png)

## Ablation Study

Ablation study of the proposed network is shown as follows:

![avatar](/fig/ablation.png)

## Comparison with the State-of-the-Art

Comparisons with other state-of-the-art methods are exhibited as follows:

![avatar](/fig/SoA.png)

## Visualization Explanation

### Attention Visualization

We extracted feature maps after attention block with different weights, visualization results are shown as follows:
![avatar](/fig/attention_example.png)


### Grad-CAM Visualization

We use Grad-CAM to generate heatmap of feature maps of different categories' feature maps, which is shown as follows:

![avatar](/fig/grad_cam_example.png)

### Mis-Classified Examples

Following examples are mis-classified image patches, followed by detail classification probability:

![avatar](/fig/mis_classified_examples.png)
