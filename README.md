# EMFiT
## Abstract
Accurate identification of embryo developmental stages plays a crucial role in assisted reproductive technology, as it not only provides clinicians with a clearer understanding of the current developmental status of embryos, but also enables the extraction of more precise morphokinetic parameters, thereby enhancing the objectivity of embryo selection and improving the success rate of embryo implantation. Currently, time-lapse imaging systems are widely used in embryo culture, capturing embryo images at multiple focal planes periodically. However, most existing studies still rely primarily on singlefocus images for stage classification and often overlook the temporal information embedded in the developmental process.To address these limitations, we propose a novel model—Embryo Multi-focus and Frame-index Transformer (EMFiT). EMFiT takes as input both multi-focus embryo images and frame-index information that reflects the temporal sequence of development. A 3D Convolutional Neural Network (CNN) encoder is employed to efficiently extract spatial features across focal planes, while the frame-index is encoded and integrated with image features through a Transformer encoder, enabling global feature modeling and deep interaction between spatial and temporal information. Furthermore, the model is trained with frame-index regression, guiding it to focus on temporal evolution during both feature learning and classification stages. Extensive experiments on a large-scale human embryo time-lapse video dataset demonstrate that EMFiT outperforms previous methods and achieves state-of-the-art performance. Specifically, it reaches a classification accuracy, precision, recall, F1-score, and Cohen’s Kappa of 75.8%, 62.4%, 64.2%, 63.3%, and 0.730, respectively. With dynamic programming (DP) post-processing to enforce temporal consistency, these metrics further improve to 77.1%, 63.0%, 66.7%, 64.8%, and 0.744. These results suggest that EMFiT offers an efficient, robust, and clinically promising solution for automatic recognition of embryo developmental stages.

## File Composition
The Results folder contains the training and validation results of the EMFiT model and its ablation studies, including loss, accuracy, etc.

The embryo_dataset_annotations_revise folder contains the cleaned annotation files of the dataset, you can download the time-lapse embryo images from this [[link]](https://zenodo.org/records/6390798).

The embryo_development_stage folder includes the implementation code of the EMFiT model and its ablation experiments.

The pretrained model weights required by the code, as well as the trained weights of the EMFiT model and its ablation experiments, can be downloaded from this [[link]](https://drive.google.com/drive/folders/1DgymniuN4OnOJzqNn2txzryzhbTH8COH?usp=drive_link).

<img width="2860" height="1324" alt="Graphical Abstract" src="https://github.com/user-attachments/assets/56875c9f-ed1f-4aa8-98c2-f6f77f4d48ab" />

## Train EMFiT
'''

'''
