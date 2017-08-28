All code used for the experiments on CIFAR.

cifar10.py                                  The tensorflow CIFAR network originally used as base
cifar10_ACOL_VGG16.py                       ACOL placed on VGG16
cifar10_ACOL_VGG16_fc7_norelu.py            ACOL placed on VGG16, before the relu
cifar10_ACOL_VGG16_onfc8.py                 ACOL placed on VGG16, on the fc8 layer
cifar10_ACOL_VGG16_varianceSelection.py     ACOL placed on VGG16, making selections of the feature spaces based. The selection is loaded                                                 from a pickled file.
cifar10_ACOLPL.py                           ACOL-PL placed on the tensorflow CIFAR network
cifar10_ACOLPL_VGG16.py                     ACOL-PL placed on VGG16, also used for ordered training
cifar10_ACOLPL_VGG16_noACOLst1.py           ACOL-PL placed on VGG16, not using the ACOL loss for the first stage of ordered training
cifar10_ACOLPL_VGG16_selectedFeatures.py    ACOL-PL placed on VGG16, making selections of the feature spaces based. The selection is loaded                                             from a pickled file.
cifar10_eval.py                             The evaluation script of the direct tensorflow CIFAR network.
cifar10_eval_ACOL.py                        I think I can delete this?
cifar10_eval_VGG.py                         The evaluation script of the direct VGG16 CIFAR network.
cifar10_eval_VGG_super.py                   The evaluation script of the direct VGG16 CIFAR network, classifying to the superclasses.
cifar10_input.py                            Inputscript for the original tensorflow CIFAR network.
cifar10_input_PL.py                         Inputscript for the original tensorflow CIFAR network.
cifar10_input_VGG16.py                      Inputscript for the VGG16 CIFAR network.
cifar10_input_VGG16_PL.py                   Inputscript for the VGG16 CIFAR network, providing partially labelled data from records made                                                 with mess\ with\ cifar\ binaries.ipynb.
cifar10_input_VGG16_PL_distortLabels.py     Inputscript for the VGG16 CIFAR network, attempted all lossless transformations distortion,                                                 providing partially labelled data from records made with mess\ with\ cifar\ binaries.ipynb.
cifar10_multi_gpu_train.py                  The original tensorflow CIFAR network utilising multiple GPU's.
cifar10_PL.py                               Direct partially labelled classification using the tensorflow CIFAR network.
cifar10_train.py                            Trainscript for the tensorflow CIFAR network.
cifar10_train_ACOL.py                       Trainscript for ACOL on the tensorflow CIFAR network.
cifar10_train_ACOL_VGG16.py                 Trainscript for ACOL on the VGG16 CIFAR network.
cifar10_train_ACOLPL.py                     Trainscript for ACOL-PL on the tensorflow CIFAR network.
cifar10_train_ACOLPL_VGG16.py               Trainscript for ACOL-PL on the VGG16 CIFAR network, also used for Ordered ACOL-PL.
cifar10_train_ACOLPL_VGG16_noACOLst1.py     Trainscript for ACOL-PL whereby the first stage does not use the ACOL loss.
cifar10_train_PL.py                         Trainscript for direct partially labelled on the tensorflow CIFAR network.
cifar10_train_VGG16.py                      Trainscript for direct classification using the VGG16 CIFAR network.
cifar10_train_VGG16_PL.py                   Trainscript for direct partially labelled classification using the VGG16 CIFAR network.
cifar10_train_VGG16_PL_superclass.py        Trainscript for direct partially labelled classification to the superclasses using the VGG16                                                 CIFAR network.
cifar10_VGG16.py                            Direct classification VGG16 CIFAR network.
cifar10_VGG16_PL.py                         Direct partially labelled classification VGG16 CIFAR network.
cifar10_VGG16_PL_superclass.py              Direct partially labelled classification to the superclasses VGG16 CIFAR network.
imagenetSelection.csv                       The 200+ imagenet classes selected for Experiment 6.
runtsne.py                                  Fast and efficient Barnes-Hutt t-sne implementation.
VGG16base.py                                The VGG16 tensorflow model.


fc8Selection.ipynb                          Some snippets to create selections on the VGG16 fc8 layer.
Running_things.ipynb                        Notebook used to queue multiple CIFAR train scripts.
mess\ with\ cifar\ binaries.ipynb           Inserts bytes into the CIFAR records to enable/disable labels for PL, and to add superclass info
vggfeatures_cifar.ipynb                     Some snippets to analyse and select subsets from VGG16.
Visualise\ ACOLcifar.ipynb                  Visualising and getting the results from any ACOL based network.


Details of the tensorflow network can be found here:
http://tensorflow.org/tutorials/deep_cnn/

