# Script to initialise data in database, only for initializing empty database

bazel build tflite_init
bazel-bin/tfite_init
bazel build tflite_main
bazel-bin/tflite_main --filename albert_lite_base_squadv1_1_TextEmbedding.tflite --model_name albert_lite_base_squadv1_1 --category TextEmbedding
bazel-bin/tflite_main --filename deeplabv3_1_default_1_ImageSegmentation.tflite --model_name deeplabv3_1_default_1 --category ImageSegmentation
bazel-bin/tflite_main --filename densenet_ImageClassification.tflite --model_name densenet --category ImageClassification
bazel-bin/tflite_main --filename inception_v1_224_quant_ImageClassification.tflite --model_name inception_v1_224_quant --category ImageClassification
bazel-bin/tflite_main --filename magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1_ImageStyleTransfer.tflite --model_name magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1 --category ImageStyleTransfer
bazel-bin/tflite_main --filename mobilebert_1_default_1_TextEmbedding.tflite --model_name mobilebert_1_default_1 --category TextEmbedding
bazel-bin/tflite_main --filename mobilenet_v2_1.0_224_quant_ImageClassification.tflite --model_name mobilenet_v2_1.0_224_quant --category ImageClassification
bazel-bin/tflite_main --filename nasnet_mobile_ImageClassification.tflite --model_name nasnet_mobile --category ImageClassification
bazel-bin/tflite_main --filename object_detection_mobile_object_localizer_v1_1_default_1_ObjectDetection.tflite --model_name object_detection_mobile_object_localizer_v1_1_default_1 --category ObjectDetection
bazel-bin/tflite_main --filename posenet_mobilenet_float_075_1_default_1_ImagePoseDetection.tflite --model_name posenet_mobilenet_float_075_1_default_1 --category ImagePoseDetection
bazel-bin/tflite_main --filename resnet_v2_101_299_ImageClassification.tflite --model_name resnet_v2_101_299 --category ImageClassification
bazel-bin/tflite_main --filename smartreply_1_default_1_TextGeneration.tflite --model_name smartreply_1_default_1 --category TextGeneration
bazel-bin/tflite_main --filename squeezenet_ImageClassification.tflite --model_name squeezenet --category ImageClassification
bazel-bin/tflite_main --filename ssd_mobilenet_v1_1_default_1_ObjectDetection.tflite --model_name ssd_mobilenet_v1_1_default_1 --category ObjectDetection
bazel-bin/tflite_main --filename inception_resnet_v2_ImageClassification.tflite --model_name inception_resnet_v2 --category ImageClassification
bazel-bin/tflite_main --filename inception_v2_224_quant_ImageClassification.tflite --model_name inception_v2_224_quant --category ImageClassification
bazel-bin/tflite_main --filename inception_v3_quant_ImageClassification.tflite --model_name inception_v3_quant --category ImageClassification
bazel-bin/tflite_main --filename inception_v4_299_quant_ImageClassification.tflite --model_name inception_v4_299_quant --category ImageClassification
bazel-bin/tflite_main --filename mnasnet_1.3_224_ImageClassification.tflite --model_name mnasnet_1.3_224 --category ImageClassification
bazel-bin/tflite_main --filename mobilenet_v1_1.0_224_ImageClassification.tflite --model_name mobilenet_v1_1.0_224 --category ImageClassification
bazel-bin/tflite_main --filename nasnet_large_ImageClassification.tflite --model_name nasnet_large --category ImageClassification
bazel-bin/tflite_main --filename MobileNet-V3_ImageClassification.tflite --model_name MobileNet-V3 --category ImageClassification
bazel-bin/tflite_main --filename lite-model_spice_1_AudioPitchExtraction.tflite --model_name lite-model_spice_1 --category AudioPitchExtraction
bazel-bin/tflite_main --filename lite-model_deeplabv3-xception65-cityscapes_1_default_1_ImageSegmentation.tflite --model_name lite-model_deeplabv3-xception65-cityscapes_1_default_1 --category ImageSegmentation
bazel-bin/tflite_main --filename lite-model_deeplabv3-mobilenetv3-cityscapes_1_default_1_ImageSegmentation.tflite --model_name lite-model_deeplabv3-mobilenetv3-cityscapes_1_default_1 --category ImageSegmentation
bazel-bin/tflite_main --filename lite-model_deeplabv3-mobilenetv2_1_default_1_ImageSegmentation.tflite --model_name lite-model_deeplabv3-mobilenetv2_1_default_1 --category ImageSegmentation
bazel-bin/tflite_main --filename ssdlite_mobiledet_dsp_320x320_coco_2020_05_19_ObjectDetection.tflite --model_name ssdlite_mobiledet_dsp_320x320_coco_2020_05_19 --category ObjectDetection
bazel-bin/tflite_main --filename ssd_mobilenet_v3_small_coco_2020_01_14_ObjectDetection.tflite --model_name ssd_mobilenet_v3_small_coco_2020_01_14 --category ObjectDetection
bazel-bin/tflite_main --filename ssd_mobilenet_v3_large_coco_2020_01_14_ObjectDetection.tflite --model_name ssd_mobilenet_v3_large_coco_2020_01_14 --category ObjectDetection

bazel build tf_main
bazel-bin/tf_main --filename efficientnet_b5_feature-vector_1_ImageFeatureVector_SavedModel.pb --model_name efficientnet_b5_feature-vector_1 --category ImageFeatureVector 
bazel-bin/tf_main --filename progan-128_1_ImageGenerator_SavedModel.pb --model_name progan-128_1 --category ImageGenerator 
bazel-bin/tf_main --filename spiral_default-fluid-gansn-celebahq64-gen-19steps_1_ImageRNNAgent_SavedModel.pb --model_name spiral_default-fluid-gansn-celebahq64-gen-19steps_1 --category ImageRNNAgent 
bazel-bin/tf_main --filename unet_industrial_class_1_1_ImageSegmentation_SavedModel.pb --model_name unet_industrial_class_1_1 --category ImageSegmentation 
bazel-bin/tf_main --filename vision_embedder_inaturalist_V2_1_ImageFeatureVector_SavedModel.pb --model_name vision_embedder_inaturalist_V2_1 --category ImageFeatureVector 
bazel-bin/tf_main --filename tweening_conv3d_bair_1_VideoGeneration_SavedModel.pb --model_name tweening_conv3d_bair_1 --category VideoGeneration
bazel-bin/tf_main --filename local-linearity_cifar10_1_ImageClassification_SavedModel.pb --model_name local-linearity_cifar10_1 --category ImageClassification  
bazel-bin/tf_main --filename wae-mmd_1_ImageFeatureVector_SavedModel.pb --model_name wae-mmd_1 --category ImageFeatureVector 
bazel-bin/tf_main --filename videoflow_encoder_1_VideoGeneration_SavedModel.pb --model_name videoflow_encoder_1 --category VideoGeneration 
bazel-bin/tf_main --filename vae_1_ImageFeatureVector_SavedModel.pb --model_name vae_1 --category ImageFeatureVector 
bazel-bin/tf_main --filename ganeval-cifar10-convnet_1_ImageClassification_SavedModel.pb --model_name ganeval-cifar10-convnet_1 --category ImageClassification 
bazel-bin/tf_main --filename wiki40b-lm-en_1_TextLanguageModel_SavedModel.pb --model_name wiki40b-lm-en_1 --category TextLanguageModel 
bazel-bin/tf_main --filename small_bert_bert_uncased_L-4_H-256_A-4_1_TextEmbedding_SavedModel.pb --model_name small_bert_bert_uncased_L-4_H-256_A-4_1 --category TextEmbedding 
bazel-bin/tf_main --filename elmo_3_TextEmbedding_SavedModel.pb --model_name elmo_3 --category TextEmbedding 
bazel-bin/tf_main --filename mil-nce_s3d_1_VideoText_SavedModel.pb --model_name mil-nce_s3d_1 --category VideoText 
bazel-bin/tf_main --filename LAReQA_mBERT_En_En_1_TextRetrievalQuestionAnswering_SavedModel.pb --model_name LAReQA_mBERT_En_En_1 --category TextRetrievalQuestionAnswering 
bazel-bin/tf_main --filename delf_1_ImageOthers_SavedModel.pb --model_name delf_1 --category ImageOthers 
bazel-bin/tf_main --filename faster_rcnn_openimages_v4_inception_resnet_v2_1_ObjectDetection_SavedModel.pb --model_name faster_rcnn_openimages_v4_inception_resnet_v2_1 --category ObjectDetection 
bazel-bin/tf_main --filename openimages_v4_ssd_mobilenet_v2_1_ObjectDetection_SavedModel.pb --model_name openimages_v4_ssd_mobilenet_v2_1 --category ObjectDetection 
bazel-bin/tf_main --filename biggan-deep-128_1_ImageGenerator_SavedModel.pb --model_name biggan-deep-128_1 --category ImageGenerator 
bazel-bin/tf_main --filename biggan-128_2_ImageGenerator_SavedModel.pb --model_name biggan-128_2 --category ImageGenerator 
bazel-bin/tf_main --filename bigbigan-resnet50_1_ImageGenerator_SavedModel.pb --model_name bigbigan-resnet50_1 --category ImageGenerator 
bazel-bin/tf_main --filename efficientnet_b6_feature-vector_1_ImageFeatureVector_SavedModel.pb --model_name efficientnet_b6_feature-vector_1 --category ImageFeatureVector 
bazel-bin/tf_main --filename efficientnet_b7_feature-vector_1_ImageFeatureVector_SavedModel.pb --model_name efficientnet_b7_feature-vector_1 --category ImageFeatureVector 
bazel-bin/tf_main --filename speech_embedding_1_AudioEmbedding_SavedModel.pb --model_name speech_embedding_1 --category AudioEmbedding 
bazel-bin/tf_main --filename compare_gan_model_15_cifar10_resnet_cifar_1_ImageGenerator_SavedModel.pb --model_name compare_gan_model_15_cifar10_resnet_cifar_1 --category ImageGenerator 
bazel-bin/tf_main --filename zfnet512_ImageClassification_FrozenGraph.pb --model_name zfnet512 --category ImageClassification --is_saved_model False
bazel-bin/tf_main --filename yolov2-voc_ObjectDetection_FrozenGraph.pb --model_name yolov2-voc --category ObjectDetection --is_saved_model False
bazel-bin/tf_main --filename vgg19_ImageClassification_FrozenGraph.pb --model_name vgg19 --category ImageClassification --is_saved_model False
bazel-bin/tf_main --filename tinyyolov2_ObjectDetection_FrozenGraph.pb --model_name tinyyolov2 --category ObjectDetection --is_saved_model False
bazel-bin/tf_main --filename super-resolution_ImageSuperResolution_FrozenGraph.pb --model_name super-resolution --category ImageSuperResolution --is_saved_model False
bazel-bin/tf_main --filename shufflenet-v2_ImageClassification_FrozenGraph.pb --model_name shufflenet-v2 --category ImageClassification --is_saved_model False
bazel-bin/tf_main --filename shufflenet_ImageClassification_FrozenGraph.pb --model_name shufflenet --category ImageClassification --is_saved_model False
bazel-bin/tf_main --filename retinanet_ObjectDetection_FrozenGraph.pb --model_name retinanet --category ObjectDetection --is_saved_model False
bazel-bin/tf_main --filename ResNet101-DUC_ImageSegmentation_FrozenGraph.pb --model_name ResNet101-DUC --category ImageSegmentation --is_saved_model False
bazel-bin/tf_main --filename rcnn-ilsvrc13_ImageClassification_FrozenGraph.pb --model_name rcnn-ilsvrc13 --category ImageClassification --is_saved_model False
bazel-bin/tf_main --filename mnist_ImageClassification_FrozenGraph.pb --model_name mnist --category ImageClassification --is_saved_model False
bazel-bin/tf_main --filename googlenet_ImageClassification_FrozenGraph.pb --model_name googlenet --category ImageClassification --is_saved_model False
bazel-bin/tf_main --filename fast_neural_style_transfer_ImageStyleTransfer_FrozenGraph.pb --model_name fast_neural_style_transfer --category ImageStyleTransfer --is_saved_model False
bazel-bin/tf_main --filename emotion-ferplus_EmotionRecognition_FrozenGraph.pb --model_name emotion-ferplus --category EmotionRecognition --is_saved_model False
bazel-bin/tf_main --filename caffenet_ImageClassification_FrozenGraph.pb --model_name caffenet --category ImageClassification --is_saved_model False
bazel-bin/tf_main --filename bvlcalexnet_ImageClassification_FrozenGraph.pb --model_name bvlcalexnet --category ImageClassification --is_saved_model False
bazel-bin/tf_main --filename context_rcnn_resnet101_snapshot_serengeti_2020_06_10_ObjectDetection_FrozenGraph.pb --model_name context_rcnn_resnet101_snapshot_serengeti_2020_06_10 --category ObjectDetection --is_saved_model False
bazel-bin/tf_main --filename rfcn_resnet101_coco_2018_01_28_ObjectDetection_FrozenGraph.pb --model_name rfcn_resnet101_coco_2018_01_28 --category ObjectDetection --is_saved_model False
bazel-bin/tf_main --filename mask_rcnn_resnet50_atrous_coco_2018_01_28_ObjectDetection_FrozenGraph.pb --model_name mask_rcnn_resnet50_atrous_coco_2018_01_28 --category ObjectDetection --is_saved_model False
bazel-bin/tf_main --filename mask_rcnn_inception_v2_coco_2018_01_28_ObjectDetection_FrozenGraph.pb --model_name mask_rcnn_inception_v2_coco_2018_01_28 --category ObjectDetection --is_saved_model False