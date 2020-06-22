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

bazel build tf_main
bazel-bin/tf_main --filename efficientnet_b5_feature-vector_1_ImageFeatureVector_SavedModel.pb --model_name efficientnet_b5_feature-vector_1 --category ImageFeatureVector --is_saved_model True
bazel-bin/tf_main --filename progan-128_1_ImageGenerator_SavedModel.pb --model_name progan-128_1 --category ImageGenerator --is_saved_model True
bazel-bin/tf_main --filename spiral_default-fluid-gansn-celebahq64-gen-19steps_1_ImageRNNAgent_SavedModel.pb --model_name spiral_default-fluid-gansn-celebahq64-gen-19steps_1 --category ImageRNNAgent --is_saved_model True
bazel-bin/tf_main --filename unet_industrial_class_1_1_ImageSegmentation_SavedModel.pb --model_name unet_industrial_class_1_1 --category ImageSegmentation --is_saved_model True
bazel-bin/tf_main --filename vision_embedder_inaturalist_V2_1_ImageFeatureVector_SavedModel.pb --model_name vision_embedder_inaturalist_V2_1 --category ImageFeatureVector --is_saved_model True
bazel-bin/tf_main --filename tweening_conv3d_bair_1_VideoGeneration_SavedModel.pb --model_name tweening_conv3d_bair_1 --category VideoGeneration --is_saved_model True 
bazel-bin/tf_main --filename local-linearity_cifar10_1_ImageClassification_SavedModel.pb --model_name local-linearity_cifar10_1 --category ImageClassification  --is_saved_model True
bazel-bin/tf_main --filename wae-mmd_1_ImageFeatureVector_SavedModel.pb --model_name wae-mmd_1 --category ImageFeatureVector --is_saved_model True
bazel-bin/tf_main --filename videoflow_encoder_1_VideoGeneration_SavedModel.pb --model_name videoflow_encoder_1 --category VideoGeneration --is_saved_model True
bazel-bin/tf_main --filename vae_1_ImageFeatureVector_SavedModel.pb --model_name vae_1 --category ImageFeatureVector --is_saved_model True
bazel-bin/tf_main --filename ganeval-cifar10-convnet_1_ImageClassification_SavedModel.pb --model_name ganeval-cifar10-convnet_1 --category ImageClassification --is_saved_model True
bazel-bin/tf_main --filename wiki40b-lm-en_1_TextLanguageModel_SavedModel.pb --model_name wiki40b-lm-en_1 --category TextLanguageModel --is_saved_model True
bazel-bin/tf_main --filename small_bert_bert_uncased_L-4_H-256_A-4_1_TextEmbedding_SavedModel.pb --model_name small_bert_bert_uncased_L-4_H-256_A-4_1 --category TextEmbedding --is_saved_model True
bazel-bin/tf_main --filename elmo_3_TextEmbedding_SavedModel.pb --model_name elmo_3 --category TextEmbedding --is_saved_model True
bazel-bin/tf_main --filename mil-nce_s3d_1_VideoText_SavedModel.pb --model_name mil-nce_s3d_1 --category VideoText --is_saved_model True
bazel-bin/tf_main --filename LAReQA_mBERT_En_En_1_TextRetrievalQuestionAnswering_SavedModel.pb --model_name LAReQA_mBERT_En_En_1 --category TextRetrievalQuestionAnswering --is_saved_model True
bazel-bin/tf_main --filename delf_1_ImageOthers_SavedModel.pb --model_name delf_1 --category ImageOthers --is_saved_model True