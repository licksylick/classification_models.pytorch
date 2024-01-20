# Classification models
Project with Neural Networks for Image Classification based on PyTorch. 

-----
## 🔥 Advantages  
* High level API (just few lines to create a neural network)
* All models architectures from timm.
* All models have pre-trained weights for faster and better convergence

## 🚀 Train

### 1. Install all necessary libs:
  ```sh
  pip3 install -r requirements.txt
  ```
Install torch.
Note: if you are using a GPU, then you need to install CUDA and replace the torch version in `requirements.txt` with the GPU-enabled version.
Otherwise, the processor will be used.
-----
### 2. Dataset structure
Directory with 2 subdirectories: `tran_val` with the number of subdirectories equal to num classes and `test` with the number of subdirectories equal to num classes:  
dataset  
 ~~~~
    dataset
     |- train_val
         |- class1
             |- image.jpg
             ...
         |- class2
             |- image.jpg
             ...
     |- test
        |- class1
             |- image.jpg
             ...
         |- class2
             |- image.jpg
             ...
  ~~~~

-----
### 3. Edit `config.yaml`
It has many of setting.  
The most important:
* `path` (in `dataset`) - set path to your data
* `num_classes` (in `model`) - set num_classes appropriate to the task
* `val_size` (in `dataset`) - percentage / 100 for split on train/val set
* `use_cross_validation` (in `common`) - bool value for use crossvalidation teqnique
* `max_epochs` (in `trainer`) - number of epochs to train
* `callbacks` - pytorch-lightning callbacks for your train
*  `arch` (in `model` params) - backbone name supported by [timm](https://github.com/huggingface/pytorch-image-models)
  
To get all supported backbones names use:
```
  timm.list_models()
```
<details>
  <summary>Supported backbones</summary>

| backbone | backbone | backbone | backbone | backbone | backbone | backbone | backbone | backbone |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|
|botnet50ts_256|caformer_b36|caformer_m36|caformer_s18|caformer_s36|cait_m36_384|cait_m48_448|cait_s24_224|cait_s24_384|
|cait_s36_384|cait_xs24_384|cait_xxs24_224|cait_xxs24_384|cait_xxs36_224|cait_xxs36_384|coat_lite_medium|coat_lite_medium_384|coat_lite_mini|
|coat_lite_small|coat_lite_tiny|coat_mini|coat_small|coat_tiny|coatnet_0_224|coatnet_0_rw_224|coatnet_1_224|coatnet_1_rw_224|
|coatnet_2_224|coatnet_2_rw_224|coatnet_3_224|coatnet_3_rw_224|coatnet_4_224|coatnet_5_224|coatnet_bn_0_rw_224|coatnet_nano_cc_224|coatnet_nano_rw_224|
|coatnet_pico_rw_224|coatnet_rmlp_0_rw_224|coatnet_rmlp_1_rw2_224|coatnet_rmlp_1_rw_224|coatnet_rmlp_2_rw_224|coatnet_rmlp_2_rw_384|coatnet_rmlp_3_rw_224|coatnet_rmlp_nano_rw_224|coatnext_nano_rw_224|
|convformer_b36|convformer_m36|convformer_s18|convformer_s36|convit_base|convit_small|convit_tiny|convmixer_768_32|convmixer_1024_20_ks9_p14|
|convmixer_1536_20|convnext_atto|convnext_atto_ols|convnext_base|convnext_femto|convnext_femto_ols|convnext_large|convnext_large_mlp|convnext_nano|
|convnext_nano_ols|convnext_pico|convnext_pico_ols|convnext_small|convnext_tiny|convnext_tiny_hnf|convnext_xlarge|convnext_xxlarge|convnextv2_atto|
|convnextv2_base|convnextv2_femto|convnextv2_huge|convnextv2_large|convnextv2_nano|convnextv2_pico|convnextv2_small|convnextv2_tiny|crossvit_9_240|
|crossvit_9_dagger_240|crossvit_15_240|crossvit_15_dagger_240|crossvit_15_dagger_408|crossvit_18_240|crossvit_18_dagger_240|crossvit_18_dagger_408|crossvit_base_240|crossvit_small_240|
|crossvit_tiny_240|cs3darknet_focus_l|cs3darknet_focus_m|cs3darknet_focus_s|cs3darknet_focus_x|cs3darknet_l|cs3darknet_m|cs3darknet_s|cs3darknet_x|
|cs3edgenet_x|cs3se_edgenet_x|cs3sedarknet_l|cs3sedarknet_x|cs3sedarknet_xdw|cspdarknet53|cspresnet50|cspresnet50d|cspresnet50w|
|cspresnext50|darknet17|darknet21|darknet53|darknetaa53|davit_base|davit_giant|davit_huge|davit_large|
|davit_small|davit_tiny|deit3_base_patch16_224|deit3_base_patch16_384|deit3_huge_patch14_224|deit3_large_patch16_224|deit3_large_patch16_384|deit3_medium_patch16_224|deit3_small_patch16_224|
|deit3_small_patch16_384|deit_base_distilled_patch16_224|deit_base_distilled_patch16_384|deit_base_patch16_224|deit_base_patch16_384|deit_small_distilled_patch16_224|deit_small_patch16_224|deit_tiny_distilled_patch16_224|deit_tiny_patch16_224|
|densenet121|densenet161|densenet169|densenet201|densenet264d|densenetblur121d|dla34|dla46_c|dla46x_c|
|dla60|dla60_res2net|dla60_res2next|dla60x|dla60x_c|dla102|dla102x|dla102x2|dla169|
|dm_nfnet_f0|dm_nfnet_f1|dm_nfnet_f2|dm_nfnet_f3|dm_nfnet_f4|dm_nfnet_f5|dm_nfnet_f6|dpn48b|dpn68|
|dpn68b|dpn92|dpn98|dpn107|dpn131|eca_botnext26ts_256|eca_halonext26ts|eca_nfnet_l0|eca_nfnet_l1|
|eca_nfnet_l2|eca_nfnet_l3|eca_resnet33ts|eca_resnext26ts|eca_vovnet39b|ecaresnet26t|ecaresnet50d|ecaresnet50d_pruned|ecaresnet50t|
|ecaresnet101d|ecaresnet101d_pruned|ecaresnet200d|ecaresnet269d|ecaresnetlight|ecaresnext26t_32x4d|ecaresnext50t_32x4d|edgenext_base|edgenext_small|
|edgenext_small_rw|edgenext_x_small|edgenext_xx_small|efficientformer_l1|efficientformer_l3|efficientformer_l7|efficientformerv2_l|efficientformerv2_s0|efficientformerv2_s1|
|efficientformerv2_s2|efficientnet_b0|efficientnet_b0_g8_gn|efficientnet_b0_g16_evos|efficientnet_b0_gn|efficientnet_b1|efficientnet_b1_pruned|efficientnet_b2|efficientnet_b2_pruned|
|efficientnet_b3|efficientnet_b3_g8_gn|efficientnet_b3_gn|efficientnet_b3_pruned|efficientnet_b4|efficientnet_b5|efficientnet_b6|efficientnet_b7|efficientnet_b8|
|efficientnet_cc_b0_4e|efficientnet_cc_b0_8e|efficientnet_cc_b1_8e|efficientnet_el|efficientnet_el_pruned|efficientnet_em|efficientnet_es|efficientnet_es_pruned|efficientnet_l2|
|efficientnet_lite0|efficientnet_lite1|efficientnet_lite2|efficientnet_lite3|efficientnet_lite4|efficientnetv2_l|efficientnetv2_m|efficientnetv2_rw_m|efficientnetv2_rw_s|
|efficientnetv2_rw_t|efficientnetv2_s|efficientnetv2_xl|efficientvit_b0|efficientvit_b1|efficientvit_b2|efficientvit_b3|efficientvit_l1|efficientvit_l2|
|efficientvit_l3|efficientvit_m0|efficientvit_m1|efficientvit_m2|efficientvit_m3|efficientvit_m4|efficientvit_m5|ese_vovnet19b_dw|ese_vovnet19b_slim|
|ese_vovnet19b_slim_dw|ese_vovnet39b|ese_vovnet39b_evos|ese_vovnet57b|ese_vovnet99b|eva02_base_patch14_224|eva02_base_patch14_448|eva02_base_patch16_clip_224|eva02_enormous_patch14_clip_224|
|eva02_large_patch14_224|eva02_large_patch14_448|eva02_large_patch14_clip_224|eva02_large_patch14_clip_336|eva02_small_patch14_224|eva02_small_patch14_336|eva02_tiny_patch14_224|eva02_tiny_patch14_336|eva_giant_patch14_224|
|eva_giant_patch14_336|eva_giant_patch14_560|eva_giant_patch14_clip_224|eva_large_patch14_196|eva_large_patch14_336|fastvit_ma36|fastvit_s12|fastvit_sa12|fastvit_sa24|
|fastvit_sa36|fastvit_t8|fastvit_t12|fbnetc_100|fbnetv3_b|fbnetv3_d|fbnetv3_g|flexivit_base|flexivit_large|
|flexivit_small|focalnet_base_lrf|focalnet_base_srf|focalnet_huge_fl3|focalnet_huge_fl4|focalnet_large_fl3|focalnet_large_fl4|focalnet_small_lrf|focalnet_small_srf|
|focalnet_tiny_lrf|focalnet_tiny_srf|focalnet_xlarge_fl3|focalnet_xlarge_fl4|gc_efficientnetv2_rw_t|gcresnet33ts|gcresnet50t|gcresnext26ts|gcresnext50ts|
|gcvit_base|gcvit_small|gcvit_tiny|gcvit_xtiny|gcvit_xxtiny|gernet_l|gernet_m|gernet_s|ghostnet_050|
|ghostnet_100|ghostnet_130|ghostnetv2_100|ghostnetv2_130|ghostnetv2_160|gmixer_12_224|gmixer_24_224|gmlp_b16_224|gmlp_s16_224|
|gmlp_ti16_224|halo2botnet50ts_256|halonet26t|halonet50ts|halonet_h1|haloregnetz_b|hardcorenas_a|hardcorenas_b|hardcorenas_c|
|hardcorenas_d|hardcorenas_e|hardcorenas_f|hrnet_w18|hrnet_w18_small|hrnet_w18_small_v2|hrnet_w18_ssld|hrnet_w30|hrnet_w32|
|hrnet_w40|hrnet_w44|hrnet_w48|hrnet_w48_ssld|hrnet_w64|inception_next_base|inception_next_small|inception_next_tiny|inception_resnet_v2|
|inception_v3|inception_v4|lambda_resnet26rpt_256|lambda_resnet26t|lambda_resnet50ts|lamhalobotnet50ts_256|lcnet_035|lcnet_050|lcnet_075|
|lcnet_100|lcnet_150|legacy_senet154|legacy_seresnet18|legacy_seresnet34|legacy_seresnet50|legacy_seresnet101|legacy_seresnet152|legacy_seresnext26_32x4d|
|legacy_seresnext50_32x4d|legacy_seresnext101_32x4d|legacy_xception|levit_128|levit_128s|levit_192|levit_256|levit_256d|levit_384|
|levit_384_s8|levit_512|levit_512_s8|levit_512d|levit_conv_128|levit_conv_128s|levit_conv_192|levit_conv_256|levit_conv_256d|
|levit_conv_384|levit_conv_384_s8|levit_conv_512|levit_conv_512_s8|levit_conv_512d|maxvit_base_tf_224|maxvit_base_tf_384|maxvit_base_tf_512|maxvit_large_tf_224|
|maxvit_large_tf_384|maxvit_large_tf_512|maxvit_nano_rw_256|maxvit_pico_rw_256|maxvit_rmlp_base_rw_224|maxvit_rmlp_base_rw_384|maxvit_rmlp_nano_rw_256|maxvit_rmlp_pico_rw_256|maxvit_rmlp_small_rw_224|
|maxvit_rmlp_small_rw_256|maxvit_rmlp_tiny_rw_256|maxvit_small_tf_224|maxvit_small_tf_384|maxvit_small_tf_512|maxvit_tiny_pm_256|maxvit_tiny_rw_224|maxvit_tiny_rw_256|maxvit_tiny_tf_224|
|maxvit_tiny_tf_384|maxvit_tiny_tf_512|maxvit_xlarge_tf_224|maxvit_xlarge_tf_384|maxvit_xlarge_tf_512|maxxvit_rmlp_nano_rw_256|maxxvit_rmlp_small_rw_256|maxxvit_rmlp_tiny_rw_256|maxxvitv2_nano_rw_256|
|maxxvitv2_rmlp_base_rw_224|maxxvitv2_rmlp_base_rw_384|maxxvitv2_rmlp_large_rw_224|mixer_b16_224|mixer_b32_224|mixer_l16_224|mixer_l32_224|mixer_s16_224|mixer_s32_224|
|mixnet_l|mixnet_m|mixnet_s|mixnet_xl|mixnet_xxl|mnasnet_050|mnasnet_075|mnasnet_100|mnasnet_140|
|mnasnet_small|mobilenetv2_035|mobilenetv2_050|mobilenetv2_075|mobilenetv2_100|mobilenetv2_110d|mobilenetv2_120d|mobilenetv2_140|mobilenetv3_large_075|
|mobilenetv3_large_100|mobilenetv3_rw|mobilenetv3_small_050|mobilenetv3_small_075|mobilenetv3_small_100|mobileone_s0|mobileone_s1|mobileone_s2|mobileone_s3|
|mobileone_s4|mobilevit_s|mobilevit_xs|mobilevit_xxs|mobilevitv2_050|mobilevitv2_075|mobilevitv2_100|mobilevitv2_125|mobilevitv2_150|
|mobilevitv2_175|mobilevitv2_200|mvitv2_base|mvitv2_base_cls|mvitv2_huge_cls|mvitv2_large|mvitv2_large_cls|mvitv2_small|mvitv2_small_cls|
|mvitv2_tiny|nasnetalarge|nest_base|nest_base_jx|nest_small|nest_small_jx|nest_tiny|nest_tiny_jx|nf_ecaresnet26|
|nf_ecaresnet50|nf_ecaresnet101|nf_regnet_b0|nf_regnet_b1|nf_regnet_b2|nf_regnet_b3|nf_regnet_b4|nf_regnet_b5|nf_resnet26|
|nf_resnet50|nf_resnet101|nf_seresnet26|nf_seresnet50|nf_seresnet101|nfnet_f0|nfnet_f1|nfnet_f2|nfnet_f3|
|nfnet_f4|nfnet_f5|nfnet_f6|nfnet_f7|nfnet_l0|pit_b_224|pit_b_distilled_224|pit_s_224|pit_s_distilled_224|
|pit_ti_224|pit_ti_distilled_224|pit_xs_224|pit_xs_distilled_224|pnasnet5large|poolformer_m36|poolformer_m48|poolformer_s12|poolformer_s24|
|poolformer_s36|poolformerv2_m36|poolformerv2_m48|poolformerv2_s12|poolformerv2_s24|poolformerv2_s36|pvt_v2_b0|pvt_v2_b1|pvt_v2_b2|
|pvt_v2_b2_li|pvt_v2_b3|pvt_v2_b4|pvt_v2_b5|regnetv_040|regnetv_064|regnetx_002|regnetx_004|regnetx_004_tv|
|regnetx_006|regnetx_008|regnetx_016|regnetx_032|regnetx_040|regnetx_064|regnetx_080|regnetx_120|regnetx_160|
|regnetx_320|regnety_002|regnety_004|regnety_006|regnety_008|regnety_008_tv|regnety_016|regnety_032|regnety_040|
|regnety_040_sgn|regnety_064|regnety_080|regnety_080_tv|regnety_120|regnety_160|regnety_320|regnety_640|regnety_1280|
|regnety_2560|regnetz_005|regnetz_040|regnetz_040_h|regnetz_b16|regnetz_b16_evos|regnetz_c16|regnetz_c16_evos|regnetz_d8|
|regnetz_d8_evos|regnetz_d32|regnetz_e8|repghostnet_050|repghostnet_058|repghostnet_080|repghostnet_100|repghostnet_111|repghostnet_130|
|repghostnet_150|repghostnet_200|repvgg_a0|repvgg_a1|repvgg_a2|repvgg_b0|repvgg_b1|repvgg_b1g4|repvgg_b2|
|repvgg_b2g4|repvgg_b3|repvgg_b3g4|repvgg_d2se|repvit_m0_9|repvit_m1|repvit_m1_0|repvit_m1_1|repvit_m1_5|
|repvit_m2|repvit_m2_3|repvit_m3|res2net50_14w_8s|res2net50_26w_4s|res2net50_26w_6s|res2net50_26w_8s|res2net50_48w_2s|res2net50d|
|res2net101_26w_4s|res2net101d|res2next50|resmlp_12_224|resmlp_24_224|resmlp_36_224|resmlp_big_24_224|resnest14d|resnest26d|
|resnest50d|resnest50d_1s4x24d|resnest50d_4s2x40d|resnest101e|resnest200e|resnest269e|resnet10t|resnet14t|resnet18|
|resnet18d|resnet26|resnet26d|resnet26t|resnet32ts|resnet33ts|resnet34|resnet34d|resnet50|
|resnet50_gn|resnet50c|resnet50d|resnet50s|resnet50t|resnet51q|resnet61q|resnet101|resnet101c|
|resnet101d|resnet101s|resnet152|resnet152c|resnet152d|resnet152s|resnet200|resnet200d|resnetaa34d|
|resnetaa50|resnetaa50d|resnetaa101d|resnetblur18|resnetblur50|resnetblur50d|resnetblur101d|resnetrs50|resnetrs101|
|resnetrs152|resnetrs200|resnetrs270|resnetrs350|resnetrs420|resnetv2_50|resnetv2_50d|resnetv2_50d_evos|resnetv2_50d_frn|
|resnetv2_50d_gn|resnetv2_50t|resnetv2_50x1_bit|resnetv2_50x3_bit|resnetv2_101|resnetv2_101d|resnetv2_101x1_bit|resnetv2_101x3_bit|resnetv2_152|
|resnetv2_152d|resnetv2_152x2_bit|resnetv2_152x4_bit|resnext26ts|resnext50_32x4d|resnext50d_32x4d|resnext101_32x4d|resnext101_32x8d|resnext101_32x16d|
|resnext101_32x32d|resnext101_64x4d|rexnet_100|rexnet_130|rexnet_150|rexnet_200|rexnet_300|rexnetr_100|rexnetr_130|
|rexnetr_150|rexnetr_200|rexnetr_300|samvit_base_patch16|samvit_base_patch16_224|samvit_huge_patch16|samvit_large_patch16|sebotnet33ts_256|sedarknet21|
|sehalonet33ts|selecsls42|selecsls42b|selecsls60|selecsls60b|selecsls84|semnasnet_050|semnasnet_075|semnasnet_100|
|semnasnet_140|senet154|sequencer2d_l|sequencer2d_m|sequencer2d_s|seresnet18|seresnet33ts|seresnet34|seresnet50|
|seresnet50t|seresnet101|seresnet152|seresnet152d|seresnet200d|seresnet269d|seresnetaa50d|seresnext26d_32x4d|seresnext26t_32x4d|
|seresnext26ts|seresnext50_32x4d|seresnext101_32x4d|seresnext101_32x8d|seresnext101_64x4d|seresnext101d_32x8d|seresnextaa101d_32x8d|seresnextaa201d_32x8d|skresnet18|
|skresnet34|skresnet50|skresnet50d|skresnext50_32x4d|spnasnet_100|swin_base_patch4_window7_224|swin_base_patch4_window12_384|swin_large_patch4_window7_224|swin_large_patch4_window12_384|
|swin_s3_base_224|swin_s3_small_224|swin_s3_tiny_224|swin_small_patch4_window7_224|swin_tiny_patch4_window7_224|swinv2_base_window8_256|swinv2_base_window12_192|swinv2_base_window12to16_192to256|swinv2_base_window12to24_192to384|
|swinv2_base_window16_256|swinv2_cr_base_224|swinv2_cr_base_384|swinv2_cr_base_ns_224|swinv2_cr_giant_224|swinv2_cr_giant_384|swinv2_cr_huge_224|swinv2_cr_huge_384|swinv2_cr_large_224|
|swinv2_cr_large_384|swinv2_cr_small_224|swinv2_cr_small_384|swinv2_cr_small_ns_224|swinv2_cr_small_ns_256|swinv2_cr_tiny_224|swinv2_cr_tiny_384|swinv2_cr_tiny_ns_224|swinv2_large_window12_192|
|swinv2_large_window12to16_192to256|swinv2_large_window12to24_192to384|swinv2_small_window8_256|swinv2_small_window16_256|swinv2_tiny_window8_256|swinv2_tiny_window16_256|tf_efficientnet_b0|tf_efficientnet_b1|tf_efficientnet_b2|
|tf_efficientnet_b3|tf_efficientnet_b4|tf_efficientnet_b5|tf_efficientnet_b6|tf_efficientnet_b7|tf_efficientnet_b8|tf_efficientnet_cc_b0_4e|tf_efficientnet_cc_b0_8e|tf_efficientnet_cc_b1_8e|
|tf_efficientnet_el|tf_efficientnet_em|tf_efficientnet_es|tf_efficientnet_l2|tf_efficientnet_lite0|tf_efficientnet_lite1|tf_efficientnet_lite2|tf_efficientnet_lite3|tf_efficientnet_lite4|
|tf_efficientnetv2_b0|tf_efficientnetv2_b1|tf_efficientnetv2_b2|tf_efficientnetv2_b3|tf_efficientnetv2_l|tf_efficientnetv2_m|tf_efficientnetv2_s|tf_efficientnetv2_xl|tf_mixnet_l|
|tf_mixnet_m|tf_mixnet_s|tf_mobilenetv3_large_075|tf_mobilenetv3_large_100|tf_mobilenetv3_large_minimal_100|tf_mobilenetv3_small_075|tf_mobilenetv3_small_100|tf_mobilenetv3_small_minimal_100|tiny_vit_5m_224|
|tiny_vit_11m_224|tiny_vit_21m_224|tiny_vit_21m_384|tiny_vit_21m_512|tinynet_a|tinynet_b|tinynet_c|tinynet_d|tinynet_e|
|tnt_b_patch16_224|tnt_s_patch16_224|tresnet_l|tresnet_m|tresnet_v2_l|tresnet_xl|twins_pcpvt_base|twins_pcpvt_large|twins_pcpvt_small|
|twins_svt_base|twins_svt_large|twins_svt_small|vgg11|vgg11_bn|vgg13|vgg13_bn|vgg16|vgg16_bn|
|vgg19|vgg19_bn|visformer_small|visformer_tiny|vit_base_patch8_224|vit_base_patch14_dinov2|vit_base_patch14_reg4_dinov2|vit_base_patch16_18x2_224|vit_base_patch16_224|
|vit_base_patch16_224_miil|vit_base_patch16_384|vit_base_patch16_clip_224|vit_base_patch16_clip_384|vit_base_patch16_clip_quickgelu_224|vit_base_patch16_gap_224|vit_base_patch16_plus_240|vit_base_patch16_reg8_gap_256|vit_base_patch16_rpn_224|
|vit_base_patch16_siglip_224|vit_base_patch16_siglip_256|vit_base_patch16_siglip_384|vit_base_patch16_siglip_512|vit_base_patch16_xp_224|vit_base_patch32_224|vit_base_patch32_384|vit_base_patch32_clip_224|vit_base_patch32_clip_256|
|vit_base_patch32_clip_384|vit_base_patch32_clip_448|vit_base_patch32_clip_quickgelu_224|vit_base_patch32_plus_256|vit_base_r26_s32_224|vit_base_r50_s16_224|vit_base_r50_s16_384|vit_base_resnet26d_224|vit_base_resnet50d_224|
|vit_giant_patch14_224|vit_giant_patch14_clip_224|vit_giant_patch14_dinov2|vit_giant_patch14_reg4_dinov2|vit_giant_patch16_gap_224|vit_gigantic_patch14_224|vit_gigantic_patch14_clip_224|vit_huge_patch14_224|vit_huge_patch14_clip_224|
|vit_huge_patch14_clip_336|vit_huge_patch14_clip_378|vit_huge_patch14_clip_quickgelu_224|vit_huge_patch14_clip_quickgelu_378|vit_huge_patch14_gap_224|vit_huge_patch14_xp_224|vit_huge_patch16_gap_448|vit_large_patch14_224|vit_large_patch14_clip_224|
|vit_large_patch14_clip_336|vit_large_patch14_clip_quickgelu_224|vit_large_patch14_clip_quickgelu_336|vit_large_patch14_dinov2|vit_large_patch14_reg4_dinov2|vit_large_patch14_xp_224|vit_large_patch16_224|vit_large_patch16_384|vit_large_patch16_siglip_256|
|vit_large_patch16_siglip_384|vit_large_patch32_224|vit_large_patch32_384|vit_large_r50_s32_224|vit_large_r50_s32_384|vit_medium_patch16_gap_240|vit_medium_patch16_gap_256|vit_medium_patch16_gap_384|vit_medium_patch16_reg4_256|
|vit_medium_patch16_reg4_gap_256|vit_relpos_base_patch16_224|vit_relpos_base_patch16_cls_224|vit_relpos_base_patch16_clsgap_224|vit_relpos_base_patch16_plus_240|vit_relpos_base_patch16_rpn_224|vit_relpos_base_patch32_plus_rpn_256|vit_relpos_medium_patch16_224|vit_relpos_medium_patch16_cls_224|
|vit_relpos_medium_patch16_rpn_224|vit_relpos_small_patch16_224|vit_relpos_small_patch16_rpn_224|vit_small_patch8_224|vit_small_patch14_dinov2|vit_small_patch14_reg4_dinov2|vit_small_patch16_18x2_224|vit_small_patch16_36x1_224|vit_small_patch16_224|
|vit_small_patch16_384|vit_small_patch32_224|vit_small_patch32_384|vit_small_r26_s32_224|vit_small_r26_s32_384|vit_small_resnet26d_224|vit_small_resnet50d_s16_224|vit_so400m_patch14_siglip_224|vit_so400m_patch14_siglip_384|
|vit_srelpos_medium_patch16_224|vit_srelpos_small_patch16_224|vit_tiny_patch16_224|vit_tiny_patch16_384|vit_tiny_r_s16_p8_224|vit_tiny_r_s16_p8_384|volo_d1_224|volo_d1_384|volo_d2_224|
|volo_d2_384|volo_d3_224|volo_d3_448|volo_d4_224|volo_d4_448|volo_d5_224|volo_d5_448|volo_d5_512|vovnet39a|
|vovnet57a|wide_resnet50_2|wide_resnet101_2|xception41|xception41p|xception65|xception65p|xception71|xcit_large_24_p8_224|
|xcit_large_24_p8_384|xcit_large_24_p16_224|xcit_large_24_p16_384|xcit_medium_24_p8_224|xcit_medium_24_p8_384|xcit_medium_24_p16_224|xcit_medium_24_p16_384|xcit_nano_12_p8_224|xcit_nano_12_p8_384|
|xcit_nano_12_p16_224|xcit_nano_12_p16_384|xcit_small_12_p8_224|xcit_small_12_p8_384|xcit_small_12_p16_224|xcit_small_12_p16_384|xcit_small_24_p8_224|xcit_small_24_p8_384|xcit_small_24_p16_224|
|xcit_small_24_p16_384|xcit_tiny_12_p8_224|xcit_tiny_12_p8_384|xcit_tiny_12_p16_224|xcit_tiny_12_p16_384|xcit_tiny_24_p8_224|xcit_tiny_24_p8_384|xcit_tiny_24_p16_224|xcit_tiny_24_p16_384|

</details>


Other:
* `exp_name` - name of your experiment (for new experiments change this name) 
* `trainer` - in `params` you can add arguments for pytorch-lightning trainer
* `model` - in `params` you can change `arch` (supports all CNN models from timm)
* `optimizers` - you can change optimizer from torch.optim
* `scheduler` - you can change scheduler from torch.optim.lr_scheduler
-----

### 4. Run the training script specifying path to your config:

```sh
python3 train.py --config config.yaml
  ```

-----
## ✅ Inference
Run `inference.py`, specifying the required config (which used for training), path to checkpoint and image:
  ```sh
  python3 inference.py --config config.yaml --model_path path/to/model.ckpt --image path/to/image.jpg
  ```
-----
## 📝 BibTex
```
@software{AutoTrackAnything,
  author = {Roman Lyskov},
  title = {AutoTrackAnything},
  year = {2024},
  url = {https://github.com/licksylick/classification_models.pytorch},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
}
```

```
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```


