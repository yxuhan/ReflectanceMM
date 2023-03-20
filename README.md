# ReflectanceMM

This is a PyTorch implementation of the following paper:

**Learning a 3D Morphable Face Reflectance Model from Low-cost Data**, CVPR 2023.

Yuxuan Han, Zhibo Wang and Feng Xu

[Project Page](https://yxuhan.github.io/ReflectanceMM/index.html) | [Paper]()

<img src="misc/ReflectanceMM.png" width="100%" >

**Abstract**: *Modeling non-Lambertian effects such as facial specularity leads to a more realistic 3D Morphable Face Model. Existing works build parametric models for diffuse and specular albedo using Light Stage data. However, only diffuse and specular albedo cannot determine the full BRDF. In addition, the requirement of Light Stage data is hard to fulfill for the research communities. This paper proposes the first 3D morphable face reflectance model with spatially varying BRDF using only low-cost publicly-available data. We apply linear shiness weighting into parametric modeling to represent spatially varying specular intensity and shiness. Then an inverse rendering algorithm is developed to reconstruct the reflectance parameters from non-Light Stage data, which are used to train an initial morphable reflectance model. To enhance the model's generalization capability and expressive power, we further propose an update-by-reconstruction strategy to finetune it on in-the-wild datasets. Experimental results show that our method obtains decent face-rendering results with plausible specularities.*
