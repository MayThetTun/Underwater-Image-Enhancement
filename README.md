# LIGHTWEIGHT UNDERWATER IMAGE ENHANCEMENT VIA IMPULSE RESPONSE OF LOW-PASS FILTER-BASED ATTENTION NETWORK
May Thet Tun, Yosuke Sugiura, Tetsuya Shimamura 
ICIP 2024 Accepted Paper

we propose an improved model of Shallow-UWnet for underwater image enhancement. In the proposed method, we enhance the learning process and solve the vanishing gradient problem by a skip connection, which concatenates the raw underwater image and the impulse response of the low-pass filter (LPF) into Shallow-UWnet. Additionally, we integrate the simple,  parameter-free attention module (SimAM) into each Convolution Block to enhance the visual quality of images. Performance evaluations with state-of-the-art methods show that the proposed method has comparable results on EUVP-Dark, UFO-120, and UIEB datasets. Moreover, the proposed model has fewer trainable parameters and the resulting faster testing time is suitable for real-time processing in underwater image enhancement, which is particularly for resource-constrained underwater robots.

Dataset Information
In this experiment, we utilized 3500 pairs of images from EUVP-ImageNet for the training, while the remaining 200 image pairs were used for validation. The testing datasets are as follows. EUVP-Dark  dataset includes 5,500 paired images that capture dark-hazed underwater scenes. For testing, 1000 images were used in accordance with Shallow-UWnet model. UFO-120 dataset captures high-quality images during oceanic explorations. Distorted images in UFO-120 dataset were created using style transfer, and 120 paired images were utilized as a benchmark to evaluate testing datasets. The UIEB dataset includes a collection of 890 real underwater images. The dataset comprises a variety of distortion levels and different light conditions,
with a range of colors and contrast levels . The reference images in UIEB dataset are free from color casts and display accurate colors.


