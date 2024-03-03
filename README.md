**README.md**

# OCGAN: Outlier Detection using Conditional Generative Adversarial Networks

This repository contains an implementation of the OCGAN model for outlier detection based on the paper titled "OCGAN: One-Class Novelty Detection Using GANs with Constrained Latent Representations" by Seyed Mohammadi et al. [\[PDF\]](https://arxiv.org/pdf/1903.08550.pdf).

## Overview

OCGAN is a novel approach for outlier detection that utilizes Conditional Generative Adversarial Networks (GANs) with constrained latent representations. It aims to learn a generator that produces both real and synthetic samples while simultaneously learning a discriminator to distinguish between the two classes.


![image](https://github.com/96harsh52/OCGAN_Mnist/assets/36518896/fd40cbc3-c820-4258-b0cf-9ec2a217d2dd)


## How to Run the Code

1. **Install Requirements**: Ensure you have all the necessary dependencies installed by running:
    ```
    pip install -r requirements.txt
    ```

2. **Run the Code**: Execute the `run.sh` script to train the OCGAN model:
    ```
    sh run.sh
    ```

3. **Test the Code**: After training, you can evaluate the model's performance using the `test_ocgan.py` script. The results will be displayed, including any outliers detected.

## Results

The results of running the `test_ocgan.py` script are shown below:

![image](https://github.com/96harsh52/OCGAN_Mnist/assets/36518896/f4bf7a24-0d75-4625-a8e0-cfc34751494d)


## Citation

If you use this code or find the OCGAN model useful in your research, please consider citing the original paper:

```
@article{mohammadi2019ocgan,
  title={OCGAN: One-Class Novelty Detection Using GANs with Constrained Latent Representations},
  author={Mohammadi, Seyed and Rostami, Mahdieh and Ranjbar, Mohsen and Hojjatpanah, Ahmadreza and Bifet, Albert},
  journal={arXiv preprint arXiv:1903.08550},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
