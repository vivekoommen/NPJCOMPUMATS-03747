# Learning two-phase microstructure evolution using neural operators and autoencoder architectures - [Link](https://rdcu.be/cURNI)

## Abstract

Phase-field modeling is an effective but computationally expensive method for capturing the mesoscale morphological and microstructure evolution in materials. Hence, fast and generalizable surrogate models are needed to alleviate the cost of computationally taxing processes such as in optimization and design of materials. The intrinsic discontinuous nature of the physical phenomena incurred by the presence of sharp phase boundaries makes the training of the surrogate model cumbersome. We develop a framework that integrates a convolutional autoencoder architecture with a deep neural operator (DeepONet) to learn the dynamic evolution of a two-phase mixture and accelerate time-to-solution in predicting the microstructure evolution. We utilize the convolutional autoencoder to provide a compact representation of the microstructure data in a low-dimensional latent space. After DeepONet is trained in the latent space, it can be used to replace the high-fidelity phase-field numerical solver in interpolation tasks or to accelerate the numerical solver in extrapolation tasks.

## Schematic representation of DeepONet with convolutional autoencoder

![alt text](algorithm.png?raw=true)

## Citation

    @article{Oommen2022,
        author={Oommen, Vivek
            and Shukla, Khemraj
            and Goswami, Somdatta
            and Dingreville, R{\'e}mi
            and Karniadakis, George Em},
        title={Learning two-phase microstructure evolution using neural operators and autoencoder architectures},
        journal={npj Computational Materials},
        year={2022},
        volume={8},
        number={1},
        pages={190}
        }


