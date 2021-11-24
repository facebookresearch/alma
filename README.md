# On Anytime Learning At Macroscale

Learning from sequential data dumps </br>

## (key) Requirements 
- Python 3.7
- Pytorch 1.9.0
- Hydra 1.1.0 (`pip install hydra-core & pip install hydra-submitit-launcher`)

## Structure

    ├── crlapi           
      ├── benchmark.py    # Creates the data stream, feeds it to the model and evaluates it
      ├── core.py         # Abstract classes for 
      ├── logger.py   
      ├── sl
        ├── architectures
          ├── ...         # NN architectures used in this project
        ├── clmodels
          ├── ...         # Models (e.g. Single, gEns, ..., )
        ├── streams
          ├── ...         # CIFAR and MNIST stream implementatins

## Running Experiments

To run experiments, you need to call the dataset specific run file, and you need to pass the configuration of the run. We have place the configurations in the previous directory (`../configs`). The config structure is as follows 


        ├── configs
            ├── mnist
               ├── run.py                 # run file
               ├── test_usage_gmoe.yaml   # This is the "gMoE" model
               ├── test_finetune_mlp.yaml # This is the "Single Model"
               ... 
            ├── cifar
               ├── run.py                 # run file
               ├── test_finetune_vgg.yaml # This is the "Single Model"
               ├── test_usage_gmoe.yaml   # This is the "gMoE" model
               ...
               
To run an e.g. mnist gMoE run, the command is (launched from the directory just above (so `cd ..`)
```
PYTHONPATH=./ python configs/mnist/run.py -cn test_usage_gmoe n_megabatches=2 replay=1 clmodel.max_epochs=200 
```

## Important arguments

`n_megabatches` : controls the number of megabatches. So `n_megabatches=1` is your regular full dataset training </br>
`replay` : whether to use replay or not </br>
`clmodel.init_from_scratch` : whether to reinitialize the model at every MB. Should only be used when `replay=1` </br>
`device` : use `cuda` or `cpu` depending on your hardware
