# Pytorch Generative Diffusion models 
Hi guys I needed a Pytorch version of the repo by Sohl-Dickstein at:
https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models

Which is an implementation of:<br>
> Deep Unsupervised Learning using Nonequilibrium Thermodynamics<br>
> Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli<br>
> International Conference on Machine Learning, 2015<br>
> http://arxiv.org/abs/1503.03585

You see it was written using theano which is no longer supported and I wanted a Pytorch version for myself 
so I thought it is definitely good to share it with others. 

There are two commands to use. First executing train.py to train your beloved model, then infer.py to do inpainting, denoising,
and generation again using your beloved trained model.

All you need is to have all .py files in one directory and give it the name of the desired output directory.
A diffusion folder will be created in your directory where all the data, model, and inference outputs go.

for example to train a model on MNIST using GPU cuda:0 and save it with a suffix "500" in a directory called omicron use the command:

``python /path/to/train.py -dir '/vol/world/omicron/' --dataset MNIST --device cuda:0 --saveName 500``

If no saveName is given the default will be used potentially overwriting previous trained models.

To resume training a model saved in omicron directory with the suffix 500 on mnist and use GPU:1 instead, and save it with a new suffix 800 just write:

``python /path/to/train.py -dir '/vol/world/omicron/' --device cuda:1 --saveName 800 --resume_file yes --loadName 500``

Everything else will be loaded from the saved model. You can also input training parameters and model parameters otherwise default values are used.
Not supplying a device will train using cpu. Good luck having your model ready before the end of the century. 

To use a trained model (e.g.  the one above with suffix 800) to generate samples, and save results using the suffix 5 just run:

``python /path/to/infer.py -dir '/vol/world/omicron/' --dataset MNIST  --loadName 800 --savename -5 --device cpu``

Unfortunately you have to supply the --dataset argument or an error will occur because I am lazy to coreect the code.

To do denoising, a number of examples should be provided as follows (using the same numbers will produce same samples froms the same set determinisitic):

``python /path/to/infer.py -dir '/vol/world/omicron/' --dataset MNIST  --loadName 800 --savename -5 --noise 0.5 --examples '10,4,2,5'``

This will do denoising of training set examples number 10,4,2,5 using a noise Sd of 0.5 and will save them using a suffix of 5

To do left half inpainting ON THE TEST SET:

``python /path/to/infer.py -dir '/vol/world/omicron/' --dataset MNIST --split test  --loadName 800 --savename -5 --inpaint 0 --examples '10,4,2,5'``

To repeat the previous thing but using a random inpainting mask that well randomly inpaints 50% of the pixels use --inpaint 1

``python /path/to/infer.py -dir '/vol/world/omicron/' --dataset MNIST --split test  --loadName 800 --savename -5 --inpaint 1 --examples '10,4,2,5'``

**Contact- ORIGINAL AUTHOR** - The original author (sohl, link to repo above) would love to hear from you. Let him know what goes right/wrong! <jaschasd@google.com>
**Contact- miro** - Now is my turn. I would love to hear from you. Let me know what goes right/wrong! <morssyamr@myvuw.ac.nz>
