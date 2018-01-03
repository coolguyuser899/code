##tf_data_example end to end dataset examples from the following link
##https://github.com/jasonbaldridge/try-tf
##updated code so it works for python 3.6

#python packages
python3.6 -m pip install sklearn
python3.6 -m pip install scipy
python3.6 -m pip install matplotlib

#install r for dataset generation
brew tap homebrew/science
brew install r

Rscript --version
R scripting front-end version 3.4.3 (2017-11-30)

#linear data, generate sample data
cd try-tf/simdata
Rscript --version
R scripting front-end version 3.4.3 (2017-11-30)

#create 2 dimension sample data
Rscript generate_linear_data.R    #Rplots.pdf created for plot graph

#moon data generate_moon_data.py
python3.6 -m pip install sklearn
python3.6 -m pip install scipy


python3.6 generate_moon_data.py > moon.txt
cat moon.txt > moon_data_train.csv
Rscript plot_data.R   #create a graph

#saturn data
Rscript generate_saturn_data.R  #change sigma value to see difference
Rscript plot_data.R

#see hyper plane graph
$ R
source('plot_hyperplane.R')
Accuracy: 0.858
python3.6 hidden.py --train simdata/moon_data_train.csv --test simdata/moon_data_eval.csv --num_epochs 100 --num_hidden 5
Accuracy: 0.968

##saturn data
python3.6 softmax.py --train simdata/saturn_data_train.csv --test simdata/saturn_data_eval.csv --num_epochs 100
Accuracy: 0.46

python3.6 hidden.py --train simdata/saturn_data_train.csv --test simdata/saturn_data_eval.csv --num_epochs 100 --num_hidden 15
Accuracy: 1.0

##create log for tensorboard
$ python3.6 annotated_softmax.py --train simdata/linear_data_train.csv --test simdata/linear_data_eval.csv --num_epochs 5 --verbose False
Accuracy: 0.995

$ tensorboard --logdir=try_tf_logs/
Starting TensorBoard b'47' at http://0.0.0.0:6006
