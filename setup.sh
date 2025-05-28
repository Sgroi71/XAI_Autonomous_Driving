
############################################################################################################
# Fabio local run (wsl ubuntu)
# clone and change folder
git clone https://github.com/Sgroi71/XAI_Autonomous_Driving.git
cd ./XAI_AUTONOMOUS_DRIVING
# Create and activate venv
python3 -m venv venv
source venv/bin/activate
# install gdown and download test dataset 
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1tHUZ4YnrkMxsxhchMZ6Jt56nNaG9LcBp -O ./dataset/road
# download train-val dataset
gdown https://drive.google.com/uc?id=1YQ9ap3o9pqbD0Pyei68rlaMDcRpUn-qz
# unzip in ./dataset/road
unzip -q ./dataset/road.zip -d ./dataset/road
# Install requirements to process dataset (ffmpeg)
sudo apt-get install ffmpeg
# Process dataset
python 3D-RetinaNet/extras/extract_videos2jpgs.py ./dataset/road/

# Download kinetics weights for the CNN backbone.
cd 3D-RetinaNet/kinetics-pt/
sh get_kinetics_weights.sh
cd ../..

# install dependencies
pip install torch torchvision
pip install scipy
pip install pyyaml
pip install easydict

cd ./dataset/road
gdown https://drive.google.com/uc?id=1uoyBiNZq1_SHif1CG_2R6d_pUWwUe7fL
cd ../..

# Launch this first time to creat exp folder
python 3D-RetinaNet/main.py /home/fabio/dev/XAI_Autonomous_Driving/dataset/ /home/fabio/dev/XAI_Autonomous_Driving/ /home/fabio/dev/XAI_Autonomous_Driving/3D-RetinaNet/kinetics-pt/ --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041

# Download road-pt weights
gdown https://drive.google.com/uc?id=1OyCpQWnHOWL4WT9Bvic3iqyHl48o0R_R
# Move model_000030_resnet50I3D.pth to the exp folder
mv ./model_000030_resnet50I3D.pth ./road/cache/resnet50I3D512-Pkinetics-b4s8x1x1-roadt3-h3x3x3/model_000030.pth

# Run inference

python 3D-RetinaNet/main.py /home/fabio/dev/XAI_Autonomous_Driving/dataset/ /home/fabio/dev/XAI_Autonomous_Driving/ /home/fabio/dev/XAI_Autonomous_Driving/3D-RetinaNet/kinetics-pt/ --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041
