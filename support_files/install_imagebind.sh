#!/bin/bash

# Exit if any command fails
set -e

# Variables
REPO_URL="https://github.com/facebookresearch/ImageBind"
REPO_DIR="ImageBind"
PYTHON_ENV="myenv"

echo "Updating system and installing dependencies..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git

#echo "Creating a virtual environment..."
#python3 -m venv $PYTHON_ENV

echo "Activating virtual environment..."
source ../$PYTHON_ENV/bin/activate

echo "Cloning the ImageBind repository..."
if [ -d "$REPO_DIR" ]; then
   echo "Repository already exists, pulling latest changes..."
   cd $REPO_DIR && git pull
else
   git clone $REPO_URL
   cd $REPO_DIR
fi

echo "Installing required Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installing ImageBind..."
pip install .

echo "ImageBind installation complete."

echo "Deleting the cloned repository..."
cd ..
rm -rf $REPO_DIR

echo "Cloned repository deleted."

echo "To activate the environment, run: source $PYTHON_ENV/bin/activate"
