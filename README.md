## Setup environment to run problem set examples:
# Create a new environment named 'finsignal' with Python 3.10
conda create -n finsignal python=3.10

# Activate the environment
conda activate finsignal

# Install core scientific and financial packages
conda install numpy pandas scipy matplotlib statsmodels scikit-learn jupyter

# Install financial-specific packages
conda install -c conda-forge pandas-datareader yfinance pywavelets

# Install visualization enhancements
conda install seaborn plotly

# Install Jupyter notebook extensions (optional but helpful)
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Launch Jupyter Notebook
jupyter notebook
