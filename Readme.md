# How to use
    python main.py <parameters>

    Parameters:
    -f <filepath> : Tests saved model (model.pkt must exist) against specific file given in <filepath>
    --save : Saves model to model.pkt
    --load : Loads model from model.pkt (model.pkt must exist)

# Files
    Main files:
    main.py - Main project using Logistic Regression - Use this file!
    paramaterReader.py - Used in main.py for reading parameters
    
    Other files:
    basicnn.py - Attempted (not complete) Neural Network Design
    colors.py - Logistic Regression where colors are preserved (after flattening 1x30,000)
    nopca.py - Logistic Regression with no PCA step in Preprocessing Pipeline

# Folders
    myex - A few images from the internet scaled/cropped to 100x100 pixels for further testing
    train - contains training/testing data (split by program)

# Data set used
    (https://www.kaggle.com/datasets/sshikamaru/fruit-recognition) - ensure train folder is in root