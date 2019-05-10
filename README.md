# Wavelet-Upsampling
Testing learning wavelet feature GANs for image upsampling. No dedicated dataset, instead, harvests random images from Google Image search.

Relies on several helper libraries, mostly for downloading of random images. 

https://github.com/hardikvasa/google-images-download
This is a script for downloading google images. There are others, but this is the most verbose and expandable. Most specifically,
  it allows for Google's safe search. This makes EVERYTHING EASIER.
  
https://github.com/nmoya/random-flickr
This is currently unused, but could be used to download random Flickr images. Currently deciding against Flickr, as each image is owned /
  copyrighted by the uploader, and isn't exactly usable in a research setting. The question to who owns the network that was trained on 
  copyrighted data won't be answered by me.
  
https://github.com/JeremyLuna/UTIL_wavelets
This is used for its wavelet recomposition.

This is Dugan's Wavelet Upsampling code. In this base directory, you will find several folders.

Files: Code/              : This is the directory for the code to run the upsampling network.
       Data/              : This directory is where the downloaded data is stored. 
       network_log/       : This is where network saved states and logs are saved.
       UpsampleTesting/   : This is a working directory used to evaluate WavSampling vs other methods
       README.md          : This file!
       test/train/val.lst : Temporary files to hold testing and training splits 



To run the network, run: python3 Code/UpsampleRunner.py. This will automatically start the data downloader, 