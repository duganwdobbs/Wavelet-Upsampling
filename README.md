# Wavelet-Upsampling
Testing learning wavelet features for image upsampling

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

This base directory will have several folders, 
  - ./network_log : This is where the saved model and logs are stored
  - ./Data/       : This is where the downloaded dataset will be stored
  - ./Code/       : This is where the code will be stored
