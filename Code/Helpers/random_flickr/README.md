random-flickr
=============

Using the Python FlickrAPI (made by Sybren Stüvel), this script randomly downloads a specified amount of photos among all photos on your Flickr account. 
This was made for the purpose of setting your screensaver as your wallpaper in MacOS. Since some screensavers allow an image folder as input, you could have a sliding tiles from your Flickr photos as your desktop wallpaper.
- Example: https://www.youtube.com/watch?v=Y19k92YCs0g

## Requirements
- Python 2.*
- Flickr user_id (See References)
- Flickr Appkey (See References)

## Usage
- Fork this repository.
- Rename the example-config.json to config.json.
- Set all the variables in the config file with your own values.
- Run: python app.py

## Comments
- The first execution takes a while, since it stores the link to all your images. Be patient.
- The app_folder must be empty and created only for this purpose. All files are removed at each exection. Do NOT leave other images in the folder besides the ones downloaded from the app.
- At each execution, all images are removed and the specified amount of images in the config.json file are randomly downloaded.
- Have fun!


## Future work
- Download the images only from an specified set.
- Detect when new images are uploaded and recreated the local file with the urls.
- Parallel execution on flickr_obj.walk_set method.

## References

- Request an Flickr Appkey: https://www.flickr.com/services/apps/create/apply/
- Find out your user_id:    http://idgettr.com/
- Flickr API Documentation: http://stuvel.eu/media/flickrapi-docs/documentation/
