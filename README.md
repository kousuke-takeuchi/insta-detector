# Instagram Object Detector

This software automatically downloads user profile and the posts uploaded to Instagram, then detect some objects.

## Usage

Run following command with argument: instagram username and password.

1. download profile and post images from Instagram. (default destination is `.instagram`)
2. analyze image and detect objects.
3. draw detected object label and bbox rectangle in the image.
4. save label drawn images (default to current directory).

```
$ python app.py --login.user=<your username> --login.password=<your password>
```
