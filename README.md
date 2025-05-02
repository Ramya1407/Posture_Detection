## Posture Detection

<img width="324" alt="TN" src="https://user-images.githubusercontent.com/101906945/204119817-f6b19fa1-9890-4a67-8aeb-3b37161a269f.png"><img width="324" alt="TN2" src="https://user-images.githubusercontent.com/101906945/204119820-cf92690c-3703-42b0-b596-f7cddb25b570.png">

This project is a machine learning model that can identify good sitting posture and bad sitting posture with both images and video. The model is trained from a dataset consisting of only 100 images of bad posture and 100 images of good posture, all of which are taken from a webcam. This model is designed to run on the Jetson Nano. 

## Installation

 - Install Python 3 on the Jetson Nano
 - Clone the jetson-inference project
 
```bash
git clone --recursive https://github.com/dusty-nv/jetson-inference
```

 - Build the jetson-inference project
 
```bash
cd jetson-inference
mkdir build
cd build
cmake ../
```

 > [!NOTE]
 > Leave everything as is during build

 - Clone this repository
 ```bash
 git clone https://github.com/SomedudeX/Nvidia-Project
 ```
 
 - Start the docker container
 
 ```bash
 cd ~/jetson-inference
 ./docker/run.sh
 ```


## Image Recognition


The following command runs the model on stationary images. Note that the model is trained to take the live feed from your webcam, hence stationary images will have higher error rates than the live feed coming out of a webcam. Use video recognition for the most optimal result. 

  ```bash
imagenet.py --model=<RESNET18.ONNX> --input_blob=input_0 --output_blob=output_0 \
--labels=<LABELS.TXT> <INPUT IMAGE> <OUTPUT IMGAE>
  ```

 > [!NOTE]
 > `<RESNET18.ONNX>` and `<LABELS.TXT>` should be paths to the files downloaded from this repo. `<INPUT IMAGE>` and `<OUTPUT IMAGE>` should be the paths to the image you wish to run the model on and the desired output path. 
 

## Video recognition

The following commands runs the model on live feed from a webcam attached to the Nano. The Nano captures and processes the feed from the webcam and uses RTP to send it to your host machine to view. 

 - Check the host computer's local IP address. 
 - Connect a webcame to the Nano and check its device number.
   
   ```bash
   ls -ltrh /dev/video*
   ```
   
 - Run the command on your Nano to use the model along with RTP. 

    ```bash
    imagenet.py --model=<RESNET18.ONNX> --input_blob=input_0 --output_blob=output_0 \
    --labels=<LABELS.TXT> /dev/video<DEVICE NUMBER> rtp://<IP ADDRESS>:1234
    ```
> [!NOTE]
> `<RESNET18.ONNX>` and `<LABELS.TXT>` should be paths to the files downloaded from this repo. `<DEVICE NUMBER>` and `<IP ADDRESS>` should be the ones from step 2 and 1 respectively. 

Now that the Jetson Nano is broadcasting live video data, we are ready to receive the data from the host computer. There are two ways to do so. One is with GStreamer and the other is with VLC Media player. Using GStreamer is recommended as its latency is much lower than VLC. Use the following commands (taken from [this file](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#rtp)) on the host computer to receive the footage. 
    

   - Using GStreamer:
     - Install [GStreamer](https://gstreamer.freedesktop.org/documentation/installing/index.html) and run this pipeline
  	
      ```bash
      $ gst-launch-1.0 -v udpsrc port=1234 \
      caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! \
      rtph264depay ! decodebin ! videoconvert ! autovideosink
      ```
    	
   - Using VLC Player:
     - Create a SDP file (.sdp) with the following contents
    	
      ```
      c=IN IP4 127.0.0.1
      m=video 1234 RTP/AVP 96
      a=rtpmap:96 H264/90000
      ```
	
     - Open the stream in VLC by double-clicking the SDP file
     - You may want to reduce the `File caching` and `Network caching` settings in VLC as [shown here](https://www.howtogeek.com/howto/windows/fix-for-vlc-skipping-and-lagging-playing-high-def-video-files/)
	
	
## Demo and Resources

 **[Image Recognition Demo](https://youtu.be/Y6P_PTaILX0)**
 
 **[Video Classification Demo](https://youtu.be/sSHaRQRecs8)**
 
 **[Furthur reading](https://github.com/dusty-nv/jetson-inference/)**
 
