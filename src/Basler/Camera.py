## =========================================================================== ## 
# MIT License
# Copyright (c) 2025 Roman Parak
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
## =========================================================================== ## 
# Author   : Roman Parak
# Email    : Roman.Parak@outlook.com
# Github   : https://github.com/rparak
# File Name: Camera.py
## =========================================================================== ##

# pypylon library for interfacing with Basler cameras
from pypylon import pylon
# Time (Time access and conversions)
import time

class Basler_Cls:
    """
    Description:
        Initialize and configure the Basler camera with a given configuration dictionary.

    Args:
        (1) config [dict]: Dictionary containing camera parameters like exposure time, gain, and balance ratios.
        (2) max_retries [int]: Maximum number of times to retry connecting if the initial attempt fails.
        (3) retry_delay [int]: Delay in seconds between retry attempts.
    """
            
    def __init__(self, config=None, max_retries=10, retry_delay=1):
        self.camera = None; self.tl_factory = None; self.devices = None
        
        # Attempt to connect to the camera.
        self.__Connect(max_retries, retry_delay)

        # Default configuration.
        self.default_config = {
            'exposure_time': 10000,
            'gain': 10,
            'balance_ratios': {'Red': 1.5, 'Green': 1.0, 'Blue': 1.0},
            'pixel_format': 'BayerRG8'
        }

        # Use user-provided config or fall back to default values.
        self.config = config if config else self.default_config

        self.__Configure()

        # Setup converter for Bayer to BGR conversion.
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Initialize the image as None
        self.image = None

    def __Connect(self, max_retries, retry_delay):
        """
        Description:
            A function that attempts to connect to the camera. If it fails, it retries a specified 
            number of times.
        """

        attempt = 0
        while attempt < max_retries:
            try:
                # Attempt to initialize camera components
                self.tl_factory = pylon.TlFactory.GetInstance()
                self.devices = self.tl_factory.EnumerateDevices()
                
                if not self.devices:
                    raise Exception('[ERROR] No Basler cameras detected. Check the connection!')

                # List available devices
                for device in self.devices:
                    print(f'[INFO] Model: {device.GetModelName()}, IP: {device.GetIpAddress()}, Serial: {device.GetSerialNumber()}')

                self.camera = pylon.InstantCamera(self.tl_factory.CreateFirstDevice())

                # Attempt to open the camera connection
                self.camera.Open()
                print(f'[INFO] Connected to: {self.camera.GetDeviceInfo().GetModelName()} '
                      f'({self.camera.GetDeviceInfo().GetIpAddress()})')

                # Exit the loop if connection is successful
                break
            except Exception as e:
                print(f'[WARNING] Connection attempt {attempt + 1} failed: {e}')
                self.camera = None; self.tl_factory = None; self.devices = None
                attempt += 1

                # Pause before attempting to connect again
                time.sleep(retry_delay)
        else:
            # If max_retries is reached without success, raise an error
            raise Exception(f'[ERROR] Failed to connect after {max_retries} attempts.')
        
    def __Configure(self):
        """
        Description:
            Apply camera settings from the provided configuration dictionary.
        """

        self.camera.Gain.SetValue(self.config['gain'])
        # Disable Auto White Balance.
        self.camera.BalanceWhiteAuto.SetValue('Off')

        # Set manual white balance ratios.
        for color, value in self.config['balance_ratios'].items():
            self.camera.BalanceRatioSelector.SetValue(color)
            self.camera.BalanceRatio.SetValue(value)

        # Set Pixel Format.
        self.camera.PixelFormat.SetValue(self.config['pixel_format'])

        # Disable Auto Exposure.
        self.camera.ExposureAuto.SetValue('Off')

        # Set the exposure time in microseconds.
        self.camera.ExposureTime.SetValue(self.config['exposure_time'])
        print(f'[INFO] Exposure Time set to: {self.config["exposure_time"]} µs')

        # The parameter MaxNumBuffer can be used to control the count of buffers
        # allocated for grabbing. The default value of this parameter is 10.
        self.camera.MaxNumBuffer.Value = 5

    def Is_Grabbing(self):
        """
        Description:
            Function to check whether the camera is currently in the process of acquiring images.
        """

        return self.camera.IsGrabbing()
    
    def Capture(self):
        """
        Description:
            Capture a single image and store it in the 'image' class parameter.
        """

        # Check if the camera is currently grabbing.
        if not self.camera.IsGrabbing():
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if result.GrabSucceeded():
            self.image = self.converter.Convert(result).GetArray()
            print(f'[INFO] Captured image with shape: {self.image.shape} and dtype: {self.image.dtype}')
            result.Release()

            return self.image
        else:
            print(f'[ERROR] Error capturing image. Code: {result.ErrorCode}, Description: {result.ErrorDescription}')
            result.Release()

            return None

    def __Close(self):
        """
        Description:
            Release the camera resources.
        """

        if self.camera is not None:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            if self.camera.IsOpen():
                self.camera.Close()
            print('[INFO] Camera resources released.')

    def __del__(self):
        """
        Description:
            Ensure resources are released when the object is destroyed.
        """

        self.__Close()