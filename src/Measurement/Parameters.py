## =========================================================================== ##
# MIT License
# Copyright (c) 2026 Roman Parak
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
# File Name: Parameters.py
## =========================================================================== ##

# Dataclasses (Data Classes)
from dataclasses import dataclass

@dataclass
class Object_Dimensions_Str:
    """
    Description:
        The structure of the main parameters of the measured object.

    Units:
        All values are in millimeters [mm]
    """

    # Object size.
    Height: float = 0.0
    Width: float = 0.0

    # Hole parameters.
    Hole_Diameter_Back: float = 0.0
    Hole_Diameter_Front: float = 0.0
    Hole_Center_Distance: float = 0.0

"""
Description:
    Reference object dimensions (ground truth).

    Object Dimensions:
        - Height: 60 mm
        - Width: 40 mm
        - Hole Diameter (Back side): 6 mm
        - Hole Diameter (Front side): 12 mm
        - Distance between holes (center-to-center): 25 mm
"""
Reference_Obj_Dimensions = Object_Dimensions_Str()
Reference_Obj_Dimensions.Height = 60.0
Reference_Obj_Dimensions.Width  = 40.0
Reference_Obj_Dimensions.Hole_Diameter_Back   = 6.0
Reference_Obj_Dimensions.Hole_Diameter_Front  = 6.0
Reference_Obj_Dimensions.Hole_Center_Distance = 25.0
