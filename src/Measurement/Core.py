## =========================================================================== ##
# MIT License
# Copyright (c) 2026
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
# Author   : <YOUR NAME>
# Email    : <YOUR EMAIL>
# Github   : https://github.com/rparak
# File Name: Core.py
## =========================================================================== ##

# Numpy (Array computing)
import numpy as np
# Typing (Support for type hints)
import typing as tp
# OpenCV for image processing
import cv2
# Math utilities
import math
# OS for path handling (used when saving results)
import os
# Custom Lib.:
#   ../Measurement/Parameters
from Measurement.Parameters import Object_Dimensions_Str

class Measure_Object_Cls:
    def __init__(self, image: np.ndarray, object_id: int, ref_obj_dim: Object_Dimensions_Str, tolerance: float, conversion_factor: tp.Dict[str, float],
                 min_hole_diameter_mm: float = 0.0, max_hole_diameter_mm: float = float('inf'),
                 hough_min_diameter_mm: tp.Optional[float] = None, hough_max_diameter_mm: tp.Optional[float] = None,
                 hough_param2: float = 47.0, mask_hough_param2: float = 20.0):

        # Input data validation.
        if image is None or isinstance(image, np.ndarray) == False or image.size == 0:
            raise ValueError('[ERROR] Invalid input image.')

        if object_id not in [0, 1]:
            raise ValueError('[ERROR] Invalid object_id (expected: 0 or 1).')

        if conversion_factor is None:
            raise ValueError('[ERROR] Conversion factor is None.')

        if 'x' not in conversion_factor or 'y' not in conversion_factor:
            raise ValueError('[ERROR] Conversion factor must contain keys "x" and "y".')

        if conversion_factor['x'] <= 0.0 or conversion_factor['y'] <= 0.0:
            raise ValueError('[ERROR] Conversion factor values must be > 0.0.')

        # Input data.
        self.__img_in: np.ndarray = image.copy()
        self.__obj_id: int = object_id
        self.__ref_obj_dim: Object_Dimensions_Str = ref_obj_dim
        self.__tolerance: float = tolerance
        self.__conversion_factor: tp.Dict[str, float] = conversion_factor

        # hole size limits (mm)
        if min_hole_diameter_mm < 0 or max_hole_diameter_mm < 0:
            raise ValueError('[ERROR] Hole size limits must be non-negative.')
        if min_hole_diameter_mm > max_hole_diameter_mm:
            raise ValueError('[ERROR] min_hole_diameter_mm cannot exceed max_hole_diameter_mm.')
        self.__min_hole_diameter_mm = min_hole_diameter_mm
        self.__max_hole_diameter_mm = max_hole_diameter_mm

        # optional Hough circle diameter limits (mm); default to hole diameter limits
        if hough_min_diameter_mm is None:
            hough_min_diameter_mm = min_hole_diameter_mm
        if hough_max_diameter_mm is None:
            hough_max_diameter_mm = max_hole_diameter_mm
        if hough_min_diameter_mm < 0 or hough_max_diameter_mm < 0:
            raise ValueError('[ERROR] Hough circle diameter limits must be non-negative.')
        if hough_min_diameter_mm > hough_max_diameter_mm:
            raise ValueError('[ERROR] hough_min_diameter_mm cannot exceed hough_max_diameter_mm.')
        self.__hough_min_diameter_mm = hough_min_diameter_mm
        self.__hough_max_diameter_mm = hough_max_diameter_mm

        # hough parameter for circle detection (param2)
        self.__hough_param2: float = float(hough_param2)
        self.__mask_hough_param2: float = float(mask_hough_param2)

        # Outer counterbore search parameters (local ROI) -- can be adjusted as needed
        self.__outer_roi_half_size = 80
        self.__outer_center_search_radius = 12
        self.__outer_radius_margin_min = 30
        self.__outer_radius_margin_max = 35
        self.__outer_radius_step = 2

        # Output visualization image with drawn measurement results.
        self.__img_out: tp.Optional[np.ndarray] = None

    @property
    def Image(self) -> tp.Optional[np.ndarray]:
        """
        Description:
            Returns the output image with drawn measurement results.

        Returns:
            (1) parameter [np.ndarray or None]: Image with visualization of detected edges, holes, dimensions
                                                and validation result.
                                                Note: None if Solve() was not executed or drawing disabled.
        """

        return self.__img_out

    def _largest_external_contour(self, binary_img: np.ndarray) -> tp.Optional[np.ndarray]:
        """Return largest external contour from a binary image, or None."""
        cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        return max(cnts, key=cv2.contourArea)

    def _crop_with_bounds(self, img: np.ndarray, cx: int, cy: int, half_size: int):
        h, w = img.shape[:2]
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)
        return img[y1:y2, x1:x2].copy(), x1, y1

    def _preprocess_roi(self, roi_bgr: np.ndarray):
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        blackhat = cv2.morphologyEx(gray_blur, cv2.MORPH_BLACKHAT, kernel)

        gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)

        return gray, gray_eq, gray_blur, blackhat, gx, gy

    def _sample_circle_points(self, cx: float, cy: float, r: float, n_points: int = 90):
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        xs = cx + r * np.cos(angles)
        ys = cy + r * np.sin(angles)
        return xs, ys, angles

    def _bilinear_sample(self, img: np.ndarray, xs: np.ndarray, ys: np.ndarray):
        h, w = img.shape[:2]
        x0 = np.floor(xs).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(ys).astype(np.int32)
        y1 = y0 + 1

        valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
        out = np.zeros_like(xs, dtype=np.float32)

        if not np.any(valid):
            return out, valid

        xv0 = x0[valid]
        xv1 = x1[valid]
        yv0 = y0[valid]
        yv1 = y1[valid]

        Ia = img[yv0, xv0].astype(np.float32)
        Ib = img[yv0, xv1].astype(np.float32)
        Ic = img[yv1, xv0].astype(np.float32)
        Id = img[yv1, xv1].astype(np.float32)

        dx = xs[valid] - xv0
        dy = ys[valid] - yv0

        wa = (1 - dx) * (1 - dy)
        wb = dx * (1 - dy)
        wc = (1 - dx) * dy
        wd = dx * dy

        out[valid] = Ia * wa + Ib * wb + Ic * wc + Id * wd
        return out, valid

    def _circle_score(self, cx: float, cy: float, r: float, blackhat: np.ndarray, gx: np.ndarray, gy: np.ndarray, inner_radius: float):
        xs, ys, angles = self._sample_circle_points(cx, cy, r, n_points=220)
        ux = np.cos(angles).astype(np.float32)
        uy = np.sin(angles).astype(np.float32)

        gx_s, valid1 = self._bilinear_sample(gx, xs, ys)
        gy_s, valid2 = self._bilinear_sample(gy, xs, ys)
        valid = valid1 & valid2
        if np.count_nonzero(valid) < 120:
            return -1e9

        radial_grad = gx_s * ux + gy_s * uy
        grad_score = np.mean(radial_grad[valid])

        xs_in, ys_in, _ = self._sample_circle_points(cx, cy, max(1, r - 3), n_points=220)
        bh_s, valid3 = self._bilinear_sample(blackhat, xs_in, ys_in)
        if np.count_nonzero(valid3) < 120:
            return -1e9

        bh_score = np.mean(bh_s[valid3])
        penalty = 0.0
        if r <= inner_radius + 2:
            penalty -= 100.0

        return float(2.5 * grad_score + 1.2 * bh_score + penalty)

    def _detect_outer_circle_in_roi(
        self,
        roi_bgr: np.ndarray,
        approx_center_roi: tp.Tuple[int, int],
        approx_inner_radius: float,
        center_search_radius_px: int,
        radius_margin_min: float,
        radius_margin_max: float,
        radius_step_px: int = 1
    ):
        gray, gray_eq, gray_blur, blackhat, gx, gy = self._preprocess_roi(roi_bgr)
        h, w = gray.shape[:2]
        x0, y0 = approx_center_roi

        best = None
        for dy in range(-center_search_radius_px, center_search_radius_px + 1):
            for dx in range(-center_search_radius_px, center_search_radius_px + 1):
                cx = x0 + dx
                cy = y0 + dy
                if cx < 5 or cy < 5 or cx >= w - 5 or cy >= h - 5:
                    continue

                r_min = max(approx_inner_radius + radius_margin_min, approx_inner_radius + 2)
                r_max = approx_inner_radius + radius_margin_max
                for r in range(int(r_min), int(r_max) + 1, radius_step_px):
                    score = self._circle_score(cx, cy, r, blackhat=blackhat, gx=gx, gy=gy, inner_radius=approx_inner_radius)
                    if best is None or score > best["score"]:
                        best = {"cx": cx, "cy": cy, "r": r, "score": score}
        return best

    def _hough_circles_in_square(
        self,
        image: np.ndarray,
        center: tp.Tuple[int, int],
        half_size: int,
        px_per_mm: float,
    ) -> tp.List[tp.Tuple[int, int, int]]:
        """Detect circles inside a square region (Hough) on the original image."""
        x, y = int(center[0]), int(center[1])
        x0 = max(0, x - half_size)
        y0 = max(0, y - half_size)
        x1 = min(image.shape[1], x + half_size)
        y1 = min(image.shape[0], y + half_size)

        roi = image[y0:y1, x0:x1]
        if roi.size == 0:
            return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 1.5)

        min_r = int(self.__hough_min_diameter_mm / (2.0 * px_per_mm)) if self.__hough_min_diameter_mm > 0 else 0
        max_r = int(self.__hough_max_diameter_mm / (2.0 * px_per_mm)) if math.isfinite(self.__hough_max_diameter_mm) else 0

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=10,
            param1=120,
            param2=self.__hough_param2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None:
            return []

        detected = np.round(circles[0, :]).astype(int)
        out = []
        for cx, cy, r in detected:
            if cx < 0 or cy < 0 or cx >= roi.shape[1] or cy >= roi.shape[0]:
                continue
            out.append((x0 + cx, y0 + cy, int(r)))
        return out

    def Solve(self, draw_result: bool = False) -> tp.Tuple[bool, Object_Dimensions_Str, float, tp.Optional[np.ndarray]]:
        # Initialize result structure.
        Result_Obj_Dimensions = Object_Dimensions_Str()

        # The angle of rotation in degrees.
        angle_of_rotation = 0.0

        # Final validation status.
        # Note:
        #   True = object within tolerance, False = NOK.
        status = False

        # Reset output image.
        self.__img_out = None

        # Get image dimensions
        img = self.__img_in.copy()
        h, w = img.shape[:2]

        # Conversion factor (average of x and y)
        px_per_mm = (self.__conversion_factor['x'] + self.__conversion_factor['y']) / 2.0

        # ------------------------------------------------------------------
        # 1. GREEN FILTER (HSV-based color segmentation)
        # ------------------------------------------------------------------
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Green mask thresholds (tunable for different lighting/camera)
        lower_green = np.array([70, 10, 20], dtype=np.uint8)
        upper_green = np.array([110, 255, 255], dtype=np.uint8)

        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Clean up green mask with morphological operations
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k, iterations=1)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k, iterations=2)

        # ------------------------------------------------------------------
        # 2. OBJECT MASK (inverse of green, i.e., not green = object area)
        # ------------------------------------------------------------------
        not_green = cv2.bitwise_not(green_mask)

        # Remove small noise in not_green
        not_green = cv2.morphologyEx(not_green, cv2.MORPH_OPEN, k, iterations=1)
        not_green = cv2.morphologyEx(not_green, cv2.MORPH_CLOSE, k, iterations=2)

        # Find largest external contour (the object)
        obj_contour = self._largest_external_contour(not_green)
        if obj_contour is None or cv2.contourArea(obj_contour) < 0.01 * (w * h):
            raise RuntimeError(
                '[ERROR] Could not find a valid object contour. '
                'Try tuning the green HSV thresholds or improve lighting/contrast.'
            )

        # Create object mask
        object_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(object_mask, [obj_contour], -1, 255, thickness=cv2.FILLED)

        # Slightly erode object mask to avoid picking up background edges
        object_mask = cv2.erode(object_mask, k, iterations=1)

        # ------------------------------------------------------------------
        # 3. MEASURE OBJECT SIZE AND ANGLE
        # ------------------------------------------------------------------
        rect = cv2.minAreaRect(obj_contour)
        (cx, cy), (w_pix, h_pix), angle = rect

        # Adjust angle so that it describes rotation of the longer side
        if w_pix < h_pix:
            angle = angle + 90.0

        # Treat longer side as "height", shorter as "width"
        long_side = max(w_pix, h_pix)
        short_side = min(w_pix, h_pix)
        height_mm = long_side * px_per_mm
        width_mm = short_side * px_per_mm

        Result_Obj_Dimensions.Height = height_mm
        Result_Obj_Dimensions.Width = width_mm
        angle_of_rotation = angle

        # ------------------------------------------------------------------
        # 4. HOLES MASK (green pixels inside the object)
        # ------------------------------------------------------------------
        holes_mask = cv2.bitwise_and(green_mask, object_mask)

        # Make holes more "circle friendly"
        holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_OPEN, k, iterations=1)
        holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_CLOSE, k, iterations=2)

        # ------------------------------------------------------------------
        # 5. HOLE DETECTION
        # ------------------------------------------------------------------
        hole_centers: tp.List[tp.Tuple[int, int]] = []
        hole_radii: tp.List[float] = []

        if self.__obj_id == 0:
            # FRONT SIDE: Hough circles on the binary holes mask
            holes_blur = cv2.GaussianBlur(holes_mask, (9, 9), 1.5)

            min_r_px = int((self.__min_hole_diameter_mm * px_per_mm) / 2.0) if self.__min_hole_diameter_mm > 0 else 5
            max_r_px = int((self.__max_hole_diameter_mm * px_per_mm) / 2.0) if math.isfinite(self.__max_hole_diameter_mm) else 0

            circles = cv2.HoughCircles(
                holes_blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(20, min(h, w) // 10),
                param1=120,
                param2=self.__mask_hough_param2,
                minRadius=min_r_px,
                maxRadius=max_r_px
            )

            if circles is not None:
                detected = np.round(circles[0, :]).astype(int)
                for x, y, r in detected:
                    if object_mask[y, x] == 0:
                        continue
                    diam_mm_candidate = (r * 2.0) * px_per_mm
                    if diam_mm_candidate < self.__min_hole_diameter_mm or diam_mm_candidate > self.__max_hole_diameter_mm:
                        continue
                    hole_centers.append((int(x), int(y)))
                    hole_radii.append(float(r))

        else:
            # BACK SIDE: contour seeds from holes mask → minEnclosingCircle
            abs_min_r_px = int((self.__min_hole_diameter_mm * px_per_mm) / 2.0) if self.__min_hole_diameter_mm > 0 else 2
            min_area_px = max(50.0, np.pi * (abs_min_r_px ** 2) * 0.1)

            hole_cnts, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hole_cnts = sorted(hole_cnts, key=cv2.contourArea, reverse=True)

            for cnt in hole_cnts:
                if cv2.contourArea(cnt) < min_area_px:
                    break

                (approx_cx, approx_cy), approx_r = cv2.minEnclosingCircle(cnt)
                approx_cx, approx_cy = int(approx_cx), int(approx_cy)

                if not (0 <= approx_cy < h and 0 <= approx_cx < w):
                    continue
                if object_mask[approx_cy, approx_cx] == 0:
                    continue

                diam_approx = (approx_r * 2.0) * px_per_mm
                if diam_approx < self.__min_hole_diameter_mm * 0.4 or diam_approx > self.__max_hole_diameter_mm * 2.5:
                    continue

                hole_centers.append((approx_cx, approx_cy))
                hole_radii.append(float(approx_r))

        # ------------------------------------------------------------------
        # 6. CALCULATE HOLE MEASUREMENTS
        # ------------------------------------------------------------------
        diam_mm = 0.0
        if len(hole_radii) > 0:
            diam_pix = float(np.mean(hole_radii)) * 2.0
            diam_mm = diam_pix * px_per_mm
            if self.__obj_id == 0:
                Result_Obj_Dimensions.Hole_Diameter_Front = diam_mm
            else:
                Result_Obj_Dimensions.Hole_Diameter_Back = diam_mm

        if len(hole_centers) >= 2:
            c1 = np.array(hole_centers[0], dtype=np.float32)
            c2 = np.array(hole_centers[1], dtype=np.float32)
            dist_pix = np.linalg.norm(c1 - c2)
            Result_Obj_Dimensions.Hole_Center_Distance = dist_pix * px_per_mm

        # ------------------------------------------------------------------
        # 6.5. OUTER COUNTERBORE SEARCH AROUND INNER HOLES (front side only)
        # ------------------------------------------------------------------
        outer_circles = []
        if self.__obj_id == 0:
            for (hx, hy), hr in zip(hole_centers, hole_radii):
                roi_bgr, roi_x, roi_y = self._crop_with_bounds(img, hx, hy, self.__outer_roi_half_size)
                approx_center_roi = (hx - roi_x, hy - roi_y)
                best = self._detect_outer_circle_in_roi(
                    roi_bgr=roi_bgr,
                    approx_center_roi=approx_center_roi,
                    approx_inner_radius=hr,
                    center_search_radius_px=self.__outer_center_search_radius,
                    radius_margin_min=self.__outer_radius_margin_min,
                    radius_margin_max=self.__outer_radius_margin_max,
                    radius_step_px=self.__outer_radius_step
                )
                if best is not None:
                    outer_circles.append((best['cx'] + roi_x, best['cy'] + roi_y, best['r'], best['score']))

        # ------------------------------------------------------------------
        # 7. TOLERANCE CHECK AGAINST REFERENCE DIMENSIONS
        # ------------------------------------------------------------------
        status = True
        ref = self.__ref_obj_dim
        tol = self.__tolerance

        if abs(Result_Obj_Dimensions.Height - ref.Height) > tol:
            status = False
        if abs(Result_Obj_Dimensions.Width - ref.Width) > tol:
            status = False
        if self.__obj_id == 0:
            if abs(Result_Obj_Dimensions.Hole_Diameter_Front - ref.Hole_Diameter_Front) > tol:
                status = False
        else:
            if abs(Result_Obj_Dimensions.Hole_Diameter_Back - ref.Hole_Diameter_Back) > tol:
                status = False
        if abs(Result_Obj_Dimensions.Hole_Center_Distance - ref.Hole_Center_Distance) > tol:
            status = False

        # ------------------------------------------------------------------
        # 8. VISUALIZATION (optional)
        # ------------------------------------------------------------------
        if draw_result:
            out = img.copy()

            # PASS / FAIL color
            result_color = (255, 165, 0) if status else (0, 0, 255)

            # Always blue contour
            contour_color = (255, 0, 0)

            # Draw object contour (always blue)
            # cv2.drawContours(out, [obj_contour], -1, contour_color, 2)

            # Draw bounding rectangle (PASS/FAIL color)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(out, [box], -1, result_color, 2)

            # Draw inner holes
            for center, radius in zip(hole_centers, hole_radii):
                cv2.circle(out, center, int(radius), result_color, 2)
                cv2.circle(out, center, 2, result_color, 3)

            # Draw line between centers
            if len(hole_centers) >= 2:
                cv2.line(out, hole_centers[0], hole_centers[1], result_color, 2)

            # Draw outer circles
            for ocx, ocy, oradius, oscore in outer_circles:
                cv2.circle(out, (ocx, ocy), int(oradius), result_color, 2)
                cv2.circle(out, (ocx, ocy), 2, result_color, 3)

            # Text
            # cv2.putText(out, f"H:{height_mm:.2f}mm W:{width_mm:.2f}mm", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # cv2.putText(out, f"HD:{diam_mm:.2f}mm D:{Result_Obj_Dimensions.Hole_Center_Distance:.2f}mm", (10, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # cv2.putText(out, f"Ang:{angle_of_rotation:.1f}deg Stat:{'PASS' if status else 'FAIL'}", (10, 70),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)

            self.__img_out = out
        else:
            self.__img_out = None

        return status, Result_Obj_Dimensions, angle_of_rotation, self.__img_out
