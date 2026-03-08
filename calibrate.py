# Stereo camera calibration using two Raspberry Pi cameras (libcamera version)
import cv2
import numpy as np
from picamera2 import Picamera2

# Checkerboard settings
CHECKERBOARD = (6, 9)  # (rows, columns) of inner corners
SQUARE_SIZE = 0.025  # meters

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = [[], []]  # imgpoints[0] for cam0, imgpoints[1] for cam1

def open_cameras():
    picam2_0 = Picamera2()
    picam2_1 = Picamera2()

    picam2_0.configure(picam2_0.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
    picam2_1.configure(picam2_1.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))

    picam2_0.start()
    picam2_1.start()

    return picam2_0, picam2_1

def release_cameras(cameras):
    for cam in cameras:
        cam.stop()

def capture_frames(cameras):
    return [cam.capture_array() for cam in cameras]

def show_frames(frames, titles):
    for frame, title in zip(frames, titles):
        cv2.imshow(title, frame)

def capture_checkerboard(frames, gray_frames):
    found = []
    corners = []
    for gray in gray_frames:
        ret, c = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        found.append(ret)
        corners.append(c)
    return found, corners

def refine_and_store(objpoints, imgpoints, found, corners, gray_frames):
    if all(found):
        objpoints.append(objp)
        for i in range(2):
            c2 = cv2.cornerSubPix(gray_frames[i], corners[i], (11,11), (-1,-1), criteria)
            imgpoints[i].append(c2)
        return True
    return False

def draw_corners(frames, found, corners):
    for i in range(2):
        if found[i]:
            cv2.drawChessboardCorners(frames[i], CHECKERBOARD, corners[i], found[i])

def main():
    cameras = open_cameras()
    print("Press 'c' to capture, 'q' to quit and calibrate.")
    titles = ['Camera 0', 'Camera 1']
    while True:
        frames = capture_frames(cameras)
        if any(f is None for f in frames):
            print("Error: Could not read from both cameras.")
            break
        show_frames(frames, titles)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            found, corners = capture_checkerboard(frames, gray_frames)
            if refine_and_store(objpoints, imgpoints, found, corners, gray_frames):
                draw_corners(frames, found, corners)
                show_frames(frames, titles)
                print(f"Captured pair #{len(objpoints)}")
            else:
                print("Checkerboard not found in both frames.")
        elif key == ord('q'):
            break
    release_cameras(cameras)
    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print("Not enough valid pairs for calibration. Need at least 5.")
        exit(1)

    print("Calibrating stereo cameras...")
    gray_shape = imgpoints[0][0].shape[-2::-1]  # (width, height)
    ret0, mtx0, dist0, rvecs0, tvecs0 = cv2.calibrateCamera(objpoints, imgpoints[0], gray_shape, None, None)
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints[1], gray_shape, None, None)
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, mtx0, dist0, mtx1, dist1, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints[0], imgpoints[1],
        mtx0, dist0, mtx1, dist1,
        gray_shape, criteria=criteria, flags=flags
    )
    print("Stereo calibration RMS error:", ret)
    print("Camera 0 matrix:\n", mtx0)
    print("Camera 0 distortion:\n", dist0)
    print("Camera 1 matrix:\n", mtx1)
    print("Camera 1 distortion:\n", dist1)
    print("Rotation between cameras:\n", R)
    print("Translation between cameras:\n", T)
    # Save stereo calibration
    np.savez('stereo_calibration.npz', mtx0=mtx0, dist0=dist0, mtx1=mtx1, dist1=dist1, R=R, T=T, E=E, F=F)
    print("Calibration data saved to stereo_calibration.npz")

    # Save left camera calibration
    np.savez('left_camera_calib.npz', mtx=mtx0, dist=dist0)
    print("Left camera calibration saved to left_camera_calib.npz")

    # Save right camera calibration
    np.savez('right_camera_calib.npz', mtx=mtx1, dist=dist1)
    print("Right camera calibration saved to right_camera_calib.npz")

if __name__ == "__main__":
    main()
