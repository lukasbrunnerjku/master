import os
import json
import cv2  
import numpy as np
import argparse
import math
from glob import glob
from typing import List, Tuple
from tqdm import tqdm
from multiprocessing import Pool
import logging

class VideoReader:
    def __init__(self, file_name, args):
        self.file_name = file_name  # path to viedeo file or camera id
        self.args = args
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        if not self.cap.isOpened():
            raise IOError('Video: {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

def capture_images(path: str, args):
    delay = 1
    esc_code = 27  # escape key
    path += '/{:02d}.png' 
    
    #import pdb; pdb.set_trace()
    
    try:
        video = int(args.video)  # camera id
    except ValueError:  # video path
        pass
    
    cap = cv2.VideoCapture(video)

    # jelly comb webcam:
    # at 30 fps resolutions are
    # 1920x1080, 1600x896, 640x480, 320x240
    # resize with cv2.resize to get arbitrary resolution
    frameRate = 30
    res = '1920x1080, 1600x896, 640x480, 320x240'.split(',')
    res = [list(map(int, r.strip().split('x'))) for r in res]
    
    assert [args.width, args.height] in res, 'Resolution not supported!'
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise Exception('Could not open video device...')

    frameId = 0
    while(cap.isOpened()):
        # cv2.getTickCount() yields number of clock-cycles since reference event
        # cv2.getTickFrequency() yields number of clock-cycles per second
        t = cv2.getTickCount()
        
        ret, frame = cap.read()
        if not ret:
            break 

        frameId += 1
        if frameId % int(frameRate) == 0:  # ~1 fps
            cv2.imwrite(path.format(int(frameId)), frame)
        
        t = (cv2.getTickCount() - t) / cv2.getTickFrequency()  # in s
 
        cv2.putText(frame, f'FPS: {1/t:.1f}', 
            (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.imshow('Video', frame)

        key = cv2.waitKey(delay)
        if key == esc_code:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def read_images(paths: List[str]):
    return np.asarray([cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths])

def search_corners(images: np.ndarray, checker_grid: Tuple[int, int]):
    image_points = []
    mask = []
    # search for checkerboard corners
    for idx, image in enumerate(tqdm(images, desc='Search for grid points...')):
        found, corners = cv2.findChessboardCorners(image, checker_grid)
        mask.append(found)
        if found:  # refine corners on subpixel level
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
            corners.reshape(-1, 2)
            image_points.append(corners)
             
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.drawChessboardCorners(image, checker_grid, corners, found)
            cv2.imwrite(f'debug/{idx:02d}.png', image)

    return np.asarray(mask, dtype=np.bool), np.asarray(image_points).squeeze(2)

def delete_images(folder_path):
    for image in glob(os.path.join(folder_path, '*.png')):
        os.remove(image)

def get_world_points(image_points, checker_size, checker_grid):
    image_points = np.asarray(image_points)
    valid_checkerboard_views, num_corners = image_points.shape[:2]
    world_points = np.zeros((valid_checkerboard_views, num_corners, 3))
    # create xy world points
    world_corners2D = (checker_size * 
        np.mgrid[:checker_grid[0], :checker_grid[1]].T.reshape(-1, 2))
    world_points[..., :2] = world_corners2D  # z coord. = 0
    return world_points.astype(np.float32)

def calibrate(
    images: np.ndarray,
    checker_grid: Tuple[int, int], 
    checker_size: float,  
    ):
    mask, image_points = search_corners(images, checker_grid)
    world_points = get_world_points(image_points, checker_size, checker_grid)

    logging.info('Calibrating camera...')
    # points: float32 and shapes of valid_views x num_corners x 2 or 3
    err, K, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, 
        image_points, images[0].shape, None, None)
    logging.info(f'Reprojection error: {err}')  # typ. <1
    summarize_intrinsics(K)
    np.savez('calib.npz', err=err, K=K, dist=dist, rvecs=rvecs, tvecs=tvecs)

    # cv2.calibrateCamera will return rotation vectors, 
    # which is another way of representing a rotation
    Rs = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]  # matrices
    return np.asarray(K), dist, np.asarray(Rs), np.asarray(tvecs), mask

def load_calibration_results(path: str = 'calib.npz'):
    npz = np.load(path)
    err = npz['err']
    dist = npz['dist'] 
    rvecs = npz['rvecs'] 
    tvecs = npz['tvecs']
    K = npz['K']
    logging.info(f'Loaded calibration results with reprojection error: {err}')
    return err, K, dist, rvecs, tvecs

def summarize_intrinsics(K):
    logging.info(f'Intrisic matrix: \n{K}')
    fx = K[0, 0]; fy = K[1, 1]; skew = K[0, 1]
    cx = K[0, 2]; cy = K[1, 2]
    logging.info(f'focal length in x: {fx}')
    logging.info(f'focal length in y: {fy}')
    logging.info(f'physical axis skewness: {skew}')
    logging.info(f'optical axis displacement in x: {cx}')
    logging.info(f'optical axis displacement in y: {cy}')
    return fx, fy, skew, cx, cy

def save_extrinsics(R, tvec, path='data/extrinsics.json'):
    assert R.shape == (3, 3)
    assert tvec.shape == (3, 1)
    extrinsics = {'R': R.tolist(), 't': tvec.tolist()}
    json.dump(extrinsics, fp=open(path, '+w'))

def reprojection_error(image_points, world_points, rvec, tvec, K, dist):
    # calculate reprojection error by using extrinsic parameters to project
    # the 3D world points into image space and measuring distance of points
    repr_points, jacobian = cv2.projectPoints(world_points, rvec, tvec, K, dist)
    repr_points = repr_points.reshape(-1, 2)
    total_error = np.sum(np.abs(image_points - repr_points)**2)
    total_points = len(world_points)
    err = np.sqrt(total_error/total_points)

def get_extrinsics(images, checker_grid, checker_size):
    # use multiple images to search for corners, on success
    # those corners will be used for extrinsics calculation
    mask, image_points = search_corners(images, checker_grid)
    assert mask.any(), 'Cannot find checkerboard in image.'
    image_points = image_points[mask, ...]  
    _, K, dist, _, _ = load_calibration_results()
        
    # 3D points of checkerboard corners: num_corners x 3
    world_points = get_world_points(image_points, checker_size, checker_grid)[0]

    # 2D points of checkerboard corners on image plane: num_corners x 2
    image_points = image_points[0]  # use first valid checkerboard view

    # find the rotation and translation vectors
    ret, rvec, tvec = cv2.solvePnP(world_points, image_points, K, dist)
    
    err = reprojection_error(image_points, world_points, rvec, tvec, K, dist)
    logging.info(f'Extrinsics reprojection error: {err}')

    return rvec, tvec

def draw_coordinate_system(image, checker_size, checker_grid, K, dist):
    assert image.ndim == 2, 'Requires a grayscale image!'

    mask, image_points = search_corners([image], checker_grid)
    if not mask[0]:
        return None
    else:
        # 3D points of checkerboard corners: num_corners x 3
        world_points = get_world_points(image_points, checker_size, checker_grid)[0]

        # 2D points of checkerboard corners on image plane: num_corners x 2
        image_points = image_points[0]

        # coordinate axis with length of 3 checkerboard boxes
        axis = 3 * checker_size * np.eye(3)

        # find the rotation and translation vectors
        err, rvecs, tvecs = cv2.solvePnP(world_points, image_points, K, dist)
        logging.info(f'Extrinsics reprojection error: {err}')

        # project 3D points of coordinate axis to 2D image plane
        # returned image_points are of shape 3 x 1 x 2
        axis_image_points, jacobian = cv2.projectPoints(axis, rvecs, tvecs, K, dist)
        axis_image_points = axis_image_points[:, 0, :].astype(np.int32)

        origin = tuple(image_points[0].astype(np.int32))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # cv2.line color in BGR space, blue x-axis, green y-axis, red z-axis
        image = cv2.line(image, origin, tuple(axis_image_points[0]), (255,0,0), 5)
        image = cv2.line(image, origin, tuple(axis_image_points[1]), (0,255,0), 5)
        image = cv2.line(image, origin, tuple(axis_image_points[2]), (0,0,255), 5)
        
        return image

def main(args):
    #import pdb; pdb.set_trace()

    # number of corners along checkerboard sides 
    # note: longer side first by convention
    checker_grid = (10, 7)  
    # length of a single checker square side
    checker_size = 25  # in mm
    # unit of the checker_size parameter will influence
    # how we treat 3D world points, thus make sure the
    # checker_size units match the units of 3D world points,
    # note: units do not matter for camera calibration because 
    # there we are only interested in the intrinsics!

    # Camera calibration has improved when working on higher resolution
    # images, I used an A4 format paper printed checkerboard on cotton 
    # which was troublesome for the corner detection part (too small), 
    # especially for checkerboards that were further away from the camera,
    # with higher resolution and traversing in x, y and z direction
    # all corners where properly detected -> error drop from 4.5 to 0.24
    # typically below reprojection error below 1 is fine 

    answ = input('Capture some test images? (webcam check) (y/n):')
    if answ.lower() == 'y':
        capture_images('debug', args)

    answ = input('Just calculate extrinsics? (y/n):')
    if answ.lower() == 'y':
        folder = args.path + '/extrinsic'

        answ = input('Take new extrinsic images (DELETE existing ones)? (y/n):')
        if answ.lower() == 'y':
            delete_images(folder)
            # caputre images of checkerboard views
            capture_images(folder, args)

            # images saved to folder can be manually reviewed
            input('Press <ENTER> to contine...')

        images = read_images(glob(folder + '/*.png'))
        rvec, tvec = get_extrinsics(images, checker_grid, checker_size)

        # get rotation matrix from vector
        R, _ = cv2.Rodrigues(rvec)
        logging.info(f'Rotation matrix:\n{R}')
        logging.info(f'Translation vetor:\n{tvec}')
        save_extrinsics(R, tvec)
        return

    answ = input('Take new calibration images (DELETE existing ones)? (y/n):')
    if answ.lower() == 'y':
        # delete previos calibration images and debug visualizations
        delete_images(args.path)
        delete_images('debug')

        # caputre calibration images of a checkerboard
        capture_images(args.path, args)

        # images saved to folder can be manually reviewed
        input('Press <ENTER> to contine...')
    else:
        logging.info('Reuse calibration images...')

    images = read_images(glob(args.path + '/*.png'))

    # get intrinsics and extrinsics plus mask of valid checkerboard views 
    K, dist, Rs, tvecs, mask = calibrate(images, checker_grid, checker_size)

    image = images[mask][0]
    image = draw_coordinate_system(image, checker_size, checker_grid, K, dist)
    if image is not None:
        cv2.imwrite(f'debug/axis.png', image)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', help='Path to video file or camera id.')
    parser.add_argument('--path', default='./calibration')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    # setup folder structure
    os.makedirs(args.path, exist_ok=True)
    os.makedirs(args.path + '/extrinsic', exist_ok=True)
    os.makedirs('debug', exist_ok=True)

    main(args)
