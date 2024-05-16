import numpy as np
import tqdm
import cv2
import math

class RayCastingResult:
    def __init__(void):
        pass
    def precompute(
            self,
            occupancy_map : np.ndarray,
            occupancy_threshold : float,
            map_resolution : float,
            subdivide : int,
            max_range : float,
            visualize : bool = True
        ):
        """
        occupancy_map : numpy array presenting the occupancy map.
                        -1 means unavailable.
                        Positive numbers mean the probability of occupancy. (E.g. 0: free; 1: occupant)
        occupancy_threshold : The threshold for considering a grid to be occupant.
        map_resolution : map grid size in cm.
        subdivide : subdivide the map grid to perform ray casting.
                    E.g., occupancy_map is a 800x800 array, subdivide=2, then 800x800x2x2x360 raycasting will be performed.
        max_range : terminating distance
        """
        self.map_shape = occupancy_map.shape
        self.occupancy_threshold = occupancy_threshold
        self.map_resolution = map_resolution
        self.subdivide = subdivide
        self.max_range = max_range
        occupancy_map_vis = np.zeros((self.map_shape[0], self.map_shape[1], 3), dtype=np.uint8)
        white_area = np.logical_and(occupancy_map >= 0.0, occupancy_map < self.occupancy_threshold)
        occupancy_map_vis[white_area, :] = [255, 255, 255]
        if visualize:
            cv2.imshow("Occupancy Map", occupancy_map_vis)
            cv2.waitKey()
        # Compute the valid region of the map. E.g. [0, occupancy_threshold)
        valid_pixels = np.nonzero(np.logical_and(occupancy_map >= 0, occupancy_map < self.occupancy_threshold))
        self.valid_region = np.array([
            np.min(valid_pixels[0]), # min Y (row index)
            np.max(valid_pixels[0]), # max Y (row index)
            np.min(valid_pixels[1]), # min X (col index)
            np.max(valid_pixels[1]), # max X (col index)
        ], dtype=np.int32)
        print("The valid region of the occupancy map is X{} ~ X{}, Y{} ~ Y{}.".format(self.valid_region[2], self.valid_region[3], self.valid_region[0], self.valid_region[1]))
        print("Which is a {}x{} rectangle region.".format(self.valid_region[3]-self.valid_region[2]+1, self.valid_region[1]-self.valid_region[0]+1))
        print("Subdivide={}, which means we will take {}x{}={} ray origins within each map grid.".format(self.subdivide, self.subdivide, self.subdivide, self.subdivide*self.subdivide))
        cv2.rectangle(
            occupancy_map_vis,
            [self.valid_region[2], self.valid_region[0]],
            [self.valid_region[3], self.valid_region[1]],
            [0, 255, 0],
            thickness=2
        )
        self.occupancy_map_vis = occupancy_map_vis
        if visualize:
            cv2.imshow("Occupancy Map", self.occupancy_map_vis)
            cv2.waitKey()
        self.result = np.empty(
            (
                (self.valid_region[1]-self.valid_region[0]+1) * self.subdivide,
                (self.valid_region[3]-self.valid_region[2]+1) * self.subdivide,
                360
            ),
            dtype=np.float64
        )
        store_row_ids, store_col_ids = np.meshgrid(
            np.arange(self.result.shape[0]),
            np.arange(self.result.shape[1])
        )
        store_row_ids, store_col_ids = store_row_ids.flatten(), store_col_ids.flatten()
        for store_row_id, store_col_id in tqdm.tqdm(zip(store_row_ids, store_col_ids)):
            # `store_row_id` and `store_col_id` is used to index `self.result`. E.g. self.result[store_row_id, store_col_id, 0:360] = ... 
            ray_origin = np.array([
                self.valid_region[2] * self.map_resolution + (store_col_id + 0.5) / self.subdivide * self.map_resolution,
                self.valid_region[0] * self.map_resolution + (store_row_id + 0.5) / self.subdivide * self.map_resolution
            ], dtype=np.float64) # X, Y
            z_gt = np.zeros((360,), dtype=np.float64)
            ray_dir = (np.arange(360, dtype=np.float64) + 0.5) / 180.0 * np.pi
            ray_dir = np.hstack([np.cos(ray_dir).reshape(360, 1), np.sin(ray_dir).reshape(360, 1)]) # 360x2
            indices = np.arange(360, dtype=np.int32)
            marching_length = 0.0
            marching_step = self.map_resolution / 2.0
            while indices.shape[0] > 0 and marching_length <= self.max_range:
                ray_target = [ray_origin] + marching_length * ray_dir[indices]
                X, Y = (ray_target[:, 0]/self.map_resolution).astype(np.int32), (ray_target[:, 1]/self.map_resolution).astype(np.int32)
                # `Y` and `X` is used to index occupancy_map.
                # Edge case: ray goes out of the map boundary
                outside = np.any([X < 0, X >= self.map_shape[1], Y < 0, Y >= self.map_shape[0]], axis=0)
                inside = np.logical_not(outside)
                z_gt[indices[outside]] = marching_length
                indices = indices[inside]
                X = X[inside]
                Y = Y[inside]
                # Detect hit
                hit = np.logical_or(occupancy_map[Y, X] >= self.occupancy_threshold, occupancy_map[Y, X] == -1)
                z_gt[indices[hit]] = marching_length
                indices = indices[np.logical_not(hit)]
                marching_length += marching_step
            # If there are remaining rays, assign `max_range`
            z_gt[indices] = self.max_range
            # Store back
            self.result[store_row_id, store_col_id, 0:360] = z_gt
            if visualize and occupancy_map[store_row_id//self.subdivide+self.valid_region[0], store_col_id//self.subdivide+self.valid_region[2]] >= 0.0 and occupancy_map[store_row_id//self.subdivide+self.valid_region[0], store_col_id//self.subdivide+self.valid_region[2]] < self.occupancy_threshold:
                vis = self.visualize_360(ray_origin[0], ray_origin[1])
                cv2.imshow("Occupancy Map", vis)
                cv2.waitKey(5)
        print("Successfully precomputed raycasting results.")     
    
    def visualize_360(self, world_x : float, world_y : float) -> np.ndarray:
        res360 = self.get_360(world_x, world_y)
        ray_dir = (np.arange(360, dtype=np.float64) + 0.5) / 180.0 * np.pi
        ray_dir = np.hstack([np.cos(ray_dir).reshape(360, 1), np.sin(ray_dir).reshape(360, 1)]) # 360x2
        occupancy_map_vis_copy = self.occupancy_map_vis.copy()
        for i in range(360):
            cv2.line(
                occupancy_map_vis_copy,
                (np.array([world_x, world_y])/self.map_resolution).astype(np.int32),
                ((np.array([world_x, world_y])+res360[i]*ray_dir[i])/self.map_resolution).astype(np.int32),
                [255,0,0]
            )
        cv2.circle(
            occupancy_map_vis_copy,
            (np.array([world_x, world_y])/self.map_resolution).astype(np.int32),
            3,
            [0,0,255]
        )
        return occupancy_map_vis_copy

    def visualize_180(self, world_x : float, world_y : float, orientation : float) -> np.ndarray:
        res180 = self.get_180(world_x, world_y, orientation)
        ray_dir = orientation - np.pi / 2.0 + (np.arange(180, dtype=np.float64) + 0.5) / 180.0 * np.pi
        ray_dir = np.hstack([np.cos(ray_dir).reshape(180, 1), np.sin(ray_dir).reshape(180, 1)]) # 180x2
        occupancy_map_vis_copy = self.occupancy_map_vis.copy()
        for i in range(180):
            cv2.line(
                occupancy_map_vis_copy,
                (np.array([world_x, world_y])/self.map_resolution).astype(np.int32),
                ((np.array([world_x, world_y])+res180[i]*ray_dir[i])/self.map_resolution).astype(np.int32),
                [255,0,0]
            )
        cv2.line(
            occupancy_map_vis_copy,
            (np.array([world_x, world_y])/self.map_resolution).astype(np.int32),
            (np.array([world_x, world_y])/self.map_resolution + 20 * np.array([np.cos(orientation), np.sin(orientation)])).astype(np.int32),
            [255,0,255],
            thickness=2
        )
        cv2.circle(
            occupancy_map_vis_copy,
            (np.array([world_x, world_y])/self.map_resolution).astype(np.int32),
            3,
            [0,0,255]
        )
        return occupancy_map_vis_copy

        
    def to_file(self, filename : str):
        np.savez(
            filename,
            map_shape=self.map_shape,
            occupancy_threshold=self.occupancy_threshold,
            map_resolution = self.map_resolution,
            subdivide=self.subdivide,
            max_range = self.max_range,
            valid_region=self.valid_region,
            occupancy_map_vis = self.occupancy_map_vis,
            result=self.result
        )
    @staticmethod
    def map_to_0_2pi(angle : float):
        return angle - np.floor(angle / 2.0 / np.pi) * 2.0 * np.pi
    
    def get_360(self, world_x : float, world_y : float) -> np.ndarray:
        store_row_id = int((world_y - self.valid_region[0] * self.map_resolution) * self.subdivide / self.map_resolution)
        store_col_id = int((world_x - self.valid_region[2] * self.map_resolution) * self.subdivide / self.map_resolution)
        if (store_row_id < 0 or store_row_id >= self.result.shape[0] or store_col_id < 0 or store_col_id >= self.result.shape[1]):
            return np.array([0.0]*360, dtype=np.float64)
        else:
            return self.result[store_row_id, store_col_id, 0:360]
        
    def get_180(self, world_x : float, world_y : float, orientation : float) -> np.ndarray:
        res360 = self.get_360(world_x, world_y)
        angle_id = int(self.map_to_0_2pi(orientation - np.pi / 2.0 + np.pi / 360.0) / np.pi * 180.0)
        if angle_id < 180:
            return res360[angle_id:angle_id+180]
        else:
            return np.concatenate((res360[angle_id:360], res360[0:angle_id-180]))  
    
    def from_file(self, filename : str):
        npz = np.load(filename)
        self.map_shape = npz["map_shape"]
        self.occupancy_threshold = npz["occupancy_threshold"]
        self.map_resolution = npz["map_resolution"]
        self.subdivide = npz["subdivide"]
        self.max_range = npz["max_range"]
        self.valid_region = npz["valid_region"]
        self.occupancy_map_vis = npz["occupancy_map_vis"]
        self.result = npz["result"]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ray_casting_result_path', default='./raycasting.npz')
    args = parser.parse_args()
    ray_casting_result = RayCastingResult()
    ray_casting_result.from_file(args.ray_casting_result_path)
    
    leftx, lefty = None, None
    def mouseCallback(event, x, y, flags, param):
        global occupancy_map_vis, leftx, lefty
        if event == cv2.EVENT_LBUTTONDOWN:
            leftx, lefty = x, y
            occupancy_map_vis = ray_casting_result.visualize_360((leftx + 0.5) * ray_casting_result.map_resolution, (lefty + 0.5) * ray_casting_result.map_resolution)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if leftx != None:
                angle = math.atan2(y - lefty, x - leftx)
            occupancy_map_vis = ray_casting_result.visualize_180((leftx + 0.5) * ray_casting_result.map_resolution, (lefty + 0.5) * ray_casting_result.map_resolution, angle)

    occupancy_map_vis = ray_casting_result.occupancy_map_vis
    cv2.namedWindow("Occupancy Map")
    cv2.setMouseCallback("Occupancy Map", mouseCallback)
    while(1):
        cv2.imshow("Occupancy Map", occupancy_map_vis)
        k = cv2.waitKey(10)
        if k == 27:
            break