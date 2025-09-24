import numpy as np
import cv2

def extract_frames_to_list(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    
    while ret:
        frames.append(image)
        ret, image = cap.read()
            
    cap.release()
    return frames

def match_features(desc1, desc2, ratio = 0.6, max_matches = 100):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    
    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k = 2)
    matches = sorted(matches, key=lambda x: x[0].distance if len(x) == 2 else float('inf'))

    good = []
    
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio * n.distance:
                good.append(m)
    
    good = sorted(good, key=lambda x: x.distance)
    return good[:max_matches]

def get_homography(kp1, kp2, matches):
    if len(matches) < 4:
        return None
    
    valid_matches = [m for m in matches if m.queryIdx < len(kp1) and m.trainIdx < len(kp2)]

    if len(valid_matches) < 4:
        print("Warning: Not enough valid matches after index check")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
    dis_pts = np.float32([kp2[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dis_pts, cv2.RANSAC, 5.0)

    if H is None or not np.isfinite(H).all():
        print(f"Invalid homography at frame {i}")
        H = np.eye(3)

    return H

def stitch_frames(frames, homographies):
    h_max, w_max = 0, 0
    corners = []

    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, homographies[i])
        corners.append(warped_pts)

    all_pts = np.concatenate(corners, axis = 0)
    x_min, y_min = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    w_max, h_max = x_max - x_min, y_max - y_min

    panorama = np.zeros((h_max, w_max, 3), dtype = np.uint8)
    offset = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    for i, frame in enumerate(frames):
        warped = cv2.warpPerspective(frame, offset @ homographies[i], (w_max, h_max))
        gray_mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
        
        cv2.copyTo(warped, mask, panorama)

    return panorama



video_file = r"c:\Users\ashto\OneDrive\Documents\Berkeley\UAVs@Berkeley\software\Minecraft_stitch_test.mp4"
frame_list = extract_frames_to_list(video_file)

downsampled_frames = frame_list[::5]

sift = cv2.SIFT_create()
sift_results = []

for frame in downsampled_frames:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(gray, None)
    if desc is not None and len(kp) > 0:
        sift_results.append((kp, desc))
    else:
        print("Warning: No keypoints/descriptors found for a frame.")
        sift_results.append(([], np.empty((0, 128), dtype=np.float32)))

homographies = [np.eye(3)]

for i in range(1, len(downsampled_frames)):
    kp1, desc1 = sift_results[i - 1]
    kp2, desc2 = sift_results[i]
    good_matches = match_features(desc1, desc2)
    H = get_homography(kp1, kp2, good_matches)

    if H is None:
        H = np.eye(3)
    else:
        H = H / H[2, 2]

    homographies.append(H)

accumulated_homographies = [homographies[0]]

for i in range(1, len(homographies)):
    H = np.dot(accumulated_homographies[i - 1], homographies[i])
    accumulated_homographies.append(H)

panorama = stitch_frames(downsampled_frames, accumulated_homographies)

cv2.imshow('panorama', panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(r"C:\Users\ashto\OneDrive\Documents\Berkeley\UAVs@Berkeley\software\Github\software_ground_school\attempted stitching\resized_stitched_panorama.jpg", panorama)