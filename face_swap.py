import cv2
import numpy as np
import mediapipe as mp

def get_images_gray(source_image, target_image):
    if source_image is None or target_image is None:
        raise ValueError("Could not read one of the images. Check the file paths.")
    
    gray_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    
    return gray_source, gray_target

def get_landmark_points(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise ValueError("No face landmarks detected.")
        landmarks = results.multi_face_landmarks[0].landmark
        points = [(landmark.x * image.shape[1], landmark.y * image.shape[0]) for landmark in landmarks]
        return points

def get_int_points(landmark_points):
    return np.array(landmark_points, dtype=np.int32)


def get_convex_hull(landmark_points):
    points = get_int_points(landmark_points)
    convex_hull = cv2.convexHull(points)
    return convex_hull

def get_mask(image):    
    mask = np.zeros_like(image)
    return mask

def fill_convex_hull(mask, convex_hull):
    cv2.fillConvexPoly(mask, convex_hull, (255, 255, 255))
    return mask

def extract_face(image, mask):
    face = cv2.bitwise_and(image, mask)
    return face

def get_bounding_rectangle(image, convex_hull):
    rect = cv2.boundingRect(convex_hull)
    x, y, w, h = rect
    return x, y, w, h

def delunay_triangulation(rect, landmark_points):
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    return triangles

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def get_index_triangles(triangles, landmark_points):
    points = get_int_points(landmark_points)
    index_triangles = []
    for triangle in triangles:
        pt1 = (triangle[0], triangle[1])
        pt2 = (triangle[2], triangle[3])
        pt3 = (triangle[4], triangle[5])
    
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1) #first index of point pt1

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            index_triangles.append(triangle)
    
    return index_triangles
def swap_faces(source_indexes_triangles, source_img, dest_img):
    h_dest, w_dest = dest_img.shape[:2]
    dest_new_face = np.zeros_like(dest_img, dtype=dest_img.dtype) # Match dtype

    source_lpoints = get_landmark_points(source_img)
    dest_lpoints = get_landmark_points(dest_img)

    if source_lpoints is None or dest_lpoints is None:
         print("Error: Could not get landmarks for one or both images inside triangularize.")
         return None # Cannot proceed without landmarks

    source_lpoints_int = get_int_points(source_lpoints)
    dest_lpoints_int = get_int_points(dest_lpoints)
    if source_lpoints_int is None or dest_lpoints_int is None:
         print("Error: Could not convert landmarks to integers.")
         return None


    skipped_count = 0
    for triangle_index in source_indexes_triangles:
        try:
            # Validate indices
            if any(idx >= len(source_lpoints) or idx >= len(dest_lpoints) for idx in triangle_index):
                 skipped_count+=1
                 continue

            # 1. Source Triangle Processing
            tr1_pt1_idx, tr1_pt2_idx, tr1_pt3_idx = triangle_index
            f_tr1_pt1, f_tr1_pt2, f_tr1_pt3 = source_lpoints[tr1_pt1_idx], source_lpoints[tr1_pt2_idx], source_lpoints[tr1_pt3_idx]
            tr1_pt1, tr1_pt2, tr1_pt3 = tuple(source_lpoints_int[tr1_pt1_idx]), tuple(source_lpoints_int[tr1_pt2_idx]), tuple(source_lpoints_int[tr1_pt3_idx])

            triangle1_int = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
            rect1 = cv2.boundingRect(triangle1_int)
            x1, y1, w1, h1 = rect1
            x1, y1 = max(0, x1), max(0, y1)

            if w1 <= 0 or h1 <= 0: skipped_count+=1; continue # Skip degenerate rect

            # Crop source triangle
            cropped_triangle_src = source_img[y1 : y1 + h1, x1 : x1 + w1]
            if cropped_triangle_src.size == 0: skipped_count+=1; continue

            # Create mask for source triangle
            points1_rel_float = np.array([[f_tr1_pt1[0]-x1, f_tr1_pt1[1]-y1], [f_tr1_pt2[0]-x1, f_tr1_pt2[1]-y1], [f_tr1_pt3[0]-x1, f_tr1_pt3[1]-y1]], np.float32)
            points1_rel_int   = np.array([[tr1_pt1[0]-x1, tr1_pt1[1]-y1], [tr1_pt2[0]-x1, tr1_pt2[1]-y1], [tr1_pt3[0]-x1, tr1_pt3[1]-y1]], np.int32)
            mask1 = np.zeros((h1, w1), np.uint8)
            cv2.fillConvexPoly(mask1, points1_rel_int, 255)

            # 2. Destination Triangle Processing
            f_tr2_pt1, f_tr2_pt2, f_tr2_pt3 = dest_lpoints[tr1_pt1_idx], dest_lpoints[tr1_pt2_idx], dest_lpoints[tr1_pt3_idx] # Use same indices
            tr2_pt1, tr2_pt2, tr2_pt3 = tuple(dest_lpoints_int[tr1_pt1_idx]), tuple(dest_lpoints_int[tr1_pt2_idx]), tuple(dest_lpoints_int[tr1_pt3_idx])

            triangle2_int = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
            rect2 = cv2.boundingRect(triangle2_int)
            x2, y2, w2, h2 = rect2
            x2, y2 = max(0, x2), max(0, y2) # Ensure non-negative origin

            if w2 <= 0 or h2 <= 0: skipped_count+=1; continue # Skip degenerate rect

            # Create mask for destination triangle
            points2_rel_float = np.array([[f_tr2_pt1[0]-x2, f_tr2_pt1[1]-y2], [f_tr2_pt2[0]-x2, f_tr2_pt2[1]-y2], [f_tr2_pt3[0]-x2, f_tr2_pt3[1]-y2]], np.float32)
            points2_rel_int   = np.array([[tr2_pt1[0]-x2, tr2_pt1[1]-y2], [tr2_pt2[0]-x2, tr2_pt2[1]-y2], [tr2_pt3[0]-x2, tr2_pt3[1]-y2]], np.int32)
            mask2 = np.zeros((h2, w2), np.uint8)
            cv2.fillConvexPoly(mask2, points2_rel_int, 255)

            # 3. Warp Triangle
            try:
                # Check for collinear points before getting transform
                if cv2.norm(points1_rel_float[0] - points1_rel_float[1]) < 1e-6 or \
                   cv2.norm(points1_rel_float[1] - points1_rel_float[2]) < 1e-6 or \
                   cv2.norm(points1_rel_float[2] - points1_rel_float[0]) < 1e-6:
                    # print("Skipping collinear source points") # Debug
                    skipped_count+=1
                    continue

                M = cv2.getAffineTransform(points1_rel_float, points2_rel_float)
                warped_triangle = cv2.warpAffine(cropped_triangle_src, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) # Use INTER_LINEAR and border mode

                # Apply destination mask to warped triangle
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask2)

                # 4. Reconstruct Destination Face (Blend triangles)
                if y2 + h2 > h_dest or x2 + w2 > w_dest:
                    skipped_count+=1
                    continue

                dest_rect_area = dest_new_face[y2 : y2 + h2, x2 : x2 + w2]
                dest_rect_area_gray = cv2.cvtColor(dest_rect_area, cv2.COLOR_BGR2GRAY)
                _, mask_existing = cv2.threshold(dest_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)

                # Ensure shapes match before blending
                if warped_triangle.shape != dest_rect_area.shape:
                    mask_existing_resized = cv2.resize(mask_existing, (warped_triangle.shape[1], warped_triangle.shape[0]), interpolation=cv2.INTER_NEAREST)
                    if warped_triangle.shape[:2] != mask_existing_resized.shape[:2]: # Double check after resize
                         print(f"Warning: Mask resize failed or shape still mismatch. Skipping.")
                         skipped_count+=1
                         continue
                    mask_to_blend = mask_existing_resized

                else:
                     mask_to_blend = mask_existing


                if len(mask_to_blend.shape) == 3: # Should not happen, but check
                     mask_to_blend = mask_to_blend[:,:,0]

                warped_triangle_masked = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_to_blend)

                dest_new_face[y2 : y2 + h2, x2 : x2 + w2] = cv2.add(dest_rect_area, warped_triangle_masked)

            except cv2.error as e:
                skipped_count+=1
                continue 

        except Exception as e:
            import traceback
            skipped_count+=1
            continue 

    if skipped_count == len(source_indexes_triangles):
        print("Error: All triangles were skipped. Cannot proceed.")
        return None


    # --- Create Final Mask and Blend using SeamlessClone with Padding ---
    dest_convexhull = get_convex_hull(dest_lpoints_int, dest_img.shape)
    if dest_convexhull is None:
        print("Error: Could not get destination convex hull. Cannot apply seamless clone.")
        return dest_new_face # Or None

    # Create destination mask
    dest_mask = np.zeros(dest_img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(dest_mask, dest_convexhull, 255)

    # Calculate bounding rect of the mask
    (x_hull, y_hull, w_hull, h_hull) = cv2.boundingRect(dest_convexhull)
    center_face_dest = (x_hull + w_hull // 2, y_hull + h_hull // 2)

    half_w = w_hull // 2
    half_h = h_hull // 2
    cx, cy = center_face_dest

    # Padding needed on each side
    pad_left = max(0, half_w - cx)
    pad_top = max(0, half_h - cy)
    pad_right = max(0, (cx + w_hull) - w_dest) 
    pad_bottom = max(0, (cy + h_hull) - h_dest) 

    # Add a small safety margin
    safety_pad = 10
    pad_top += safety_pad
    pad_bottom += safety_pad
    pad_left += safety_pad
    pad_right += safety_pad

    try:
        # Pad the original destination image
        padded_dest_img = cv2.copyMakeBorder(dest_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101) # Reflect padding often looks better

        dest_mask_inv = cv2.bitwise_not(dest_mask)
        img_dest_noface = cv2.bitwise_and(dest_img, dest_img, mask=dest_mask_inv)
        result_intermediate = cv2.add(img_dest_noface, dest_new_face)

        padded_result = cv2.copyMakeBorder(result_intermediate, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]) # Pad result with black

        # Pad the mask
        padded_mask = cv2.copyMakeBorder(dest_mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0) # Pad mask with 0

        # Adjust the center coordinates for the padded image
        new_center = (center_face_dest[0] + pad_left, center_face_dest[1] + pad_top)

        # Perform Seamless Clone on Padded Images
        cloned_padded = cv2.seamlessClone(padded_result, padded_dest_img, padded_mask, new_center, cv2.NORMAL_CLONE)

        # Crop Back to Original Size
        final_result = cloned_padded[pad_top : pad_top + h_dest, pad_left : pad_left + w_dest]

        return final_result

    except cv2.error as e:
         print(f"Error during padding or seamlessClone: {e}")
         import traceback
         traceback.print_exc()
         # Fallback: show the non-seamless result
         cv2.imshow("Raw Blended Face (Seamless Failed)", result_intermediate)
         cv2.waitKey(0)
         cv2.destroyAllWindows()
         return None
    except Exception as e:
         print(f"An unexpected error occurred during final blending: {e}")
         import traceback
         traceback.print_exc()
         return None