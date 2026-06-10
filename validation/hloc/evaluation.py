
import torch
import os
from PIL import Image

from modules.vudnet import VUDNet
from validation.hloc.utils.netvlad.utils import image_process
from validation.vudnet_helper import VUDNet_helper

from validation.hloc.utils.get_colmap_info import estimate_pose_pnp, build_camera_matrix, build_2d3d_correspondence, remove_duplicate_points, load_colmap_model, generate_infomap, get_pose, rotmat_to_qvec, save_aachen_submission, find_all_query_images
from validation.hloc.utils.read_colmaps import read_intrinsics_binary

from validation.hloc.utils.netvlad.netvlad import NetVLAD

def imageRetrieval(query_img, netvlad_model,global_desc_names):
    
    global_desc, names = torch.squeeze(global_desc_names[0]), global_desc_names[1]
    query_global_desc = netvlad_model(query_img[None])["global_descriptor"]
    
    similarity = torch.mm(query_global_desc, global_desc.t().cuda())
    _, idx = similarity.max(dim=1)
    
    #num_seq = names[idx].split("/")[0]
    #img_name = names[idx].split("/")[1]
    
    #return img_name, num_seq
    return names[idx]

def createNetVlad():
    conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NetVLAD(conf).eval().to(device)
    
    return model

def localiza_set(query_img_folder):
    
    
    db_path = "datasets/aachen/"
    
    images, name_to_image, pointid_to_xyz = load_colmap_model("datasets/aachen/sparse/0")
    cameras = read_intrinsics_binary("datasets/aachen/sparse/0/cameras.bin")
    
    
    infomap = generate_infomap(images)
    #Load the global descriptor
    global_desc_names = torch.load("datasets/aachen/global_desc.pt")
    netvlad_model = createNetVlad()
    
    # Load the model
    model = VUDNet(top_k=4096)
    model_helper = VUDNet_helper(model)
    query_images = find_all_query_images(query_img_folder)

    print(f"Found {len(query_images)} query images")
    
    
    results = []
    
    for img_path in query_images:
        img_name = os.path.basename(img_path)
        if img_name.endswith(".jpg") or img_name.endswith(".png"):
            print(f"Processing {img_path}...")
            
            try:
                query_img = Image.open(img_path) 
        
            except Exception as e:
                print(f"Error opening image: {img_path}")
                print(e)
                continue
            
            original_image = image_process(query_img)
            query_image = original_image.cuda()
            
            #Get the reference image name 
            ref_name= imageRetrieval(query_image, netvlad_model,global_desc_names)
            ref_name_jpg = ref_name.replace("png", "jpg")
            R_ref, t_ref = get_pose(ref_name, infomap)
            ref_img_path = os.path.join(db_path, ref_name_jpg)
            
            kpts0, kpts1, sigma0, sigma1 = model_helper.match(img_path, ref_img_path)
            
            ref_image = name_to_image[ref_name]

            query_pts, world_pts = build_2d3d_correspondence(
                kpts0,
                kpts1,
                ref_image,
                pointid_to_xyz)

            if query_pts is None:
                print("No valid 2D-3D matches")
                continue
            
            print("2D-3D correspondences:", len(query_pts))
            
            query_pts, world_pts = remove_duplicate_points(
                query_pts,
                world_pts)

            print("2D-3D correspondences:", len(query_pts))
            
            camera = cameras[ref_image.camera_id]

            K = build_camera_matrix(camera)
            
            pose = estimate_pose_pnp(query_pts,world_pts,K)

            if pose is None:
                print("PnP failed")
                continue

            R, t, inliers = pose

            print(f"Inliers: {len(inliers)} / {len(query_pts)}")
            
            qvec = rotmat_to_qvec(R)

            tx, ty, tz = t.reshape(-1)

            results.append((
                img_name,
                qvec,
                [tx, ty, tz]
            )   )

        save_aachen_submission(results,"Aachen_eval_VUDNet.txt")
        print(f"Saved {len(results)} poses.")


            
            
            
            
            

            




if __name__ == "__main__":
    query_img_folder = "datasets/aachen/query/"
    localiza_set(query_img_folder)



            
            
    
    
    
