
import torch
import os
from PIL import Image

from modules.vudnet import VUDNet
from validation.hloc.utils.netvlad.utils import image_process
from validation.vudnet_helper import VUDNet_helper

from validation.hloc.utils.get_colmap_info import generate_infomap, get_pose

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
    
    db_path = "datasets/aachen/db/"
    
    infomap = generate_infomap("datasets/aachen/sparse/0/images.bin")
    #Load the global descriptor
    global_desc_names = torch.load("datasets/aachen/global_desc.pt")
    netvlad_model = createNetVlad()
    
    # Load the model
    model = VUDNet(top_k=4096)
    model_helper = VUDNet_helper(model)
    
    for img_name in os.listdir(query_img_folder):
        if img_name.endswith(".jpg") or img_name.endswith(".png"):
            img_path = os.path.join(query_img_folder, img_name)
            print(f"Processing {img_path}...")
            
            try:
                query_img = Image.open(img_path) 
        
            except:
                print(f"Error opening image: {img_path}")
                continue
            
            original_image = image_process(query_img)
            query_image = original_image.cuda()
            
            #Get the reference image name 
            ref_name= imageRetrieval(query_image, netvlad_model,global_desc_names)
            
            ref_img_path = os.path.join(db_path, ref_name)
            q, v = get_pose(ref_name, infomap)
            print(f"Reference image: {ref_name}, qvec: {q}, tvec: {v}")



if __name__ == "__main__":
    query_img_folder = "datasets/aachen/query/day/milestone"
    localiza_set(query_img_folder)



            
            
    
    
    
