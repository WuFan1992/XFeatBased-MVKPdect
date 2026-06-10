import torch
import os
from PIL import Image

from validation.hloc.utils.netvlad.netvlad import NetVLAD


from validation.hloc.utils.netvlad.utils import image_process, get_registered_images

###  Command #############
# python getdes.py -s ../../GSplatLoc/gsplatloc-main/datasets/wholehead/ -m ../../GSplatLoc/gsplatloc-main/output_wholescene/img_2000_head
########################################################################

from tqdm import tqdm

def getNetVladDesc(folder_path, model, registered_imgs):

    imgs_name = []
    global_desc = []


    for filename in tqdm(sorted(registered_imgs),
                         desc="Extracting NetVLAD"):
        img_path = os.path.join(folder_path, filename.replace("png", "jpg"))


        if not os.path.exists(img_path):
            print(f"[Warning] Missing image: {img_path}")
            continue

        query_img = Image.open(img_path).convert("RGB")

        original_image = image_process(query_img)
        image = original_image.cuda()

        with torch.no_grad():
            output = model(image[None])["global_descriptor"]

        global_desc.append(output.cpu())
        imgs_name.append(filename)

    return global_desc, imgs_name




# Discard layers at the end of base network
conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}
db_folder = "datasets/aachen/"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NetVLAD(conf).eval().to(device)
output_folder = "datasets/aachen/global_desc.pt"
colmap_model = "datasets/aachen/sparse/0/images.bin"
registered_imgs = get_registered_images(colmap_model)

print(f"COLMAP registered images: {len(registered_imgs)}")


#Prepare the dataset 


global_desc, imgs_name = getNetVladDesc(db_folder, model, registered_imgs)
desc_names_tensor = [torch.stack(global_desc), imgs_name]
torch.save(desc_names_tensor, output_folder)




"""
#Get the image 
img_dir_query = "../datasets/wholehead/images/seq-01"   
query_img_name = "frame-000969.color.png"
query_img_path = os.path.join(img_dir_query, query_img_name)
query_img = cv2.imread(query_img_path) # [H,W,C] = [480,640,3]
query_img_tensor = torch.tensor(query_img).permute(2,0,1)[None].cuda() # [C,H,W]

# This is just toy example. Typically, the number of samples in each classes are 4.
#labels = torch.randint(0, 10, (40, )).long()
#x = torch.rand(40, 3, 128, 128).cuda()
output = model(query_img_tensor.to(torch.float))

#triplet_loss = criterion(output, labels)
"""

"""

conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NetVLAD(conf).eval().to(device)


#Get the image 
img_dir_query = "../datasets/wholehead/images/seq-01"   
query_img_name = "frame-000969.color.png"
query_img_path = os.path.join(img_dir_query, query_img_name)
query_img = cv2.imread(query_img_path) # [H,W,C] = [480,640,3]
query_img_tensor = torch.tensor(query_img).permute(2,0,1)[None].cuda() # [C,H,W]


pred = model(query_img_tensor)
print(pred["global_descriptor"])


"""