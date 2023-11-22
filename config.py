class Config():
        #training args

        epoch=100
        batch_size=2
        Vis_img="./TNO/vi"
        Inf_img="./TNO/ir"

        test_Vis_img="./TNO/vi"
        test_Inf_img="./TNO/ir"
        HEIGHT = 256
        WIDTH = 256
        color = 'grayscale'
        in_channel = 1  #if you choose color=grayscle,the in_channel must be 1,'RGB'means in_channel=3
        save_model_dir = "models" #"path to folder where trained model will be saved."
        fusion_strategy='l1_norm'
        image_size = 256 #"size of training images, default is 256 X 256"
        cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
        result = "./result"
        ssim_weights = [1, 10, 100, 1000]
    

        lr = 1e-4 #"learning rate, default is 0.001"


        # for test Final_cat_epoch_9_Wed_Jan__9_04_16_28_2019_1.0_1.0.model
        model_path_gray = "./models/DenseFuse_epoch_100_gray.pth"
        model_path_rgb = "./models/DenseFuse_epoch_100_rgb.pth"