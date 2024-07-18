import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from utils import save_imgs, myResize, myRandomRotation, myNormalize, myToTensor, get_logger
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from models.AMLU_Net import AMLUNET
# from models.six_stage_unet import SIXUNET

class Val_dataset(Dataset):
    def __init__(self, path_Data = 'C:/Users/yuxilong/Desktop/AMLUNet-pytorch-main/data/muscle_us/test/images'):
        super(Val_dataset, self)

        test_transformer = transforms.Compose(
            [
                myNormalize('muscle_us', train=False),
                myToTensor(),
                myResize(256, 256)
            ]
        )

        # images_list = os.listdir(path_Data + 'test/images/')
        # masks_list = os.listdir(path_Data + 'test/masks/')
        images_list = os.listdir(path_Data + 'images/')
        masks_list = os.listdir(path_Data + 'masks/')
        images_list = sorted(images_list)
        masks_list = sorted(masks_list)
        self.data = []
        for i in range(len(images_list)):
            # img_path = path_Data + 'test/images/' + images_list[i]
            # mask_path = path_Data + 'test/masks/' + masks_list[i]
            img_path = path_Data + 'images/' + images_list[i]
            mask_path = path_Data + 'masks/' + masks_list[i]
            self.data.append([img_path, mask_path])
        self.transformer = test_transformer

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)

# 准备所需验证数据
val_data =  Val_dataset(path_Data = 'C:/Users/yuxilong/Desktop/AMLUNet-pytorch-main/data/ACSA/val/')
val_loader = DataLoader(val_data,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=0,
                                drop_last=True)

# 准备模型
model = AMLUNET(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True, gt_ds=True)

# 加载权重并载入模型
checkpoint = torch.load('C:/Users/yuxilong/Desktop/AMLUNet-pytorch-main/results/amlunet_ACSA/checkpoints/best-epoch243-loss0.2567.pth')
model.load_state_dict(checkpoint)

model.eval()
preds = []
gts = []
loss_list = []


with torch.no_grad():
    for i, data in enumerate(tqdm(val_loader)):
        img, msk = data
        img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
        img = img.cpu()
        gt_pre, out = model(img)
        # out = model(img)
        msk = msk.squeeze(1).cpu().detach().numpy()

        gts.append(msk)
        if type(out) is tuple:
            out = out[0]
        out = out.squeeze(1).cpu().detach().numpy()
        preds.append(out)
        # print(img.size())
        # print(msk)
        # print(out)
        if i % 1 == 0:
            save_imgs(img, msk, out, i, 'C:/Users/yuxilong/Desktop/segmentation_result/amlunet_acsa/', 'muscle_us', 0.5,
                      test_data_name=None)

    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds >= 0.5, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    test_data_name = None
    logger = get_logger('train', log_dir='C:/Users/yuxilong/Desktop/exp_for_paper/AMLUNet-pytorch-main/results/amlunet1_ms')
    if test_data_name is not None:
        log_info = f'test_datasets_name: {test_data_name}'
        print(log_info)
        logger.info(log_info)
    log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
            specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
    print(log_info)
    logger.info(log_info)