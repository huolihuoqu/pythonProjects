from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))  # 类别转为整型int
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        # img = Image.open(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# if __name__ == '__main__':
#     traintxt_path = './data/catVSdog/train.txt'
#     train_dataset = MyDateset(traintxt_path)
#     print(len(train_dataset))
#     img, target = train_dataset[0]
#     print(img.size)
#     print(target)