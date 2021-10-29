import torch
from dataset import Endoscope
from model import resnet18, PFMLP, resnet, PFMlpNet
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import numpy as np


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = PFMLP(net=resnet18(), n_att=3, in_dim=512)
model = PFMLP(net=PFMlpNet(patch_size=40, num_layers=6), n_att=6, in_dim=256)
model.to(gpu)

epoch = 200
lr = 0.001
min_lr = 0.00001
lr_update_step = 20
save_path = "./params/pfmlp-att3-patch40-max-800.pth"
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=0)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
loss_fn = BCELoss()
endscope_dataset = Endoscope(
    roots=["E:/内镜图片资料/2020pcr阳性", "E:/内镜图片资料/2021PCR阳性", "E:/内镜图片资料/2020pcr阴性"],
    labels=[1, 1, 0]
)

if __name__ == '__main__':
    def train(load=False):
        def update_lr(optimizer, gamma=0.5):
            lr = 0
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            lr = max(lr * gamma, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print("lr update finished  cur lr: %.5f" % lr)

        if load:
            model.load_state_dict(torch.load(save_path))
        best_acc = 0
        for epoch_count in range(1, epoch+1):
            correct, correct_thresh = 0, 0.5
            access_idx = np.random.permutation(len(endscope_dataset))
            loss_val = 0
            error_data = 0
            for i in range(len(endscope_dataset)):
                ds = endscope_dataset[access_idx[i].item()]
                if len(ds) <= 8:
                    error_data += 1
                    continue
                dataloader = DataLoader(dataset=ds, batch_size=16, shuffle=True)
                out, w = model(dataloader)
                if (out[0, 0].item() >= correct_thresh and ds.label == 1) or (out[0, 0].item() < correct_thresh and ds.label == 0):
                    correct += 1

                loss = loss_fn(out[:, 0], torch.Tensor([ds.label]).to(gpu))
                loss_val += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("\repoch: %d  %d / %d  loss: %.5f  acc: %.5f" % (epoch_count, i+1, len(endscope_dataset), loss.item(), correct/(i+1-error_data)), end="")
            acc = correct/len(endscope_dataset)
            print("\n%.5f" % loss_val)
            if acc > best_acc:
                best_acc = acc
                print("save...")
                torch.save(model.state_dict(), save_path)
                print("save finish !!")

            if epoch_count % lr_update_step == 0:
                update_lr(optimizer, 0.5)

    def evaluate(load=False):
        if load:
            model.load_state_dict(torch.load(save_path))
        correct, correct_thresh = 0, 0.5
        model.eval()
        error_data_num = 0
        with torch.no_grad():
            for i in range(0, len(endscope_dataset)):
                ds = endscope_dataset[i]
                if len(ds) <= 8:
                    print("\ndata error!\n")
                    error_data_num += 1
                    continue
                dataloader = DataLoader(dataset=ds, batch_size=16, shuffle=True)
                out, w = model(dataloader)
                # out = 1-out
                loss = loss_fn(out[:, 0], torch.Tensor([ds.label]).to(gpu))
                if (out[0, 0].item() >= correct_thresh and ds.label == 1) or (out[0, 0].item() < correct_thresh and ds.label == 0):
                    correct += 1
                print("\r%d / %d  loss: %.5f  acc: %.5f" % (i+1, len(endscope_dataset), loss.item(), correct/(i+1-error_data_num)), end="")

    # train(load=False)
    evaluate(load=True)