import torch
import torch.nn as nn
import torchvision

import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary

class PatchEmbed(nn.Module):
    def __init__(self,in_chans,patch_size, embed_size=768):
        super().__init__()
        # self. img_size = img_size
        self.patch_size = patch_size
        #self.n_patches = (img_size//patch_size) **2 #number of patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_size,
            kernel_size = patch_size,
            stride = patch_size
        )

    def forward(self,x):
        
        x = self.proj(x)
        x = x.flatten(2) # flatten last two dimensions
        x= x.transpose(1,2) # transpose the dims
        return x



class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size =224,
                 inchannels = 3,
                 dropout = 0.1,
                 mlp_size = 512,
                 num_transformer_layers = 12,
                 num_heads= 6,
                 embed_dim = 768,
                 patch_size = 16,
                 num_classes =10
                 ) -> None:

        super().__init__()
        self.encode = PatchEmbed(
            in_chans=inchannels,
            patch_size = patch_size,
            embed_size= embed_dim,
            )
        

        assert (img_size % patch_size)==0, "image size and patch size should be divisible"
        
        # cls token which is ultimately used to train the model
        self.cls = nn.Parameter(torch.randn(1,1,embed_dim),requires_grad=True)
        
        # positional encoding generally done with sin and cosine

        patch_num = (img_size * img_size)// patch_size **2
        self.positinal_encoding = nn.Parameter(torch.randn(1,patch_num+1,embed_dim))

        self.embedding_dropout = nn.Dropout(dropout)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=mlp_size,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
            ),
            num_layers= num_transformer_layers-1
            )
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim,
                      out_features=num_classes)
        )
    def forward(self,x):
            batch_size = x.shape[0]
            x = self.encode(x)
            class_token = self.cls.expand(batch_size,-1,-1)
            x = torch.cat((class_token,x), dim=1)
            x = self.positinal_encoding + x
            x = self.embedding_dropout(x)
            x = self.transformer(x)
            x = self.mlp(x[:,0])
            return x



def calculate_accuracy(y_pred, y): # calcualting the accuracy of the model
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


transform = transforms.Compose(
    [
    
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(20),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
      transforms.Resize(size=224),
     ])

batch_size = 8

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True,transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                           ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = VisionTransformer()
# vit.to('cuda')
# demo_img = torch.randn(1,3,224,224).to('cuda')
# # print(vit.forward(demo_img))
# summary(vit, (3,224,224))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from torch.optim.lr_scheduler import CosineAnnealingLR
optimizer = optim.SGD(model.parameters(), lr = 1e-1)

criterion = nn.CrossEntropyLoss()
lr_scheduler = CosineAnnealingLR(optimizer, 150)
model = model.to(device)
criterion = criterion.to(device)



def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred= model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def train(model, iterator, optimizer, criterion, device): # traing the model on with the images in iterator
    
    epoch_loss = 0
    epoch_acc = 0
   
    model.train()
   
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        acc = calculate_accuracy(y_pred, y)
        
        loss.backward()
        optimizer.step()
       
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    lr_scheduler.step()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

if __name__=='__main__':
    loss =0
    for i in range(150):
    # using train iterator
        train_loss , epoch_acc = train(model,trainloader, optimizer, criterion, device=device)
        print("train loss per epoch => ",i, train_loss, "tain acc per epoch=> " , epoch_acc,"\n")
        
        #using validation iterator
        epoch_loss , epoch_valid_acc = evaluate(model, testloader, criterion, device)
        print("val loss per epoch =>", epoch_loss , "val acc per epoch =>",epoch_valid_acc)
        
        # saving the model whenever there is decrease in loss
        # tracking the model
        if epoch_loss<loss:
            torch.save(model,"./model.pt")
            loss= epoch_loss

