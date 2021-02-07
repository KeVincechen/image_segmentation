import os
import ruamel.yaml

yaml_files_dir = os.path.join(os.path.dirname(__file__), 'yamls')  # 配置文件存放目录
os.makedirs(yaml_files_dir, exist_ok=True)
example_yaml = r'''
# example
imgs_dir: F:\dataset\segmentation\天池比赛数据\suichang_round1_train_210120\road\images  # 训练图片所在目录
masks_dir: F:\dataset\segmentation\天池比赛数据\suichang_round1_train_210120\road\masks  # 训练标签所在目录                                                                        
network: Unet  # 要使用的模型的类名，在model包内定义的类: Unet,Dinknet,DeepLabV3...  
backbone_name: resnet34  # backbone名称，resnet34,resnet50,resnet101... 如果没有使用到（如unet），则可为空                                                                   
dataset: TianchiRoadDataset  # 自定义的dataset的类名，区分大小写: 在dataset包里定义的类                                                              
loss_weights: [1,50]  # 损失函数中的权重参数,列表     
in_channels: 3                                                                      
num_classes: 2  # 模型输出通道数（分类数量）                                                                          
batch_size: 32  # 批次大小                                                                           
learning_rate: 0.0001  # 学习率                      
checkpoint_path: null  # 该模型的预训练模型文件的路径                      
split_val_rate: 0.1  # 验证集划分比例                        
num_workers: 4  # 数据加载进程数量                             
save_top_k: 10  # 保存结果最好的k个模型                             
save_mode: max  # 保存模式：max，min，auto，与monitor相对应                             
model_saved_filename: model-{epoch:04d}-{val_acc:.4f} # 保存的模型的文件名（格式化）
backbone_pretrained: true  # 是否使用backbone预训练权重: true,false. 
gpus: 1  # 使用的gpu编号                                    
accumulate_grad_batches: 8  # 梯度累积轮数
optim: Adam  # 优化器名称：在torch.optim包下定义的优化器类                
loss_func: MyCrossEntropyLoss  # 要使用的损失函数类名，区分大小写：在loss包内定义的类
metrics: acc  # 要使用的评价指标函数名：在metrics包类定义的函数
'''


def create_yaml(yaml_filename):
    yaml_file_path = os.path.join(yaml_files_dir, yaml_filename)
    yaml = ruamel.yaml.YAML()
    with open(yaml_file_path, 'w', encoding='utf-8') as f:
        args = yaml.load(example_yaml)
        yaml.dump(args, f)
    print(f'{yaml_filename}配置文件创建成功！')


if __name__ == '__main__':
    create_yaml('tianchi-road-deeplabv3.yaml')
