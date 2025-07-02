from pytorch_fid import fid_score

# 两个数据分布的目录路径列表
path_real_data = 'S:/data/coco/128/small/images'  # 替换为真实数据的目录路径
path_generated_data = 'G:/demo5'  # 替换为生成模型生成的数据的目录路径

path = [path_real_data, path_generated_data]

# 计算FID距离值，确保batch_size参数是整数


fid = fid_score.calculate_fid_given_paths(path,
                                          batch_size=64, num_workers=0, device='cuda', dims=2048
                                          )
print('FID value:', fid)
