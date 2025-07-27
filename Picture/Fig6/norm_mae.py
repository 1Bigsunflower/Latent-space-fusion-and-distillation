import pandas as pd

def scale_by_task(file_name):
    df = pd.read_csv(file_name)

    # 对每个 Task 分组单独归一化MAE
    df['MAE_Norm'] = df.groupby('Task')['MAE'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    new_file_name = file_name.replace('_418', '_scaled_418')
    df.to_csv(new_file_name, index=False)


if __name__ == '__main__':
    scale_by_task('CGCNN_418')
    scale_by_task('MEGNet_418')
