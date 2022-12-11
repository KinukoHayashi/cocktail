# ブランデーモカベイズ最適化プログラム
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime
import GPy
import GPyOpt
import csv
from matplotlib import pyplot as plt

# 得られたx,yを入力。x,yともにnumpyの2D arrayとなることに注意

# 既知の実験データを読み込み前処理(標準化)を実行

data = pd.read_csv("cocktail.csv", encoding="utf-8")
# 実験における変数の数を定義
num_x = 3
# 各変数の刻み幅(step),最小値(min),最大値(max),標準化後の刻み幅(norm_step),標準化後の定義域(norm_range)を定義
# 摩擦係数をかんがえるのでmax = 1，min = 0とした. 刻み幅は適当に0.01 としたので要検討

for i in range(num_x):
    # print(i)
    exec("x"+str(i)+"_step = data.iloc[0,"+str(i)+"]")
    exec("x"+str(i)+"_min = data.iloc[1,"+str(i)+"]")
    exec("x"+str(i)+"_max = data.iloc[2,"+str(i)+"]")
    exec("norm_x"+str(i)+"_step = ((x"+str(i) +
         "_step)/((x"+str(i)+"_max)-(x"+str(i)+"_min)))")

norm_x0_range = np.arange(0, 1.2, 0.2)
norm_x1_range = np.arange(0, 1.2, 0.2)
norm_x2_range = np.arange(0, 1.25, 0.25)

# MinMaxScalerを用いて標準化処理を実行（標準化前：data→標準化後：norm_dataにデータ名を変更）
# 読み込んだ既知の実験データに定義域外のデータ（最小値や最大値の範囲を超える設定値）がある場合はこの方法では上手くいかないので注意
# 読み込んだデータの3行目（最小値）以下のxの値のみを標準化を実行

data_before_norm = data.iloc[1:, :-1]
mmscaler = MinMaxScaler()
mmscaler.fit(data_before_norm)
norm_data_wo_y = mmscaler.transform(data_before_norm)
norm_data_wo_y_df = pd.DataFrame(norm_data_wo_y)
# norm_data_wo_y_dfの先頭2行（行番号0と1）は最小値と最大値なので取り除く
new_norm_data_wo_y = norm_data_wo_y_df.iloc[2:, :]
new_norm_data_wo_y_reset = new_norm_data_wo_y.reset_index(drop=True)
df_new_norm_data_wo_y_reset = pd.DataFrame(new_norm_data_wo_y_reset)
# 標準化処理したデータにもともとのyの値を結合していくためにyの値だけ格納されたデータフレームを作成していく
data_y = data.iloc[3:, -1]
# のちのデータフレームの結合を考えてindex番号をリセット
data_y_reset = data_y.reset_index(drop=True)
df_data_y_reset = pd.DataFrame(data_y_reset)
# 標準化したxのみのデータフレームとyのみのデータフレームを結合し、csvファイルに書き出し
norm_data = pd.concat(
    [df_new_norm_data_wo_y_reset, df_data_y_reset], axis=1)
norm_data.columns = data.columns
norm_data.to_csv("norm_cocktail.csv", index=False)
# 正規化後のデータに対してベイズ最適化を実行（yの値をなるべく小さくする条件探索）

# 乱数初期化（設定は必須でない？）
np.random.seed(1234)
# 定義域（条件の探索範囲）の設定
# typeは‘continuous’:連続変数，‘discrete’：離散変数，‘categorical’：カテゴリで指定
bounds = [{'name': 'brandy',
           'type': 'discrete', 'domain': norm_x0_range}, {'name': 'Kahlua', 'type': 'discrete', 'domain': norm_x1_range}, {'name': 'cream', 'type': 'discrete', 'domain': norm_x2_range}]

# 既知データをインプットしていく
# データ入力用の箱を作っておく
X_step = []
Y_step = []
# 標準化済みのcsvデータ（既知の実験データ）を読み込む
norm_data = pd.read_csv("norm_cocktail.csv", encoding="utf-8")
norm_data_x = norm_data.iloc[:, 0:num_x]  # 全行のxを読み込む
norm_data_y = norm_data.iloc[:, num_x:num_x+1]  # 全行のyを読み込むs
# xとyをリスト化する
X_step = np.asarray(norm_data_x)
Y_step = np.asarray(norm_data_y)

nextx = []
# 以下のパラメータでベイズ最適化を実行する
params = {'acquisition_type': 'LCB',  # 獲得関数としてLower Confidence Boundを指定
          'acquisition_weight': 1,  # LCBのパラメータを設定．デフォルトは2
          # GPに使うカーネルの設定 #boundsの次元でinputの次元を定義
          'kernel': GPy.kern.Matern52(input_dim=len(bounds), ARD=True),
          'f': None,  # 最適化する関数の設定（実験結果は分からないので設定しない）．
          'domain': bounds,  # パラメータの探索範囲の指定
          'model_type': 'GP',  # 標準的なガウシアンプロセスを指定．
          'X': X_step,  # 既知データの説明変数（インプットするx）
          'Y': Y_step,  # 既知データの目的変数（インプットするy）
          'de_duplication': True,  # 重複したデータをサンプルしないように設定．
          "normalize_Y": True,  # defaltはTrue
          "exact_feval": False  # defaltはFalse
          }
# ベイズ最適化のモデルの作成
bo_step = GPyOpt.methods.BayesianOptimization(**params)

# 次の候補点のx,y(予測値,分散)の計算
x_suggest = bo_step.suggest_next_locations(ignored_X=X_step)
y_predict = bo_step.model.model.predict(
    x_suggest)  # y_predictは(予測平均，予測分散)がタプルで返ってくる
y_mean = y_predict[0]
y_variance = y_predict[1]
# 次の実験候補点のxについて標準化処理をしない状態のスケールで表示
for j in range(0, num_x):
    exec("nextx.append(x"+str(j) +
         "_min+(x_suggest[0,"+str(j)+"])*((x"+str(j)+"_max)-(x"+str(j)+"_min)))")

print("次のブランデーは " + str(round(nextx[0], 3)) + "mlです")
print("次のカルーアは " + str(round(nextx[1], 3)) + "mlです")
print("次のクリームは " + str(round(nextx[2], 3)) + "mlです")

# y_meanとy_varianceはndarrayなのでfloatに変換
print("次の実験点でのmeanは " + "{:.4f}".format(float(y_mean)) + "です")
print("次の実験点でのstdは " + "{:.4f}".format(float(y_variance)) + "です")

bo_step.plot_acquisition()
