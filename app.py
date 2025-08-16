
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(page_title="心脏病预测 · Web Demo", page_icon="❤️", layout="centered")

st.title("❤️ 心脏病预测（Streamlit Cloud 版）")
st.markdown(
    """
    - 优先从仓库根目录加载 `best_model.pkl` 与 `heart_disease_uci .csv`；
    - 如果不存在，你也可以在侧边栏**临时上传**它们；
    - 表单根据特征列自动生成，点击 **预测** 查看结果。
    """
)

# === 期望的默认文件名（与仓库中的文件保持一致）===
MODEL_DEFAULT = Path("best_model.pkl")
CSV_DEFAULT = Path("heart_disease_uci .csv")  # 注意：文件名里有空格

st.sidebar.header("📂 加载文件")
st.sidebar.caption("如果仓库里没有对应文件，可在此处临时上传。")

# --- 模型加载 ---
model_bytes = None
model_file = None
if MODEL_DEFAULT.exists():
    model_file = MODEL_DEFAULT.open("rb")
else:
    model_bytes = st.sidebar.file_uploader("上传模型（.pkl）", type=["pkl"], key="model")

# --- CSV 加载（用于生成表单与默认值） ---
csv_df = None
if CSV_DEFAULT.exists():
    try:
        csv_df = pd.read_csv(CSV_DEFAULT)
    except Exception as e:
        st.warning(f"仓库内 CSV 读取失败：{e}")
else:
    csv_file = st.sidebar.file_uploader("上传示例 CSV（用于推断列 & 取值范围）", type=["csv"], key="csv")
    if csv_file is not None:
        try:
            csv_df = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"上传的 CSV 读取失败：{e}")

@st.cache_resource(show_spinner=False)
def load_model(_file_like, _path: Path):
    if _file_like is not None:
        return pickle.load(_file_like)
    elif _path.exists():
        with _path.open("rb") as f:
            return pickle.load(f)
    else:
        return None

model = load_model(model_bytes, MODEL_DEFAULT)

if model is None:
    st.error("未找到可用模型。请将 `best_model.pkl` 放在仓库根目录，或在侧边栏上传模型文件。")
    st.stop()

# 推断特征名
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
elif csv_df is not None:
    feature_names = list(csv_df.columns)
else:
    st.error("无法确定特征名。请提供包含完整列名的 CSV，或使用带 `feature_names_in_` 的模型（建议保存为含预处理的 Pipeline）。")
    st.stop()

st.sidebar.header("🧮 填写特征值")
st.sidebar.caption("数值列使用分位数范围；分类列提供下拉选项（若唯一值不多）。")

def build_default(series: pd.Series):
    if series is None or series.empty:
        return ""
    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())
    try:
        return series.mode().iloc[0]
    except Exception:
        return ""

def widget_for_column(col: str):
    col_series = csv_df[col] if (csv_df is not None and col in csv_df.columns) else None
    is_numeric = pd.api.types.is_numeric_dtype(col_series) if col_series is not None else True

    if is_numeric:
        if col_series is not None and not col_series.empty:
            q01 = float(col_series.quantile(0.01))
            q99 = float(col_series.quantile(0.99))
            default_val = build_default(col_series)
            # 判断是否为整数型
            non_na = col_series.dropna()
            is_int_like = len(non_na) > 0 and all(float(x).is_integer() for x in non_na[: min(200, len(non_na))])
            step = 1.0 if is_int_like else 0.1
        else:
            q01, q99, default_val, step = 0.0, 100.0, 0.0, 1.0
        return st.sidebar.number_input(f"{col}", min_value=q01, max_value=q99, value=float(default_val), step=step, key=f"num_{col}")
    else:
        if col_series is not None and not col_series.empty:
            uniques = col_series.dropna().unique().tolist()
        else:
            uniques = []
        if 0 < len(uniques) <= 15:
            default_val = build_default(col_series) if col_series is not None else (uniques[0] if uniques else "")
            idx = uniques.index(default_val) if default_val in uniques else 0
            return st.sidebar.selectbox(f"{col}", options=uniques or [""], index=idx, key=f"cat_{col}")
        else:
            default_val = build_default(col_series) if col_series is not None else ""
            return st.sidebar.text_input(f"{col}", value=str(default_val), key=f"text_{col}")

# 构建输入
user_input = {col: widget_for_column(col) for col in feature_names}
X = pd.DataFrame([user_input], columns=feature_names)

st.markdown("### 你的输入")
st.dataframe(X, use_container_width=True)

# 预测
if st.sidebar.button("预测", type="primary", use_container_width=True):
    try:
        y_pred = model.predict(X)
        st.markdown("### 预测结果")
        st.success(f"预测类别：**{y_pred[0]}**")

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                if proba.shape[1] == 2:
                    st.info(f"阳性概率：**{proba[0,1]:.3f}**；阴性概率：{proba[0,0]:.3f}")
                else:
                    prob_series = pd.Series(proba[0], index=getattr(model, "classes_", list(range(proba.shape[1]))))
                    st.write("各类别概率：")
                    st.dataframe(prob_series.to_frame("probability"))
            except Exception:
                pass

        st.caption("若结果异常，请确认模型是否包含预处理（建议保存为包含编码/标准化等步骤的 scikit-learn Pipeline）。")
    except Exception as e:
        st.error(f"预测失败：{e}")
        st.caption("常见原因：输入类型与训练不一致、缺少预处理、列名不匹配等。建议将预处理与模型一起用 Pipeline 保存，再导出。")

st.divider()
with st.expander("⚙️ 部署与使用说明", expanded=False):
    st.markdown(
        """
        **仓库结构建议：**
        ```
        ├─ app.py
        ├─ requirements.txt
        ├─ best_model.pkl
        └─ heart_disease_uci .csv
        ```
        **本地运行：**
        ```bash
        pip install -r requirements.txt
        streamlit run app.py
        ```
        **在线部署：**
        - 打开 [streamlit.io](https://streamlit.io/cloud)，Create app → 选你的仓库与分支 → 指定命令 `streamlit run app.py`
        - 如果模型较大，建议使用 Git LFS。
        - 模型依赖了额外库（如 xgboost/lightgbm/catboost），请把对应包加入 `requirements.txt`。
        """
    )
