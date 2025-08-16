
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="心脏病预测 · Streamlit", page_icon="❤️", layout="centered")

st.title("❤️ 心脏病预测（Streamlit 版）")
st.caption("建议：统一使用列名 **thalch**（最大心率）。本应用优先加载仓库根目录下的模型文件。")

# ---------- 1) 选择与加载模型（优先 best_model_pipe.pkl，回退 best_model.pkl；否则允许上传） ----------
BASE_DIR = Path(__file__).resolve().parent
CANDIDATES = [BASE_DIR / "best_model_pipe.pkl", BASE_DIR / "best_model.pkl"]

st.subheader("① 加载模型")
loaded = None
loaded_path = None
for p in CANDIDATES:
    if p.exists():
        loaded = joblib.load(p)
        loaded_path = p
        break

upload_file = None
if loaded is None:
    upload_file = st.file_uploader("未在仓库根目录找到模型文件。请上传：best_model_pipe.pkl 或 best_model.pkl", type=["pkl"])
    if upload_file:
        loaded = joblib.load(upload_file)
        loaded_path = Path(upload_file.name)

st.write("📄 模型来源：", str(loaded_path) if loaded_path else "（未找到，依赖上传）")
st.write("🧠 对象类型：", type(loaded).__name__ if loaded is not None else "None")

if loaded is None or not hasattr(loaded, "predict"):
    st.error("当前加载的对象不是可预测的模型/Pipeline。请上传/放置正确的模型文件（建议为包含预处理的 sklearn Pipeline）。")
    st.stop()

model = loaded

# ---------- 2) 推断特征名：优先使用模型的 feature_names_in_；否则允许提供 CSV ----------
st.subheader("② 定义输入特征")
feature_names = None
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)

csv_df = None
if feature_names is None:
    st.info("模型未提供 feature_names_in_。请上传一个示例 CSV 用于构建表单。")
    csv_up = st.file_uploader("上传示例 CSV（列名将用于生成输入表单）", type=["csv"], key="csv_schema")
    if csv_up:
        try:
            csv_df = pd.read_csv(csv_up)
            feature_names = list(csv_df.columns)
        except Exception as e:
            st.error(f"CSV 读取失败：{e}")

if not feature_names:
    st.error("无法确定特征名。请：1）使用带 feature_names_in_ 的模型导出；或 2）提供包含完整列名的 CSV。")
    st.stop()

# ---------- 3) 构建表单（数值/枚举自动判断；确保支持 thalch） ----------
st.subheader("③ 填写特征值")
st.sidebar.header("填写特征")
st.sidebar.caption("数值列设置分位数范围；枚举列用下拉。")

# 若有 CSV，用它来估计合理范围/默认值
schema_df = csv_df

def build_default(series: pd.Series):
    if series is None or series.empty:
        return 0 if series is None else (series.median() if pd.api.types.is_numeric_dtype(series) else "")
    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())
    try:
        return series.mode().iloc[0]
    except Exception:
        return ""

def widget_for(col: str):
    series = schema_df[col] if (schema_df is not None and col in schema_df.columns) else None
    is_num = pd.api.types.is_numeric_dtype(series) if series is not None else True
    if is_num:
        if series is not None and not series.empty:
            q01 = float(series.quantile(0.01))
            q99 = float(series.quantile(0.99))
            default = build_default(series)
            non_na = series.dropna()
            is_int_like = len(non_na) > 0 and all(float(x).is_integer() for x in non_na.iloc[: min(200, len(non_na))])
            step = 1.0 if is_int_like else 0.1
        else:
            q01, q99, default, step = 0.0, 100.0, 0.0, 1.0
        return st.sidebar.number_input(col, min_value=q01, max_value=q99, value=float(default), step=step, key=f"num_{col}")
    else:
        opts = series.dropna().unique().tolist() if (series is not None and not series.empty) else []
        if 0 < len(opts) <= 15:
            default = build_default(series) if series is not None else (opts[0] if opts else "")
            idx = opts.index(default) if default in opts else 0
            return st.sidebar.selectbox(col, options=opts or [""], index=idx, key=f"sel_{col}")
        else:
            default = build_default(series) if series is not None else ""
            return st.sidebar.text_input(col, value=str(default), key=f"txt_{col}")

inputs = {c: widget_for(c) for c in feature_names}
X = pd.DataFrame([inputs], columns=feature_names)

st.markdown("**你的输入**")
st.dataframe(X, use_container_width=True)

# ---------- 4) 预测 ----------
st.subheader("④ 预测")
if st.sidebar.button("开始预测", type="primary", use_container_width=True):
    try:
        y_pred = model.predict(X)
        st.success(f"预测类别：**{y_pred[0]}**")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                st.info(f"阳性概率：**{proba[0,1]:.3f}**；阴性概率：{proba[0,0]:.3f}")
            else:
                st.write("各类别概率：")
                st.dataframe(pd.Series(proba[0]).to_frame("probability"))
    except Exception as e:
        st.error(f"预测失败：{e}")
        st.caption("多半是列名/类型不一致。请确保表单列名与训练时一致（统一使用 thalch），并使用包含预处理的 Pipeline。")

st.divider()
with st.expander("⚙️ 使用与部署提示"):
    st.markdown("""
- **模型文件优先顺序**：`best_model_pipe.pkl` → `best_model.pkl` → 上传。
- **强烈建议**：把预处理与模型一起封装为 `sklearn.pipeline.Pipeline` 再导出。
- **部署**：将 `app.py`、`requirements.txt`、模型文件放到 GitHub 仓库根目录，Streamlit Community Cloud 选择该仓库即可。
- **列名**：请统一用 `thalch` 作为最大心率字段名（不是 `thalach`）。
""")
